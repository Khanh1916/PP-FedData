#!/usr/bin/env python3
"""Extract NormalData PCAP files into split CSV files using TShark.

Default input:
  <script directory>/DoS-DDoS-MQTT-IoT_Dataset/NormalData

Default output:
  <dataset>/NormalData/CSV_Extracted_Split

Each CSV has at most 100,000 packet rows by default. The first 33 columns
follow the field order of the Attack CSV schema supplied for this project.
Two metadata columns are appended: Label and Capture_ID.

This is packet-level extraction, not flow-level extraction. Validate the
exported values and schema against the original Attack CSVs before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "DoS-DDoS-MQTT-IoT_Dataset"
PCAP_EXTENSIONS = {".pcap", ".pcapng", ".cap"}

# Output-column name -> TShark field.
# The order is based on the 33 columns observed in the supplied Attack CSVs.
FIELDS = [
    ("No.", "frame.number"),
    ("Message Type", "mqtt.msgtype"),
    ("QoS Level", "mqtt.qos"),
    ("QoS Level.1", "mqtt.conflag.qos"),
    ("Requested QoS", "mqtt.sub.qos"),
    ("Epoch Time", "frame.time_epoch"),
    ("Protocol", "_ws.col.protocol"),
    ("Source", "_ws.col.Source"),
    ("Frame length on the wire", "frame.len"),
    ("Time delta from previous displayed frame", "frame.time_delta_displayed"),
    ("Time since reference or first frame", "frame.time_relative"),
    ("Frame length on the wire.1", "frame.len"),
    ("Stream index", "tcp.stream"),
    ("iRTT", "tcp.analysis.initial_rtt"),
    ("Time since first frame in this TCP stream", "tcp.time_relative"),
    ("TCP Segment Len", "tcp.len"),
    ("Calculated window size", "tcp.window_size"),
    ("Syn", "tcp.flags.syn"),
    ("Reset", "tcp.flags.reset"),
    ("Acknowledgment", "tcp.flags.ack"),
    ("Clean Session Flag", "mqtt.conflag.cleansess"),
    ("Keep Alive", "mqtt.kalive"),
    ("User Name Length", "mqtt.username_len"),
    ("Password Length", "mqtt.passwd_len"),
    ("Retain", "mqtt.retain"),
    ("Clean Session Flag.1", "mqtt.conflag.cleansess"),
    ("Will Retain", "mqtt.conflag.retain"),
    ("Will Flag", "mqtt.conflag.willflag"),
    ("Will Message Length", "mqtt.willmsg_len"),
    ("Will Topic Length", "mqtt.willtopic_len"),
    ("Topic Length", "mqtt.topic_len"),
    ("Msg Len", "mqtt.len"),
    ("Info", "_ws.col.info"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET,
        help="Dataset root containing the NormalData folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: NormalData/CSV_Extracted_Split).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100_000,
        help="Maximum data rows per CSV (default: 100000).",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        help="Decode this TCP port as MQTT; use 0 to disable (default: 1883).",
    )
    parser.add_argument(
        "--tshark",
        type=str,
        default=None,
        help="Path to tshark.exe if it is not available on PATH.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N PCAP files (useful for testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output parts for each processed PCAP.",
    )
    return parser.parse_args()


def find_tshark(explicit_path: str | None = None) -> str:
    """Find TShark on PATH or in common Windows installation folders."""
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.is_file():
            return str(candidate)
        found = shutil.which(explicit_path)
        if found:
            return found
        raise FileNotFoundError(f"Cannot find TShark: {explicit_path}")

    found = shutil.which("tshark")
    if found:
        return found

    candidates = [
        Path(r"C:\Program Files\Wireshark\tshark.exe"),
        Path(r"C:\Program Files (x86)\Wireshark\tshark.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        'Cannot find tshark.exe. Install Wireshark or use --tshark "path".'
    )


def list_pcaps(normal_dir: Path, output_dir: Path) -> list[Path]:
    """Recursively find PCAP files, excluding the output directory."""
    output_resolved = output_dir.resolve()
    found: list[Path] = []

    for path in normal_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in PCAP_EXTENSIONS:
            continue

        try:
            path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            pass

        found.append(path)

    return sorted(found, key=lambda p: str(p).casefold())


def prefix_for(pcap: Path, normal_dir: Path) -> str:
    """Build a filesystem-safe output prefix from the relative PCAP path."""
    relative = pcap.relative_to(normal_dir).with_suffix("")
    raw = "__".join(relative.parts)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")


def make_tshark_command(
    tshark: str, pcap: Path, mqtt_port: int
) -> list[str]:
    """Construct a TShark command that streams packet fields to stdout."""
    command = [tshark, "-n", "-r", str(pcap)]

    if mqtt_port:
        command.extend(["-d", f"tcp.port=={mqtt_port},mqtt"])

    command.extend(
        [
            "-T", "fields",
            "-E", "header=n",
            "-E", "separator=,",
            "-E", "quote=d",
            "-E", "escape=y",
            "-E", "occurrence=f",
        ]
    )

    for _, field in FIELDS:
        command.extend(["-e", field])

    return command


def validate_fields(tshark: str, pcap: Path, mqtt_port: int) -> None:
    """Check requested TShark fields on one packet before full extraction."""
    command = [tshark, "-n", "-r", str(pcap), "-c", "1"]

    if mqtt_port:
        command.extend(["-d", f"tcp.port=={mqtt_port},mqtt"])

    command.extend(
        [
            "-T", "fields",
            "-E", "header=n",
            "-E", "separator=,",
            "-E", "quote=d",
            "-E", "escape=y",
            "-E", "occurrence=f",
        ]
    )

    for _, field in FIELDS:
        command.extend(["-e", field])

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "TShark rejected one or more fields or could not read the sample "
            f"PCAP: {pcap}\n{result.stderr.strip()}"
        )


def existing_parts(output_dir: Path, prefix: str) -> list[Path]:
    return sorted(
        output_dir.glob(f"{prefix}_part[0-9][0-9][0-9].csv")
    )


def terminate_process(proc: subprocess.Popen[str]) -> None:
    """Stop TShark safely when extraction is interrupted or fails."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def extract_pcap(
    tshark: str,
    pcap: Path,
    normal_dir: Path,
    output_dir: Path,
    max_rows: int,
    mqtt_port: int,
    overwrite: bool,
) -> dict[str, object]:
    """Stream one PCAP to CSV parts, each with at most max_rows records."""
    prefix = prefix_for(pcap, normal_dir)
    marker = output_dir / f"{prefix}.done.json"
    parts = existing_parts(output_dir, prefix)
    capture_id = pcap.relative_to(normal_dir).as_posix()

    if marker.exists() and not overwrite:
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
            print(
                f"Skip completed: {pcap.name} "
                f"({previous.get('rows', '?')} rows)"
            )
            return {
                "PCAP": capture_id,
                "Rows": previous.get("rows", ""),
                "Parts": previous.get("parts", ""),
                "Status": "Skipped complete",
                "Error": "",
            }
        except (OSError, json.JSONDecodeError):
            print(
                f"Invalid completion marker for {pcap.name}; "
                "use --overwrite to re-extract."
            )
            return {
                "PCAP": capture_id,
                "Rows": "",
                "Parts": len(parts),
                "Status": "Skipped incomplete",
                "Error": "Invalid completion marker",
            }

    if parts and not overwrite:
        print(
            f"Partial output exists for {pcap.name}; "
            "use --overwrite to replace it."
        )
        return {
            "PCAP": capture_id,
            "Rows": "",
            "Parts": len(parts),
            "Status": "Skipped incomplete",
            "Error": "Rerun with --overwrite to replace partial output",
        }

    if overwrite:
        for path in parts:
            path.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)

    command = make_tshark_command(tshark, pcap, mqtt_port)
    headers = [name for name, _ in FIELDS] + ["Label", "Capture_ID"]

    created_parts: list[Path] = []
    current_file = None
    current_path: Path | None = None
    writer = None
    total_rows = 0
    rows_in_part = 0
    part_number = 0

    # A temporary file prevents TShark stderr from blocking while stdout streams.
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as error_file:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=error_file,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        try:
            if proc.stdout is None:
                raise RuntimeError("Could not open TShark stdout.")

            reader = csv.reader(proc.stdout)
            for row in reader:
                if len(row) < len(FIELDS):
                    row.extend([""] * (len(FIELDS) - len(row)))
                elif len(row) > len(FIELDS):
                    raise ValueError(
                        f"TShark returned {len(row)} fields; "
                        f"expected {len(FIELDS)}."
                    )

                if current_file is None or rows_in_part >= max_rows:
                    if current_file is not None:
                        current_file.close()
                    part_number += 1
                    rows_in_part = 0

                    current_path = output_dir / (
                        f"{prefix}_part{part_number:03d}.csv"
                    )
                    current_file = current_path.open(
                        "w", newline="", encoding="utf-8-sig"
                    )
                    writer = csv.writer(current_file)
                    writer.writerow(headers)
                    created_parts.append(current_path)

                assert writer is not None
                writer.writerow(row + ["Normal", capture_id])
                rows_in_part += 1
                total_rows += 1

                if total_rows % 500_000 == 0:
                    print(f"    Exported {total_rows:,} rows...")

            proc.stdout.close()
            return_code = proc.wait()

        except BaseException:
            terminate_process(proc)
            if current_file is not None:
                current_file.close()
                current_file = None
            for path in created_parts:
                path.unlink(missing_ok=True)
            raise

        finally:
            if current_file is not None:
                current_file.close()
            if proc.stdout is not None and not proc.stdout.closed:
                proc.stdout.close()

        error_file.seek(0)
        error_text = error_file.read()

    if return_code != 0:
        for path in created_parts:
            path.unlink(missing_ok=True)
        raise RuntimeError(
            f"TShark exited with code {return_code}:\n"
            f"{error_text.strip()[-4000:]}"
        )

    # Mark completion only after all CSV parts have been written successfully.
    marker_data = {
        "pcap": capture_id,
        "rows": total_rows,
        "parts": part_number,
        "max_rows_per_file": max_rows,
        "label": "Normal",
        "columns": headers,
    }
    temporary_marker = marker.with_name(marker.name + ".tmp")
    temporary_marker.write_text(
        json.dumps(marker_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_marker.replace(marker)

    print(
        f"Exported {total_rows:,} rows into {part_number} CSV part(s): "
        f"{pcap.name}"
    )

    return {
        "PCAP": capture_id,
        "Rows": total_rows,
        "Parts": part_number,
        "Status": "OK",
        "Error": "",
    }


def save_summary(
    output_dir: Path, results: list[dict[str, object]]
) -> Path:
    """Write a CSV summary for the current extraction run."""
    path = output_dir / "normal_extraction_summary.csv"
    columns = ["PCAP", "Rows", "Parts", "Status", "Error"]

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    return path


def main() -> int:
    args = parse_args()

    if args.max_rows <= 0:
        print("ERROR: --max-rows must be greater than zero.")
        return 2
    if args.limit is not None and args.limit <= 0:
        print("ERROR: --limit must be greater than zero.")
        return 2
    if not 0 <= args.mqtt_port <= 65535:
        print("ERROR: --mqtt-port must be between 0 and 65535.")
        return 2

    dataset_dir = args.dataset_dir.expanduser().resolve()
    normal_dir = dataset_dir / "NormalData"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else normal_dir / "CSV_Extracted_Split"
    )

    if not normal_dir.is_dir():
        print(f"ERROR: NormalData folder not found: {normal_dir}")
        return 2

    try:
        tshark = find_tshark(args.tshark)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    pcaps = list_pcaps(normal_dir, output_dir)
    if args.limit is not None:
        pcaps = pcaps[:args.limit]

    if not pcaps:
        print(f"No PCAP files found under: {normal_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(pcaps)} PCAP file(s).")
    print(f"TShark: {tshark}")
    print(f"Output: {output_dir}")
    print(f"Maximum rows per CSV: {args.max_rows:,}")

    print("\nChecking TShark fields on the first PCAP...")
    try:
        validate_fields(tshark, pcaps[0], args.mqtt_port)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR during TShark preflight:\n{exc}")
        return 2
    print("TShark field check passed.")

    results: list[dict[str, object]] = []
    try:
        for index, pcap in enumerate(pcaps, 1):
            print(f"\n[{index}/{len(pcaps)}] {pcap.name}")
            try:
                result = extract_pcap(
                    tshark=tshark,
                    pcap=pcap,
                    normal_dir=normal_dir,
                    output_dir=output_dir,
                    max_rows=args.max_rows,
                    mqtt_port=args.mqtt_port,
                    overwrite=args.overwrite,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"ERROR: {exc}")
                result = {
                    "PCAP": pcap.relative_to(normal_dir).as_posix(),
                    "Rows": "",
                    "Parts": "",
                    "Status": "ERROR",
                    "Error": str(exc),
                }
            results.append(result)

    except KeyboardInterrupt:
        print(
            "\nInterrupted. The incomplete parts for the current PCAP "
            "were removed."
        )
        if results:
            print(f"Summary: {save_summary(output_dir, results)}")
        return 130

    summary = save_summary(output_dir, results)
    successful = sum(r["Status"] == "OK" for r in results)
    skipped = sum(str(r["Status"]).startswith("Skipped") for r in results)
    failed = sum(r["Status"] == "ERROR" for r in results)
    total_rows = sum(
        int(r["Rows"])
        for r in results
        if isinstance(r["Rows"], int)
    )

    print("\n" + "=" * 65)
    print("FINISHED")
    print(f"Processed successfully in this run: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Rows exported in this run: {total_rows:,}")
    print(f"Summary CSV: {summary}")

    if failed or skipped:
        print("Check Status/Error in the summary CSV for unfinished PCAPs.")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

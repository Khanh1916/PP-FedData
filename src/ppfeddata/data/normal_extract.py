"""Re-extraction of the Normal PCAP with the authors' export settings (round R, option A of the provenance check).

The release has no Normal CSV. The group's first extraction (`tools/extract_normal_pcap.py`, folder NormalData/CSV_Extracted_Split)
kept every packet, in one pass. The authors exported their attack CSV with two-pass dissection and a display filter that keeps the
packets sent to the broker (`check-extraction`, section 4 of results/reports/data_provenance.md). This module runs the same script with
these two settings added into a folder of its own (`OUT_DIR`), so that the Normal and the attack rows come from the same export; the
first extraction stays untouched. The script itself is not changed: its command builder is wrapped for the run only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ppfeddata.data.extraction_check import load_extractor, with_options

logger = logging.getLogger("ppfeddata.normal_extract")

AUTHOR_SETTINGS = ("-2", "-Y", "ip.dst=={broker}")      # the settings that reproduce 47 of the 48 attack CSV (the 48th used `ip`)
OUT_DIR = "CSV_Authors_Settings"


def authors_options(broker: str) -> tuple[str, ...]:
    return tuple(o.format(broker=broker) for o in AUTHOR_SETTINGS)


def extract_all(raw_root: Path, extractor_path: str | Path, broker: str, out_name: str = OUT_DIR, max_rows: int = 100_000,
                mqtt_port: int = 1883) -> list[dict[str, Any]]:
    """Every PCAP of NormalData/Pcap Files into NormalData/<out_name>, in parts of `max_rows` rows with the script's completion markers.
    Resumable: a capture with a marker is skipped, one left half-written by an interruption is extracted again."""
    ext = load_extractor(extractor_path)
    tshark = ext.find_tshark(None)
    nd = Path(raw_root) / "NormalData"
    out = nd / out_name
    out.mkdir(parents=True, exist_ok=True)
    pcaps = sorted((nd / "Pcap Files").glob("*.pcap"), key=lambda p: str(p).casefold())
    if not pcaps:
        raise FileNotFoundError(f"no PCAP under {nd / 'Pcap Files'}")
    own = ext.make_tshark_command
    ext.make_tshark_command = with_options(own, authors_options(broker))
    results = []
    try:
        for i, pcap in enumerate(pcaps, 1):
            r = ext.extract_pcap(tshark, pcap, nd, out, max_rows, mqtt_port, False)
            if r.get("Status") == "Skipped incomplete":
                r = ext.extract_pcap(tshark, pcap, nd, out, max_rows, mqtt_port, True)
            logger.info("[%d/%d] %s: %s rows (%s)", i, len(pcaps), pcap.name, r.get("Rows"), r.get("Status"))
            results.append(r)
    finally:
        ext.make_tshark_command = own
    ext.save_summary(out, results)
    return results

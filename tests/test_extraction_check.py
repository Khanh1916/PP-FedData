"""Provenance check of the raw data (data/extraction_check.py): pairing of CSV and PCAP names, the column comparison, who sent a packet."""
import numpy as np
import pandas as pd

from ppfeddata.data import extraction_check as ec

DEC = {"msg_len": {"action": "numeric"}, "syn": {"action": "binary"}, "message_type": {"action": "categorical"}, "info": {"action": "drop"}}


def test_names_of_one_capture_pair_up():
    assert ec._key("SYN_DDoS-AD_1") == ec._key("SYN-DDoS-AD_1") == "syn_ddos_ad_1"
    assert ec._key("BF1_DDoS_AD_13 ") == ec._key("BF1_DDoS_AD_13")


def test_attack_pairs_find_the_pcap_and_flag_empty_csv(tmp_path):
    for sub in ("CSV Files", "Pcap Files"):
        (tmp_path / "SYN TCP Flooding" / "DDoS" / sub).mkdir(parents=True)
    (tmp_path / "SYN TCP Flooding" / "DDoS" / "CSV Files" / "SYN_DDoS-AD_1.csv").write_text("No.\n1\n")
    (tmp_path / "SYN TCP Flooding" / "DDoS" / "CSV Files" / "SYN_DDoS-AD_2.csv").write_text("")
    (tmp_path / "SYN TCP Flooding" / "DDoS" / "Pcap Files" / "SYN-DDoS-AD_1.pcap").write_bytes(b"x")
    p = {r["csv"].stem: r for r in ec.attack_pairs(tmp_path)}
    assert p["SYN_DDoS-AD_1"]["pcap"].name == "SYN-DDoS-AD_1.pcap" and not p["SYN_DDoS-AD_1"]["empty"]
    assert p["SYN_DDoS-AD_2"]["pcap"] is None and p["SYN_DDoS-AD_2"]["empty"]
    assert p["SYN_DDoS-AD_1"]["scenario"] == "SYN TCP Flooding" and p["SYN_DDoS-AD_1"]["attack_type"] == "DDoS"


def test_roles_follow_the_feature_decisions():
    r = ec.roles(DEC, ["msg_len", "syn", "message_type", "info", "no", "clean_session_flag_1"])
    assert r == {"msg_len": "numeric", "syn": "binary", "message_type": "categorical", "info": "text", "no": "numeric", "clean_session_flag_1": "binary"}


def test_compare_counts_values_equal_after_the_pipeline_parsers():
    official = pd.DataFrame({"No.": ["1", "2", "3"], "Message Type": ["Publish Message,Publish Message", "Connect Command", ""],
                             "Syn": ["Not set", "Set", "Not set"], "Msg Len": ["40,41", "12", ""], "Info": ["a", "b", "c"]})
    ours = pd.DataFrame({"No.": ["1", "2", "3", "4"], "Message Type": ["3", "1", "", "2"], "Syn": ["0", "1", "1", "0"],
                         "Msg Len": ["40", "12", "", "9"], "Info": ["a", "B", "c", "d"], "Label": ["Normal"] * 4, "Capture_ID": ["x"] * 4})
    c = ec.compare(official, ours, DEC)
    assert (c["rows_official"], c["rows_ours"], c["rows_common"], c["only_ours"]) == (3, 4, 3, 1)
    col = c["columns"]
    assert col["message_type"]["match"] == 1.0 and col["message_type"]["raw_match"] < 1      # labels vs codes: equal once parsed
    assert col["msg_len"]["match"] == 1.0                                                   # first of "40,41"
    assert np.isclose(col["syn"]["match"], 2 / 3) and col["syn"]["mismatch_rows"] == 1
    assert np.isclose(col["info"]["match"], 2 / 3) and not col["info"]["used"] and col["msg_len"]["used"]


def test_sender_and_display_filter_are_recognised():
    ours = pd.DataFrame({"No.": ["1", "2", "3", "4", "5"], "Source": ["10.0.0.2", "10.0.0.9", "aa:bb:cc:dd:ee:ff", "10.0.0.2", "10.0.0.9"],
                         "Info": ["5000 → 1883 [SYN]", "1883 → 5000 [SYN, ACK]", "Conf. Root", "5000 → 1883 [ACK]", "Connect Ack"]})
    assert ec.broker_of(ours) == "10.0.0.9"
    assert ec.population(ours, "10.0.0.9").tolist() == ["client", "broker", "non-IP", "client", "broker"]
    official = pd.DataFrame({"No.": ["1", "4"], "Epoch Time": ["100.0", "100.5"], "Time delta from previous displayed frame": ["0.0", "0.5"]})
    k = ec.kept_population(official, ours)
    assert k["by_sender"] == {"client": {"packets": 2, "kept": 2}, "broker": {"packets": 2, "kept": 0}, "non-IP": {"packets": 1, "kept": 0}}
    assert k["delta_is_gap_to_previous_kept"] == 1.0


def test_extra_tshark_options_go_before_the_input_file():
    own = lambda tshark, pcap, port: [tshark, "-n", "-r", str(pcap), "-d", f"tcp.port=={port},mqtt", "-T", "fields"]       # noqa: E731
    cmd = ec.with_options(own, ("-2", "-Y", "ip.dst==10.0.0.9"))("tshark", "a.pcap", 1883)
    assert cmd == ["tshark", "-n", "-2", "-Y", "ip.dst==10.0.0.9", "-r", "a.pcap", "-d", "tcp.port==1883,mqtt", "-T", "fields"]


def test_extract_restores_the_script_after_extra_options(tmp_path):
    seen = []

    class Script:                                    # stands for tools/extract_normal_pcap.py
        @staticmethod
        def make_tshark_command(tshark, pcap, port):
            return [tshark, "-n", "-r", str(pcap)]

        def extract_pcap(self, tshark, pcap, normal_dir, out_dir, max_rows, port, overwrite):
            seen.append(self.make_tshark_command(tshark, pcap, port))
            (out_dir / "x_part001.csv").write_text("No.\n1\n")
            return {"Status": "OK"}

    s = Script()
    own = s.make_tshark_command
    parts = ec.extract(s, "tshark", tmp_path / "a.pcap", tmp_path / "out", options=("-2",))
    assert seen[0] == ["tshark", "-n", "-2", "-r", str(tmp_path / "a.pcap")] and [p.name for p in parts] == ["x_part001.csv"]
    assert s.make_tshark_command == own


def test_local_files_are_matched_with_the_release(tmp_path):
    root = tmp_path / "MyCopy"
    (root / "NormalData" / "Pcap Files").mkdir(parents=True)
    (root / "NormalData" / "CSV_Extracted_Split").mkdir(parents=True)
    (root / "NormalData" / "Pcap Files" / "N1.pcap").write_bytes(b"12345")
    (root / "NormalData" / "Pcap Files" / "N2.pcap").write_bytes(b"12")
    (root / "NormalData" / "CSV_Extracted_Split" / "N1_part001.csv").write_text("x")
    listing = [("Release/NormalData/Pcap Files/N1.pcap", 5), ("Release/NormalData/Pcap Files/N2.pcap", 3), ("Release/SYN/DoS/CSV Files/a.csv", 1)]
    m = ec.match_listing(listing, root)
    assert m["files"] == 3 and m["by_kind"]["Normal pcap"] == {"same size": 1, "size differs": 1, "missing": 0}
    assert m["by_kind"]["attack csv"]["missing"] == 1 and m["local_csv_pcap_not_in_release"] == {"NormalData/CSV_Extracted_Split": 1}


def test_broker_and_normal_time_delta(tmp_path):
    assert ec.broker_from({"method": [{"kept": {"broker": "10.0.0.9"}}], "normal_csv": [{"broker": "10.0.0.9"}, {"broker": "10.0.0.8"}]}) == "10.0.0.9"
    assert ec.broker_from({}) is None
    pd.DataFrame({"Epoch Time": ["0.0", "0.001", "0.002", "0.004"], "Source": ["10.0.0.2", "10.0.0.9", "10.0.0.2", "10.0.0.2"],
                  "Protocol": ["TCP", "MQTT", "MQTT", "TCP"], "Time delta from previous displayed frame": ["0", "0.001", "0.001", "0.002"]}
                 ).to_csv(tmp_path / "Pcap_Files__N1_part001.csv", index=False)
    r = ec.normal_time_delta(tmp_path, "10.0.0.9", ("N1", "N2"))
    assert len(r) == 1 and r[0]["client_rows"] == 3
    assert np.isclose(r[0]["quartiles_s_now"][1], 0.001) and np.isclose(r[0]["quartiles_s_authors"][1], 0.002)       # the broker's packet in between


def test_report_renders_the_three_checks():
    col = {"msg_len": {"role": "numeric", "used": True, "raw_match": 0.9, "match": 1.0, "mismatch_rows": 0}}
    out = {"versions": {"tshark": "TShark 4.4.7"}, "extraction_command": "tshark -n -r <pcap>",
           "method": [{"capture": "X_1", "status": "compared", "packets": 10, "rows_official": 6, "rows_ours": 10, "rows_common": 6, "columns": col,
                       "kept": {"broker": "10.0.0.9", "by_sender": {"client": {"packets": 6, "kept": 6}, "broker": {"packets": 4, "kept": 0},
                                                                    "non-IP": {"packets": 0, "kept": 0}}, "delta_is_gap_to_previous_kept": 1.0}},
                      {"capture": "X_2", "status": "empty official CSV"}],
           "normal_csv": [{"pcap": "Pcap Files/N1.pcap", "parts": 1, "parts_recorded": 1, "rows": 10, "rows_recorded": 10, "frames_1_to_n": True,
                           "labels": ["Normal"], "capture_ids": ["Pcap Files/N1.pcap"], "by_sender": {"client": 6, "broker": 4, "non-IP": 0},
                           "by_sender_tcp_mqtt": {"client": 6, "broker": 4, "non-IP": 0}}],
           "normal_pcap": {"packets": [{"pcap": "Pcap Files/N1.pcap", "present": True, "packets": 10, "rows": 10, "equal": True}],
                           "reextract": [{"pcap": "Pcap Files/N1.pcap", "parts_old": 1, "parts_new": 1, "byte_identical": True}]}}
    md = ec.render(out)
    assert "| X_1 | 10 | 6 | 10 | 6 | 100.00 % |" in md and "| X_2 | - | - | - | - | empty official CSV |" in md
    assert "| broker | 4 | 0 | 0.0 % |" in md and "| `msg_len` | numeric | yes | 100.000 % | 90.0 % |" in md
    assert "**1 of 1** captures" in md and "| broker | 4 | 40.0 % | 4 | 40.0 % |" in md
    assert "**1 of 1**" in md and "| Pcap Files/N1.pcap | 1 / 1 | yes |" in md and "TShark 4.4.7" in md
    assert "## 0." not in md and "## 4." not in md and "## 5." not in md                       # sections of the checks that did not run
    out["kaggle"] = {"dataset": "a/b", "files": 3, "bytes": 2e9, "by_kind": {"attack csv": {"same size": 2, "size differs": 0, "missing": 0}},
                     "issues": [], "local_csv_pcap_not_in_release": {"NormalData/CSV_Extracted_Split": 5}}
    out["recipe"] = [{"capture": "X_1", "filter": "ip.dst==10.0.0.9", "rows_common": 6, "rows_used_mismatch": 1, "rows_used_mismatch_retransmission": 1,
                      "columns": {"msg_len": {"role": "numeric", "used": True, "raw_match": 1.0, "match": 1.0, "mismatch_rows": 0}}}]
    out["impact"] = {"broker": "10.0.0.9", "broker_share": {"train": {"NORMAL": 0.44}, "test": {"NORMAL": 0.45}},
                     "irtt_on_syn_rows": {"NORMAL": {"syn_rows": 10, "irtt_missing": 1.0}},
                     "normal_time_delta": [{"capture": "N1", "client_rows": 3, "quartiles_s_now": [0.0005, 0.001, 0.0015], "quartiles_s_authors": [0.001, 0.002, 0.002],
                                            "median_ratio": 2.0}],
                     "test_rows": 10, "test_rows_client": 8, "train_rows_client": 6, "runs_scored": 2, "runs_skipped": {"no saved predictions": 0},
                     "configs": 2, "spearman": 1.0, "pairs": 1, "pairs_flipped": 0, "largest_gap_flipped": 0.0, "shift": {"mean": -0.02, "min": -0.03, "max": -0.01},
                     "by_config": [{"config": "B0-rf", "seeds": 3, "macro_f1": 0.45, "macro_f1_client": 0.43, "normal_recall_broker": 1.0, "normal_recall_client": 0.95},
                                   {"config": "Z-other", "seeds": 3, "macro_f1": 0.3, "macro_f1_client": 0.29, "normal_recall_broker": 1.0, "normal_recall_client": 0.9}],
                     "rf_retrain": [{"seed": 0, "train_all_test_all": 0.45, "train_all_test_client": 0.43, "train_client_test_client": 0.43}]}
    md = ec.render(out)
    assert "| attack csv | 2 | 0 | 0 |" in md and "`NormalData/CSV_Extracted_Split` 5" in md
    assert "| `ip.dst==10.0.0.9` | 1 (X_1) | 1 |" in md and "| `msg_len` | yes | 100.000 % | 100.000 % |" in md and "1 of 6" in md
    assert "| NORMAL | 44.0 % | 45.0 % |" in md and "| NORMAL | 10 | 100.0 % |" in md and "| N1 | 3 | 0.50 / 1.00 / 1.50 | 1.00 / 2.00 / 2.00 | 2.00 |" in md
    assert "mean -0.020" in md and "| B0-rf | 3 | 0.450 | 0.430 | 1.000 | 0.950 |" in md and "Z-other" not in md
    assert "| 0 | 0.450 | 0.430 | 0.430 |" in md


def test_limitation_reads_the_provenance_numbers():
    from ppfeddata.limitations import provenance_text

    assert "display filter" in provenance_text({}) and "left for later" in provenance_text({})
    pv = {"normal_pcap": {"packets": [{"present": True, "equal": True}] * 2},
          "recipe": [{"filter": "ip.dst==10.0.0.9"}, {"filter": "ip.dst==10.0.0.9"}, {"filter": "ip"}],
          "impact": {"broker_share": {"train": {"NORMAL": 0.442}}, "irtt_on_syn_rows": {"NORMAL": {"syn_rows": 5, "irtt_missing": 1.0}, "SYN": {"syn_rows": 5, "irtt_missing": 0.5}},
                     "normal_time_delta": [{"median_ratio": 1.01}, {"median_ratio": 1.04}], "runs_scored": 9, "spearman": 0.993, "largest_gap_flipped": 0.014,
                     "shift": {"mean": -0.019, "min": -0.029, "max": -0.009},
                     "rf_retrain": [{"train_all_test_client": 0.427, "train_client_test_client": 0.427}, {"train_all_test_client": 0.425, "train_client_test_client": 0.426}]}}
    t = provenance_text(pv)
    assert "2 of 2 captures" in t and "`ip.dst==10.0.0.9`" in t and "in 2 of 3 attack captures; `ip` in 1" in t and "3 of 3 captures" in t
    assert "44.2 %" in t and "1.01-1.04" in t and "100.0 %, against 50.0 %-50.0 %" in t and "-0.019" in t and "0.993" in t and "+0.000 to +0.001" in t

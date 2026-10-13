# `tools/` — scripts run once, outside the pipeline

| File | Role |
|---|---|
| `extract_normal_pcap.py` | the script that produced the **Normal CSV** of the raw data: the Kaggle release of the dataset has the authors' CSV for the attack captures but only PCAP for the 49 Normal captures, so every Normal packet was exported with TShark into the 33 columns of the attack CSV (plus `Label`, `Capture_ID`), in parts of at most 100,000 rows |

The file is kept exactly as it was run (SHA-256 `4a436c1bf1db0fa3f0302260b84421be5ea453f5f59f9eca2ad5ddd1a7453729`), so that
`python -m ppfeddata.cli check-extraction` tests the procedure that made the data, not a later copy of it: the check loads this file and
calls its own `extract_pcap`. The result is `results/reports/data_provenance.md`: the Normal CSV holds one row per packet of every
Normal PCAP and re-extracts byte for byte, and the same procedure run on the attack PCAP gives the values of the authors' CSV; the
differences that remain are the authors' export settings (a display filter and two-pass dissection), which the check reproduces.

Run it again (TShark 4.4 on the path, or `--tshark`; the dataset folder in the default place or `--dataset-dir`):

```bash
python tools/extract_normal_pcap.py --dataset-dir "D:/path/to/DoS-DDoS-MQTT-IoT_Dataset" --overwrite
```

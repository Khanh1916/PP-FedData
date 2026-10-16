# `tools/` — scripts run once, outside the pipeline

| File | Role |
|---|---|
| `extract_normal_pcap.py` | the script that produces the **Normal CSV** of the raw data: the Kaggle release of the dataset has the authors' CSV for the attack captures but only PCAP for the 49 Normal captures, so the Normal packets are exported with TShark into the 33 columns of the attack CSV (plus `Label`, `Capture_ID`), in parts of at most 100,000 rows (first study: every packet; round R: with the authors' export settings, through `extract-normal`) |

The file is kept exactly as it was run (SHA-256 `4a436c1bf1db0fa3f0302260b84421be5ea453f5f59f9eca2ad5ddd1a7453729`), so that
`python -m ppfeddata.cli check-extraction` tests the procedure that made the data, not a later copy of it: the check loads this file and
calls its own `extract_pcap`. The result is `results/reports/data_provenance.md`: the Normal CSV holds one row per packet of every
Normal PCAP and re-extracts byte for byte, and the same procedure run on the attack PCAP gives the values of the authors' CSV; the
differences that remain are the authors' export settings (a display filter and two-pass dissection), which the check reproduces.

Since round R the Normal CSV of the study comes from this script run with the authors' two export settings added (two-pass dissection, a
display filter on the packets sent to the broker): `python -m ppfeddata.cli extract-normal` (module `data/normal_extract.py`) wraps the
script's command builder for the run, writes to `NormalData/CSV_Authors_Settings` and leaves the script itself unchanged. The first
extraction below (every packet, one pass) is kept in `NormalData/CSV_Extracted_Split` as the reference of the provenance check.

Run the first extraction again (TShark 4.4 on the path, or `--tshark`; the dataset folder in the default place or `--dataset-dir`):

```bash
python tools/extract_normal_pcap.py --dataset-dir "D:/path/to/DoS-DDoS-MQTT-IoT_Dataset" --overwrite
```

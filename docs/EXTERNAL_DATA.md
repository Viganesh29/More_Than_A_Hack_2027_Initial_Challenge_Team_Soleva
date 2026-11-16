# External Data Attribution, Licensing, and Verification

## 1. Overview and Purpose
This document provides legal attribution, licensing terms, source repository references, and verification instructions for the external datasets utilized in the **More Than A Hack 2027 Phase 1 (Ball Possession)** challenge.

In accordance with the competition rules (`RULES.md`), parameter selection and benchmarking (`scripts/external_cv.py`, `scripts/robust_cv.py`) use only these external games. No data from the competition match is used for either.

---

## 2. Attribution & Provider Acknowledgments
All external match tracking and event datasets utilized in this project are authored and provided by:

**Metrica Sports S.L.**  
Amsterdam, The Netherlands & Barcelona, Spain  
Official Website: [https://metrica-sports.com/](https://metrica-sports.com/)  
GitHub Repository: [https://github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)  

The team gratefully acknowledges Metrica Sports for their commitment to open sports science, as well as the *Friends of Tracking* initiative for establishing foundational data standards and tutorials across the sports analytics community.

---

## 3. Dataset Inventory

The external benchmark suite consists of three complete football matches, encompassing 9 total files:

| Match ID | Format | Files Included | Sampling Rate | Coordinate System |
| :--- | :--- | :--- | :--- | :--- |
| **Sample Game 1** | CSV (Metrica Legacy) | `Sample_Game_1_RawTrackingData_Home_Team.csv`<br>`Sample_Game_1_RawTrackingData_Away_Team.csv`<br>`Sample_Game_1_RawEventsData.csv` | 25 Hz | Normalized $[0, 1] \times [0, 1]$<br>Pitch: $105 \times 68$ m |
| **Sample Game 2** | CSV (Metrica Legacy) | `Sample_Game_2_RawTrackingData_Home_Team.csv`<br>`Sample_Game_2_RawTrackingData_Away_Team.csv`<br>`Sample_Game_2_RawEventsData.csv` | 25 Hz | Normalized $[0, 1] \times [0, 1]$<br>Pitch: $105 \times 68$ m |
| **Sample Game 3** | FIFA EPTS + JSON | `Sample_Game_3_tracking.txt`<br>`Sample_Game_3_metadata.xml`<br>`Sample_Game_3_events.json` | 25 Hz | FIFA EPTS Metric Pitch<br>Pitch: $105 \times 68$ m |

### Target Directory Layout
External datasets are downloaded directly into the gitignored `ext_data/` directory:
```
<repository root>/
└── ext_data/
    ├── checksums.json
    ├── Sample_Game_1/
    │   ├── Sample_Game_1_RawTrackingData_Home_Team.csv
    │   ├── Sample_Game_1_RawTrackingData_Away_Team.csv
    │   └── Sample_Game_1_RawEventsData.csv
    ├── Sample_Game_2/
    │   ├── Sample_Game_2_RawTrackingData_Home_Team.csv
    │   ├── Sample_Game_2_RawTrackingData_Away_Team.csv
    │   └── Sample_Game_2_RawEventsData.csv
    └── Sample_Game_3/
        ├── Sample_Game_3_tracking.txt
        ├── Sample_Game_3_metadata.xml
        └── Sample_Game_3_events.json
```

---

## 4. Licensing Terms & Permitted Usage

### Terms
The `metrica-sports/sample-data` repository does not ship a formal licence file. Its README ("Legal stuff") asks users to be responsible with the data and to acknowledge Metrica Sports as the source when it is used publicly, which this project does here and in the technical document.
- **Anonymisation**: player, team and competition names are anonymised by Metrica Sports.
- **Use in this project**: evaluation and parameter selection only, leave-one-game-out over games 1–3: on the original tracking (`scripts/external_cv.py`) and on copies degraded to resemble automatic tracking (`scripts/robust_cv.py`).
- **Redistribution**: the files are downloaded into the gitignored `ext_data/` folder and are not committed.

### Relation to the competition rules
- External data about other games is allowed (RULES.md, CHALLENGE.md "What you may use"). The Metrica event files are event data produced and published by Metrica Sports for those other games (we did not label anything ourselves); how they are turned into possession labels is described in `src/event_labels.py` and in the technical document.
- No competition ground truth, video or human-produced data about the competition match was used to choose or fit any parameter.

---

## 5. Download and Verification Guide

### 5.1 Automated Download
To fetch all 9 files from a pinned commit of `metrica-sports/sample-data` (`e706dd5`), with retries and SHA-256 verification against the hashes recorded in the script:
```bash
python scripts/download_external_data.py
```

### 5.2 Verification Only
To verify that all required datasets are present and uncorrupted without downloading:
```bash
python scripts/download_external_data.py --verify-only
```

### 5.3 Force Re-download
To force a clean re-download of all files:
```bash
python scripts/download_external_data.py --force
```

### 5.4 Manual Verification via System Utilities
You can independently compute the SHA-256 hashes using native operating system tools:

- **Windows PowerShell**:
  ```powershell
  Get-FileHash -Algorithm SHA256 ext_data/Sample_Game_1/Sample_Game_1_RawEventsData.csv
  ```
- **Linux / macOS**:
  ```bash
  sha256sum ext_data/Sample_Game_1/Sample_Game_1_RawEventsData.csv
  ```

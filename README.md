# More Than A Hack 2027 — Phase 1 (Ball Possession) — team Soleva

Rule-based possession engine over the provided automatic tracking (ATD). One label per video
frame: `H`, `A`, `D`, or `X` outside the graded halves. The method, the possession rules, the external
data and the limitations are described in [Soleva.md](Soleva.md).

## Reproduce the submission

Requires Python 3.10 (3.8+ works).

```bash
pip install -r requirements.txt

# Put the gated competition bundle in ./data (or set MTAH_DATA_DIR), with the original names:
#   data/tactical/BAR-ATH_tactical_atd.xml
#   data/tactical/BAR-ATH_tactical_atd.txt

python run_pipeline.py                 # writes Soleva.csv, about 10-20 s
python evaluation/validate_submission.py Soleva.csv
```

`run_pipeline.py` is deterministic. Its only inputs are the two ATD files and `params_tuned.json`.
It stops with an error if the data is missing; there is no synthetic or fallback mode.

## Reproduce the parameter selection (external data only)

```bash
python scripts/download_external_data.py      # Metrica public sample games 1-3 -> ext_data/
python scripts/external_cv.py --stage 1 --no-write   # stage 1: core rules, clean games
python scripts/external_cv.py --no-write              # stage 2: + ball-gap rules, clean games
python scripts/robust_cv.py --write-params            # stage 3: degraded games -> params_tuned.json
                                                      # and docs/robust_cv_results.json (~17 min)
python scripts/robust_cv.py --refine --write-params   # stage 4: refinement, 4 noise seeds (~13 min)
```

## Tests

```bash
python -m pytest tests          # no competition data needed; uses a fake ATD file
```

## Layout

| path | what |
|---|---|
| `run_pipeline.py` | data → `Soleva.csv` |
| `src/atd_loader.py` | FIFA/EPTS reader (competition ATD and Metrica Game 3) |
| `src/possession_model.py` | the possession rules |
| `src/event_labels.py` | reference labels for the external Metrica games, built from their events |
| `scripts/external_cv.py` | external evaluation, stages 1-2 (clean Metrica games) |
| `scripts/robust_cv.py` | stages 3-4: selection on degraded Metrica games (final parameters) |
| `games.py`, `evaluation/` | copied unchanged from the official challenge repository (MIT) |

Data handling: `data/`, `ext_data/`, videos and generated submissions are gitignored. No match
data is committed.

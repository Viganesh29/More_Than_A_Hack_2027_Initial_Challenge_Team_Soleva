# Team Soleva — Phase 1 Ball Possession: technical document

## 1. Approach

We built a deterministic, rule-based possession engine over the provided automatic tracking
(ATD). It writes one label per video frame. We chose rules over a learned model for three reasons:

- **The definition is rule-shaped.** CHALLENGE.md defines possession by rules: possession persists
  through loose and contested phases, it changes only on *controlled* possession, and a dead ball
  is a stoppage. Each rule maps to a small number of physical thresholds that can be explained and
  inspected.
- **No labels are allowed on the provided game.** Labelled tracking with possession is scarce. The
  only external labelled source we had is three public Metrica games with event data, which is
  enough to *select* a handful of thresholds but not to train a model that would transfer to
  noisier, uncorrected tracking.
- **The ranking metric (spellF1) rewards whole, stable possessions.** Hysteresis and holding
  possession through ball dropouts address this directly. A per-frame classifier would need the
  same post-processing anyway.

The pipeline (`run_pipeline.py`) works as follows:

1. **Load the tactical ATD** (`src/atd_loader.py`). It reads the raw FIFA/EPTS text file with
   numpy and maps track slots to teams from the XML (the first `<Team>` is home). It aligns frames
   by `video_frame = frame_id - 1`. The graded halves are taken from the official `games.py`
   (video frames 11185–90651 and 110808–190197).
2. **Per-frame features.** We compute the distance from the ball to the nearest home player and to
   the nearest away player, whether the ball is tracked, and whether it is outside the pitch. We
   also compute a *stoppage* signal: the median player speed, smoothed over 1 s, below a threshold
   while the ball is not moving, sustained for at least 2 s.
3. **State machine per half** (H / A / D). The rules are listed in Section 4.
4. **Post-processing.** Short D gaps between two runs of the same team are bridged, unless the
   ball was clearly out. Possessions shorter than 1 s are merged into the neighbouring possession.
5. **Output.** Frames outside the two halves are written as `X`, which the scorer never grades.
   The file is validated with the official `possession_metrics.validate_submission`.

## 2. Assets used

| asset | used? | how / why |
|---|---|---|
| Tactical ATD (players and ball) | **yes** | the only input to the engine |
| `games.py` (official) | yes | graded half boundaries, pitch size, file paths |
| Smart Tagging | no | phases are not possessions, and `Team` is N/A on all throw-ins. Not evaluated, for lack of time |
| Homography | no | the ATD is already in pitch coordinates |
| Tactical and broadcast video | no | not decoded. The broadcast feed is not frame-aligned to the tactical video |
| Public 5-minute ground truth | check only | used only to confirm frame alignment and file format, as the rules allow. Not used for tuning or selection |
| Official `evaluation/` scripts | yes | copied unchanged and used to score the external games |

## 3. Training data

No model is trained. The engine has a small number of thresholds, chosen by a grid search on
**external data only**:

| dataset | source | labelled by | amount |
|---|---|---|---|
| Metrica Sports Sample Games 1, 2, 3 (tracking + events) | <https://github.com/metrica-sports/sample-data>, commit `e706dd5`, SHA-256 pinned in `scripts/download_external_data.py` | Metrica Sports (event data published with the games; we labelled nothing ourselves) | 3 full matches, about 430,000 frames |

The event data are not possession labels. `src/event_labels.py` converts them into frame labels
with these rules:

- The possession team is the team of the most recent on-ball event (pass, set piece, recovery,
  shot, ball lost, carry). A `BALL LOST` keeps the losing team in possession until the opponent's
  `RECOVERY`.
- A dead ball starts at a ball-out event, a shot out or goal, a foul, offside or handball, and
  lasts until the next on-ball event. A set piece always opens a dead interval after the
  preceding action.

These references have about 360–410 possessions per match, a mean possession of about 9 s and
about 40% dead ball. **No data of any kind from the provided game was used to choose any
parameter.**

### External evaluation (leave-one-game-out, official scorer)

For each held-out game, we choose the parameter set with the best mean spellF1 on the other two
games and report its score on the held-out game. The comparison point is the official baseline
algorithm (nearest player, ±50-frame majority vote) run on the same games.

| held-out game | engine spellF1 | engine macroF1 | baseline spellF1 | baseline macroF1 | possessions: engine / reference |
|---|---|---|---|---|---|
| Game 1 | 0.652 | 0.908 | 0.441 | 0.828 | 296 / 383 |
| Game 2 | 0.591 | 0.907 | 0.415 | 0.858 | 261 / 359 |
| Game 3 | 0.570 | 0.888 | 0.395 | 0.841 | 421 / 408 |
| **mean** | **0.605** | **0.901** | **0.417** | **0.842** | |

Grid: 288 parameter sets (control radius 1.0–3.0 m, dwell 3–20 frames, stoppage speed off / 0.8 / 1.2 m/s, missing-ball hold 5–20 s, minimum possession 1–2 s). The selected set was the same in two of the three folds. The final parameters are the best mean over all three games: radius 1.0 m, dwell 5 frames, stoppage speed 1.2 m/s, missing-ball hold 125 frames, minimum possession 25 frames. With the other values fixed, switching the stoppage-speed rule off lowers the mean from 0.614 to 0.600 (+0.014 for the rule), which is within the noise band.

**These numbers are not comparable to the leaderboard.** The Metrica tracking is cleaner than the
competition's uncorrected ATD, and our reference labels come from events, whereas the competition
ground truth is derived from corrected tracking. They show that the rules improve on the baseline
algorithm on independent matches. They do not predict our leaderboard score.

## 4. Possession rules implemented

- **Ball alive → one team has it.** Possession starts with the team of the nearest player when the
  ball is near a player after a stoppage. This is the restart taker: kick-off, throw-in, free kick,
  goal kick or corner.
- **Loose balls, passes in flight and contested balls** keep the last team in possession
  (CHALLENGE.md). They never produce D.
- **Ball not tracked** (41% of graded frames): the last team keeps possession. If the ball has been
  missing for more than `max_missing_hold_frames`, the frame becomes D, on the assumption that a
  long disappearance usually means the ball left the playing area.
- **Change of possession (challenges):** the opponent must be the nearest team to the ball, within
  `distance_threshold` metres, and closer than the current team by `switch_margin_m`, for
  `dwell_frames` consecutive tracked frames. A single touch, a deflection or a lost aerial duel
  does not last long enough to flip possession. Possessions shorter than `min_spell_frames` are
  merged into their neighbours.
- **Dead ball (D):**
  - the tracked ball is beyond a touchline or goal line by more than 1 m;
  - a stoppage: players are near-static and the ball is not moving for at least 2 s (free kicks,
    injuries, goal celebrations);
  - the ball has been missing for too long (above);
  - the tracked ball is far from every player for 5 s.
- **Restarts:** after a stoppage, the first player within 3.5 m of the ball once play resumes gives
  that team possession.

Final parameters are in `params_tuned.json`, and every configuration's per-game scores are in
`docs/external_cv_results.json`.

## 5. Limitations and what we would do next

- **Domain shift.** The parameters were selected on cleaner Metrica tracking. The competition ATD
  has fragmented tracks and identity switches, so the best dwell time and control radius may
  differ there. We did not adjust them, because there is no permitted way to measure them on the
  provided game.
- **Long ball gaps are ambiguous.** Holding possession through a 4-second gap is right inside a
  possession, but wrong across an unseen turnover or an unseen ball-out. A ball-trajectory model
  (extrapolating flight and detecting crossings of the lines) would separate these cases.
- **The stoppage detector is coarse.** On the external games it adds only about 0.014 spellF1,
  within the noise band. Smart Tagging set-piece intervals could confirm restarts, but we did not
  evaluate this.
- **Team assignment of the nearest player** relies on the slot-to-team mapping in the ATD metadata.
  A slot that is wrong for part of the match would flip possession labels during that stretch.
- **Small external sample.** Three matches, about 1,150 possessions in total. With a noise band of
  about 0.04, differences between the top configurations are not significant.

## 6. Reproduction

```bash
pip install -r requirements.txt
# competition bundle in ./data (or MTAH_DATA_DIR): data/tactical/BAR-ATH_tactical_atd.{xml,txt}
python run_pipeline.py                    # -> Soleva.csv
python evaluation/validate_submission.py Soleva.csv

# optional: reproduce the parameter selection on external data
python scripts/download_external_data.py
python scripts/external_cv.py
```

The pipeline is deterministic: the same inputs always produce the same file.

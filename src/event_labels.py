"""src/event_labels.py - Frame-level possession labels for the public Metrica sample games.

Used ONLY for evaluating the possession engine on external data (scripts/external_cv.py). The
competition match has no event data and none is used for it.

Source: https://github.com/metrica-sports/sample-data (Sample Games 1-3). The event files were
produced and published by Metrica Sports for those anonymised games. They are not
possession labels, so the rules below turn them into one H / A / D / X label per tracking frame,
following the possession definition in the challenge's CHALLENGE.md:

  * Possession team = team of the most recent on-ball event (PASS, SET PIECE, RECOVERY, SHOT,
    BALL LOST, CARRY). A BALL LOST keeps the losing team in possession until the opponent's
    RECOVERY ("contested or loose -> stays with the last team"). CHALLENGE events never change
    possession on their own.
  * Dead ball (D) starts at a stoppage event and lasts until the next on-ball event (the restart):
      - BALL OUT (at its end frame), SHOT whose subtype contains OUT or GOAL (end frame);
      - FAULT RECEIVED, CHALLENGE with a FAULT subtype that is not ADVANTAGE (start frame);
      - BALL LOST with subtype OFFSIDE / HAND BALL / END HALF (start frame).
    A SET PIECE that was not preceded by a detected stoppage also opens a dead interval from the
    end of the previous on-ball event, because a set piece always follows a stoppage.
  * Frames before the first event or after the last event of a period are X (not scored).

These are approximations: event timing is human-clicked, and the competition ground truth is
derived differently (from corrected tracking). Scores on these labels measure the engine against
an independent reference, not against the competition's definition.
"""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

ON_BALL = {"PASS", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST", "CARRY"}
LOST_STOPPAGE_SUBTYPES = ("OFFSIDE", "HAND BALL", "END HALF")


def load_metrica_events(path: Union[str, Path], home_team_id: Optional[str] = None) -> pd.DataFrame:
    """Loads Metrica events (CSV for Games 1-2, JSON for Game 3) into a common table.

    Columns: team ('H'/'A'), type, subtype, period, start_frame, end_frame (1-based frame ids).
    For JSON, `home_team_id` is the first <Team> id of the tracking metadata (EPTS convention).
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        rows = []
        for e in data["data"] if isinstance(data, dict) else data:
            team_id = (e.get("team") or {}).get("id")
            subtypes = e.get("subtypes")
            if isinstance(subtypes, list):
                sub = "-".join(s.get("name", "") for s in subtypes if s)
            elif isinstance(subtypes, dict):
                sub = subtypes.get("name", "")
            else:
                sub = ""
            rows.append(
                {
                    "team": "H" if team_id == home_team_id else "A",
                    "type": (e.get("type") or {}).get("name", ""),
                    "subtype": sub or "",
                    "period": int(e.get("period") or 0),
                    "start_frame": int((e.get("start") or {}).get("frame") or 0),
                    "end_frame": int((e.get("end") or {}).get("frame") or 0),
                }
            )
        df = pd.DataFrame(rows)
    else:
        raw = pd.read_csv(path)
        df = pd.DataFrame(
            {
                "team": np.where(raw["Team"].str.strip().str.lower() == "home", "H", "A"),
                "type": raw["Type"].fillna("").str.strip().str.upper(),
                "subtype": raw["Subtype"].fillna("").astype(str).str.strip().str.upper(),
                "period": raw["Period"].astype(int),
                "start_frame": raw["Start Frame"].astype(int),
                "end_frame": raw["End Frame"].astype(int),
            }
        )
    df["type"] = df["type"].str.upper()
    df["subtype"] = df["subtype"].str.upper()
    df["end_frame"] = np.maximum(df["end_frame"], df["start_frame"])
    return df.sort_values(["period", "start_frame", "end_frame"], kind="stable").reset_index(drop=True)


def _stoppage_frame(row) -> Optional[int]:
    """Frame at which this event stops play, or None if it does not."""
    etype, sub = row.type, row.subtype
    if etype == "BALL OUT":
        return int(row.end_frame)
    if etype == "SHOT" and ("OUT" in sub or "GOAL" in sub):
        return int(row.end_frame)
    if etype == "FAULT RECEIVED":
        return int(row.start_frame)
    if etype == "CHALLENGE" and "FAULT" in sub and "ADVANTAGE" not in sub:
        return int(row.start_frame)
    if etype == "BALL LOST" and any(s in sub for s in LOST_STOPPAGE_SUBTYPES):
        return int(row.start_frame)
    return None


def labels_from_events(events: pd.DataFrame, n_frames: int) -> np.ndarray:
    """One label per tracking row (row r = frame id r + 1): 'H', 'A', 'D' or 'X'."""
    labels = np.full(n_frames, "X", dtype="<U1")

    for period in sorted(events["period"].unique()):
        ev = events[events["period"] == period]
        if ev.empty:
            continue
        first = max(1, int(ev["start_frame"].min()))
        last = min(n_frames, int(ev["end_frame"].max()))
        if last < first:
            continue

        span = last - first + 1
        team = np.full(span, "X", dtype="<U1")
        dead = np.zeros(span, dtype=bool)

        on_ball = ev[ev["type"].isin(ON_BALL)]
        on_starts = on_ball["start_frame"].to_numpy()

        # Possession team: forward fill from on-ball event starts.
        marks = np.full(span, "", dtype="<U1")
        for row in on_ball.itertuples():
            k = int(row.start_frame) - first
            if 0 <= k < span:
                marks[k] = row.team
        current = "X"
        for k in range(span):
            if marks[k]:
                current = marks[k]
            team[k] = current

        def mark_dead(start_frame: int) -> None:
            later = on_starts[on_starts > start_frame]
            end_frame = int(later[0]) if len(later) else last + 1
            a = max(start_frame, first) - first
            b = min(end_frame, last + 1) - first
            if b > a:
                dead[a:b] = True

        prev_on_ball_end = None
        for row in ev.itertuples():
            stop = _stoppage_frame(row)
            if stop is not None:
                mark_dead(stop)
            if row.type == "SET PIECE" and prev_on_ball_end is not None:
                k = int(row.start_frame) - first
                gap_start = prev_on_ball_end + 1
                if k > 0 and gap_start < row.start_frame and not dead[max(0, k - 1)]:
                    mark_dead(gap_start)
            if row.type in ON_BALL:
                prev_on_ball_end = int(row.end_frame)

        period_labels = np.where(dead, "D", team)
        labels[first - 1 : last] = period_labels

    return labels

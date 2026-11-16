"""scripts/external_cv.py - Leave-one-game-out evaluation on the public Metrica sample games.

Everything here runs on EXTERNAL data only (https://github.com/metrica-sports/sample-data,
downloaded by scripts/download_external_data.py into the gitignored ext_data/). The competition
match, its video and its ground truth are never opened.

What it does:
  1. Loads tracking for Metrica Sample Games 1-3 and builds reference labels from their event data
     (src/event_labels.py).
  2. Scores the official baseline algorithm (nearest player, +/-50-frame majority vote, the same
     as starter/baseline.py) and the possession engine with the official scorer
     (evaluation/possession_metrics.py).
  3. Leave-one-game-out: for each held-out game, picks the parameter set with the best mean
     spellF1 on the other two games and reports its score on the held-out game.
  4. Picks the final parameters as the best mean spellF1 over all three games and writes them,
     with every score, to params_tuned.json and docs/external_cv_results.json.

    python scripts/download_external_data.py
    python scripts/external_cv.py            # full grid, a few minutes
    python scripts/external_cv.py --quick    # defaults + baseline only
"""

import argparse
import itertools
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402  official scorer
from src.atd_loader import parse_epts_metadata, read_epts_raw, to_unified  # noqa: E402
from src.data_loader import load_tracking  # noqa: E402
from src.event_labels import labels_from_events, load_metrica_events  # noqa: E402
from src.possession_model import PossessionConfig, predict_possession  # noqa: E402
from src.tracking import UnifiedTrackingData  # noqa: E402

EXT = os.path.join(ROOT, "ext_data")

# Grid of physically meaningful values. Not fitted to the competition match in any way.
GRID = {
    "distance_threshold": [1.0, 1.5, 2.0, 3.0],
    "dwell_frames": [3, 5, 10, 20],
    "stoppage_speed_mps": [0.0, 0.8, 1.2],  # 0.0 = stoppage-speed rule off
    "max_missing_hold_frames": [125, 250, 500],
    "min_spell_frames": [25, 50],
}


def load_game(game: int) -> Tuple[UnifiedTrackingData, np.ndarray]:
    """(tracking, reference labels) for one Metrica sample game, rows = frame_id - 1."""
    folder = os.path.join(EXT, f"Sample_Game_{game}")
    if game in (1, 2):
        td = load_tracking(folder)
        if td.frame_indices[0] != 1 or not np.array_equal(
            td.frame_indices, np.arange(1, td.num_frames + 1)
        ):
            raise ValueError(f"Game {game}: tracking frames are not 1..N")
        events = load_metrica_events(
            os.path.join(folder, f"Sample_Game_{game}_RawEventsData.csv")
        )
    else:
        meta = os.path.join(folder, "Sample_Game_3_metadata.xml")
        layout = parse_epts_metadata(meta)
        players, ball, _ = read_epts_raw(os.path.join(folder, "Sample_Game_3_tracking.txt"), layout)
        n = len(ball)
        periods = np.zeros(n, dtype=np.int32)
        p = layout.provider_params
        for half, (a, b) in enumerate(
            [("first_half_start", "first_half_end"), ("second_half_start", "second_half_end")], 1
        ):
            if p.get(a) and p.get(b):
                periods[int(p[a]) - 1 : int(p[b])] = half
        td = to_unified(players, ball, layout.is_home, periods, layout.field_size or (105.0, 68.0),
                        layout.frame_rate or 25.0, layout.player_ids)
        root_team = _first_team_id(meta)
        events = load_metrica_events(os.path.join(folder, "Sample_Game_3_events.json"), root_team)
    labels = labels_from_events(events, td.num_frames)
    return td, labels


def _first_team_id(meta_path: str) -> str:
    import xml.etree.ElementTree as ET

    for e in ET.parse(meta_path).getroot().iter():
        if e.tag.rsplit("}", 1)[-1] == "Team" and e.get("id"):
            return e.get("id")
    raise ValueError("no Team in metadata")


def baseline_labels(td: UnifiedTrackingData, window: int = 50) -> np.ndarray:
    """The official baseline: nearest player's team, D without a ball, +/-window majority vote."""
    n = td.num_frames
    codes = np.full(n, -1, dtype=np.int8)
    valid = td.ball_visible_mask
    idx = np.flatnonzero(valid)
    b = td.ball_position[idx, None, :]
    dh = np.linalg.norm(td.home_positions[idx] - b, axis=2)
    da = np.linalg.norm(td.away_positions[idx] - b, axis=2)
    dh = np.where(np.isnan(dh), np.inf, dh).min(axis=1)
    da = np.where(np.isnan(da), np.inf, da).min(axis=1)
    seen = np.isfinite(np.minimum(dh, da))
    codes[idx[seen]] = np.where(dh[seen] <= da[seen], 0, 1)

    home_c = np.concatenate(([0], np.cumsum(codes == 0)))
    away_c = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(n)
    lo, hi = np.maximum(0, i - window), np.minimum(n, i + window + 1)
    home, away = home_c[hi] - home_c[lo], away_c[hi] - away_c[lo]
    out = np.full(n, "D", dtype="<U1")
    has = (home + away) > 0
    out[has] = np.where(home[has] >= away[has], "H", "A")
    return out


def engine_labels(td: UnifiedTrackingData, params: Dict) -> np.ndarray:
    """Engine prediction with X outside the tracking periods."""
    labels = predict_possession(td, params)
    labels = labels.astype("<U1")
    labels[td.periods == 0] = "X"
    return labels


def score(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Official metrics on one game."""
    result = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
    return {
        "spellF1": round(float(result["spell_f1_mean"]), 4),
        "macroF1": round(float(result["macro_f1"]), 4),
        "n_pred_spells": int(result["n_pred_spells"]),
        "n_gt_spells": int(result["n_gt_spells"]),
    }


def grid_configs(quick: bool) -> List[Dict]:
    if quick:
        return [{}]
    keys = list(GRID)
    configs = []
    for values in itertools.product(*(GRID[k] for k in keys)):
        cfg = dict(zip(keys, values))
        if cfg["stoppage_speed_mps"] == 0.0:
            cfg["enable_stoppage_speed"] = False
            cfg.pop("stoppage_speed_mps")
        configs.append(cfg)
    # de-duplicate (stoppage off makes the speed value irrelevant)
    unique = {json.dumps(c, sort_keys=True): c for c in configs}
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="defaults and baseline only")
    parser.add_argument("--games", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--no-write", action="store_true", help="do not update params_tuned.json")
    args = parser.parse_args()

    t0 = time.time()
    data = {}
    for g in args.games:
        data[g] = load_game(g)
        print(f"loaded game {g}: {data[g][0].num_frames} frames ({time.time() - t0:.0f}s)")

    baseline = {g: score(gt, baseline_labels(td)) for g, (td, gt) in data.items()}
    print("baseline (official algorithm, window 50):", baseline)

    configs = grid_configs(args.quick)
    print(f"evaluating {len(configs)} parameter sets x {len(data)} games")
    table = []
    for i, cfg in enumerate(configs):
        per_game = {g: score(gt, engine_labels(td, cfg)) for g, (td, gt) in data.items()}
        table.append({"params": cfg, "per_game": per_game})
        if (i + 1) % 10 == 0 or i + 1 == len(configs):
            print(f"  {i + 1}/{len(configs)} done ({time.time() - t0:.0f}s)")

    def mean_spell(entry, games):
        return float(np.mean([entry["per_game"][g]["spellF1"] for g in games]))

    games = list(data)
    folds = []
    for held_out in games:
        train = [g for g in games if g != held_out]
        best = max(table, key=lambda e: mean_spell(e, train))
        folds.append(
            {
                "held_out_game": held_out,
                "selected_params": best["params"],
                "train_mean_spellF1": round(mean_spell(best, train), 4),
                "held_out": best["per_game"][held_out],
                "baseline_held_out": baseline[held_out],
            }
        )
    final = max(table, key=lambda e: mean_spell(e, games))
    defaults = next((e for e in table if e["params"] == {}), None)

    summary = {
        "data": "Metrica Sports public sample games 1-3 (external); labels derived from their event data",
        "scorer": "official evaluation/possession_metrics.py",
        "baseline_per_game": baseline,
        "baseline_mean_spellF1": round(float(np.mean([baseline[g]["spellF1"] for g in games])), 4),
        "logo_folds": folds,
        "logo_mean_held_out_spellF1": round(float(np.mean([f["held_out"]["spellF1"] for f in folds])), 4),
        "logo_mean_held_out_macroF1": round(float(np.mean([f["held_out"]["macroF1"] for f in folds])), 4),
        "final_params": final["params"],
        "final_per_game": final["per_game"],
        "n_configs": len(configs),
        "grid": GRID if not args.quick else None,
        "defaults_per_game": defaults["per_game"] if defaults else None,
    }
    print(json.dumps({k: summary[k] for k in ("baseline_mean_spellF1", "logo_mean_held_out_spellF1",
                                              "logo_mean_held_out_macroF1", "final_params")}, indent=2))
    for f in folds:
        print(f"  held out game {f['held_out_game']}: engine {f['held_out']}  baseline {f['baseline_held_out']}")

    if args.no_write:
        return
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "external_cv_results.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "all_configs": table}, fh, indent=1)

    if not args.quick:
        full = PossessionConfig.from_dict(final["params"]).__dict__
        manifest = {
            "schema_version": "2.0.0",
            "model_name": "rule_based_possession_v2",
            "notes": (
                "Parameters selected by scripts/external_cv.py on the public Metrica sample games "
                "only (best mean spellF1 over games 1-3 against event-derived labels). Nothing was "
                "fitted to the competition match, its video or its ground truth."
            ),
            "external_cv": {
                "baseline_mean_spellF1": summary["baseline_mean_spellF1"],
                "logo_mean_held_out_spellF1": summary["logo_mean_held_out_spellF1"],
                "logo_mean_held_out_macroF1": summary["logo_mean_held_out_macroF1"],
            },
            "parameters": full,
        }
        with open(os.path.join(ROOT, "params_tuned.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        print("wrote params_tuned.json and docs/external_cv_results.json")


if __name__ == "__main__":
    main()

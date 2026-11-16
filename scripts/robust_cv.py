"""scripts/robust_cv.py - Robust parameter selection on degraded external tracking.

Why: parameters selected on the clean Metrica sample games may not transfer to the competition
ATD, which is uncorrected (CHALLENGE.md, DATA.md). This script degrades the Metrica tracking so
that it looks more like automatic tracking, using only properties stated in the official
documentation, and selects parameters on the degraded games:

  * extra ball dropouts during play until the ball is missing on about 41% of in-period frames
    (CHALLENGE.md: "the ball is missing on 41% of graded frames"), with random gap lengths, some
    longer than 5 s;
  * Gaussian position noise (players 0.3 m, ball 0.5 m);
  * player track dropouts (about 10% of frames, in chunks), a stand-in for fragmented tracks.

The reference labels still come from the Metrica event data (src/event_labels.py). Nothing from
the competition match, its video or its ground truth is used.

Selection: random search over a wider parameter space (fixed seed), leave-one-game-out on the
degraded games (2 noise seeds per game), compared against the current params_tuned.json on exactly
the same degraded data.

    python scripts/robust_cv.py                 # ~15-25 min, writes docs/robust_cv_results.json
    python scripts/robust_cv.py --write-params  # also writes params_tuned.json if adopted
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from external_cv import baseline_labels, engine_labels, load_game, score  # noqa: E402
from src.params_loader import load_params_tuned  # noqa: E402
from src.possession_model import PossessionConfig  # noqa: E402
from src.tracking import UnifiedTrackingData  # noqa: E402

TARGET_BALL_VISIBLE = 0.59  # CHALLENGE.md: the ball is missing on 41% of graded frames
NOISE_SEEDS = (11, 22)
SEARCH_SEED = 2027
N_RANDOM = 400

SPACE = {
    "distance_threshold": [0.75, 1.0, 1.25, 1.5, 2.0],
    "dwell_frames": [3, 5, 8, 12],
    "switch_margin_m": [0.25, 0.5, 1.0],
    "max_missing_hold_frames": [75, 100, 125, 175, 250],
    "ball_interp_max_gap": [0, 25, 50, 100],
    "dead_long_gaps_from_start": [False, True],
    "restart_search_window": [2.5, 3.5, 5.0],
    "bridging_max_gap": [0, 25, 50],
    "min_spell_frames": [15, 25, 40],
}
FIXED = {"enable_stoppage_speed": False}


def _chunks_until(rng, mask_ok: np.ndarray, n_remove: int, mean_len: float, max_len: int):
    """Random chunks of indices (only where mask_ok) totalling about n_remove frames."""
    n = len(mask_ok)
    out = np.zeros(n, dtype=bool)
    removed = 0
    candidates = np.flatnonzero(mask_ok)
    if len(candidates) == 0 or n_remove <= 0:
        return out
    attempts = 0
    while removed < n_remove and attempts < 100000:
        attempts += 1
        start = int(candidates[rng.integers(len(candidates))])
        length = int(min(max_len, max(5, rng.exponential(mean_len))))
        seg = slice(start, min(n, start + length))
        new = mask_ok[seg] & ~out[seg]
        out[seg] |= mask_ok[seg]
        removed += int(new.sum())
    return out


def degrade(td: UnifiedTrackingData, seed: int) -> UnifiedTrackingData:
    """Noisier, gappier copy of a tracking dataset (deterministic for a given seed)."""
    rng = np.random.default_rng(seed)
    in_period = td.periods > 0
    ball = td.ball_position.copy()
    visible = ~np.isnan(ball).any(axis=1)

    # 1. extra ball dropouts on frames where the ball is currently visible
    n_period = int(in_period.sum())
    n_visible = int((visible & in_period).sum())
    n_remove = int(n_visible - TARGET_BALL_VISIBLE * n_period)
    drop = _chunks_until(rng, visible & in_period, n_remove, mean_len=40.0, max_len=300)
    ball[drop] = np.nan

    # 2. position noise
    ball = ball + rng.normal(0.0, 0.5, size=ball.shape).astype(np.float32)
    home = td.home_positions + rng.normal(0.0, 0.3, size=td.home_positions.shape).astype(np.float32)
    away = td.away_positions + rng.normal(0.0, 0.3, size=td.away_positions.shape).astype(np.float32)

    # 3. player track dropouts (~10% of each track's frames, in chunks)
    for arr in (home, away):
        for p in range(arr.shape[1]):
            seen = ~np.isnan(arr[:, p, 0])
            gone = _chunks_until(rng, seen, int(0.10 * seen.sum()), mean_len=50.0, max_len=500)
            arr[gone, p, :] = np.nan

    return dataclasses.replace(td, ball_position=ball, home_positions=home, away_positions=away)


def sample_configs(n: int, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    seen, configs = set(), []
    while len(configs) < n:
        cfg = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
        cfg = {k: (v.item() if hasattr(v, "item") else v) for k, v in cfg.items()}
        cfg.update(FIXED)
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=N_RANDOM)
    parser.add_argument("--write-params", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    games = [1, 2, 3]
    clean, degraded = {}, {}
    for g in games:
        td, gt = load_game(g)
        clean[g] = (td, gt)
        degraded[g] = [(degrade(td, seed * 10 + g), gt) for seed in NOISE_SEEDS]
        vis = [float((~np.isnan(d.ball_position).any(1))[d.periods > 0].mean()) for d, _ in degraded[g]]
        print(f"game {g}: ball visible clean {clean[g][0].ball_visible_mask[td.periods > 0].mean():.3f}"
              f" -> degraded {np.round(vis, 3).tolist()}  ({time.time() - t0:.0f}s)")

    def eval_cfg(cfg: Dict) -> Dict:
        per_game = {}
        for g in games:
            runs = [score(gt, engine_labels(td, cfg)) for td, gt in degraded[g]]
            per_game[g] = {
                "spellF1": round(float(np.mean([r["spellF1"] for r in runs])), 4),
                "macroF1": round(float(np.mean([r["macroF1"] for r in runs])), 4),
            }
        return per_game

    current = load_params_tuned(os.path.join(ROOT, "params_tuned.json"))
    current_entry = {"params": current, "per_game": eval_cfg(current)}
    base_deg = {}
    for g in games:
        runs = [score(gt, baseline_labels(td)) for td, gt in degraded[g]]
        base_deg[g] = round(float(np.mean([r["spellF1"] for r in runs])), 4)
    print("current params on degraded data:", current_entry["per_game"])
    print("baseline algorithm on degraded data:", base_deg)

    configs = sample_configs(args.n, SEARCH_SEED)
    table = []
    for i, cfg in enumerate(configs):
        table.append({"params": cfg, "per_game": eval_cfg(cfg)})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(configs)} ({time.time() - t0:.0f}s)")
    table.append(current_entry)

    def mean_spell(entry, gs):
        return float(np.mean([entry["per_game"][g]["spellF1"] for g in gs]))

    folds = []
    for held in games:
        train = [g for g in games if g != held]
        best = max(table, key=lambda e: mean_spell(e, train))
        folds.append({
            "held_out_game": held,
            "selected_params": best["params"],
            "held_out_spellF1": best["per_game"][held]["spellF1"],
            "current_params_spellF1": current_entry["per_game"][held]["spellF1"],
            "baseline_spellF1": base_deg[held],
        })
    final = max(table, key=lambda e: mean_spell(e, games))
    new_mean = float(np.mean([f["held_out_spellF1"] for f in folds]))
    cur_mean = float(np.mean([f["current_params_spellF1"] for f in folds]))
    wins = sum(f["held_out_spellF1"] > f["current_params_spellF1"] for f in folds)
    adopt = new_mean > cur_mean and wins >= 2

    final_clean = {g: score(gt, engine_labels(td, final["params"])) for g, (td, gt) in clean.items()}
    cur_clean = {g: score(gt, engine_labels(td, current)) for g, (td, gt) in clean.items()}

    summary = {
        "degradation": {"target_ball_visible": TARGET_BALL_VISIBLE, "ball_noise_m": 0.5,
                        "player_noise_m": 0.3, "player_dropout": 0.10, "noise_seeds": NOISE_SEEDS},
        "search": {"n_random": args.n, "seed": SEARCH_SEED, "space": SPACE, "fixed": FIXED},
        "logo_folds": folds,
        "logo_mean_held_out_new": round(new_mean, 4),
        "logo_mean_held_out_current": round(cur_mean, 4),
        "held_out_games_improved": wins,
        "adopt": adopt,
        "final_params": final["params"],
        "final_degraded_per_game": final["per_game"],
        "final_clean_per_game": final_clean,
        "current_clean_per_game": cur_clean,
        "baseline_degraded_per_game": base_deg,
    }
    print(json.dumps({k: summary[k] for k in ("logo_mean_held_out_new", "logo_mean_held_out_current",
                                              "held_out_games_improved", "adopt", "final_params")}, indent=1))
    for f in folds:
        print(f"  held out {f['held_out_game']}: new {f['held_out_spellF1']}  current "
              f"{f['current_params_spellF1']}  baseline {f['baseline_spellF1']}")
    print("clean data, final:", {g: v["spellF1"] for g, v in final_clean.items()},
          " current:", {g: v["spellF1"] for g, v in cur_clean.items()})

    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "robust_cv_results.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "all_configs": table}, fh, indent=1)

    if args.write_params and adopt:
        full = dataclasses.asdict(PossessionConfig.from_dict(final["params"]))
        manifest = {
            "schema_version": "3.0.0",
            "model_name": "rule_based_possession_v3",
            "notes": (
                "Parameters selected by scripts/robust_cv.py on degraded versions of the public "
                "Metrica sample games (leave-one-game-out, best mean spellF1). Nothing was fitted "
                "to the competition match, its video or its ground truth."
            ),
            "external_cv": {
                "selection": "degraded Metrica games, leave-one-game-out",
                "logo_mean_held_out_spellF1_degraded": round(new_mean, 4),
                "previous_params_same_data": round(cur_mean, 4),
                "clean_games_spellF1": {str(g): v["spellF1"] for g, v in final_clean.items()},
            },
            "parameters": full,
        }
        with open(os.path.join(ROOT, "params_tuned.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        print("adopted: wrote params_tuned.json")
    elif args.write_params:
        print("not adopted: params_tuned.json unchanged")


if __name__ == "__main__":
    main()

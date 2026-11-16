"""run_pipeline.py - Regenerates the submission file from the provided match data.

    python run_pipeline.py                       # writes Soleva.csv
    python run_pipeline.py --data-dir D:/mtah     # or set MTAH_DATA_DIR

Steps:
  1. Load the frozen parameters from params_tuned.json (chosen on external Metrica data only).
  2. Load the tactical ATD (data/tactical/BAR-ATH_tactical_atd.{xml,txt}) with src/atd_loader.py.
     video_frame = frame_id - 1; graded halves come from games.half_video_blocks().
  3. Run the possession engine (src/possession_model.py) on each half.
  4. Write one row per video frame (199,319 rows). Frames outside the two halves are X, which the
     scorer never grades.
  5. Validate the file with the official evaluation/possession_metrics.validate_submission.

There is no synthetic or fallback mode: if the match data is missing, the pipeline stops with an
error naming the missing files. Only the automatic tracking is used. The video, the Smart Tagging,
the homography and every ground-truth file are not read.
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Optional, Union

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402  official validator
from src.atd_loader import load_atd  # noqa: E402
from src.params_loader import load_params_tuned  # noqa: E402
from src.possession_model import predict_possession  # noqa: E402


def run_pipeline(
    output_path: Union[str, Path] = "Soleva.csv",
    data_dir: Optional[str] = None,
    params_path: Union[str, Path] = "params_tuned.json",
    quiet: bool = False,
) -> Path:
    """Runs the full pipeline and returns the path of the validated submission file."""
    log = (lambda *a: None) if quiet else print
    t0 = time.time()
    out_file = Path(output_path).resolve()

    log(f"[1/5] parameters: {params_path}")
    params = load_params_tuned(params_path)

    log("[2/5] loading tactical ATD ...")
    tracking = load_atd(data_dir=data_dir)
    log(
        f"      {tracking.num_frames} frames, ball tracked on "
        f"{100.0 * tracking.ball_visible_mask.mean():.1f}%, "
        f"{len(tracking.home_track_ids)} home / {len(tracking.away_track_ids)} away slots"
    )

    log("[3/5] predicting possession ...")
    labels = predict_possession(tracking, params).astype("<U1")
    labels[tracking.periods == 0] = "X"  # outside the graded halves

    n_video = games.GAMES[games.GAME]["n_video_frames"]
    if len(labels) != n_video:
        raise RuntimeError(f"expected {n_video} labels, got {len(labels)}")

    log(f"[4/5] writing {out_file}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{i},{lbl}\n" for i, lbl in enumerate(labels))

    log("[5/5] validating with the official checker ...")
    reason = pm.validate_submission(pm.read_label_csv(out_file), n_video)
    if reason:
        raise RuntimeError(f"submission failed validation: {reason}")

    graded = tracking.periods > 0
    counts = {k: int(np.sum(labels[graded] == k)) for k in ("H", "A", "D")}
    log(f"      OK. graded frames: {counts}  ({time.time() - t0:.0f}s)")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", default="Soleva.csv")
    parser.add_argument("--data-dir", default=None,
                        help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    parser.add_argument("--params", default="params_tuned.json")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.output, args.data_dir, args.params, args.quiet)


if __name__ == "__main__":
    main()

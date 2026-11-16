"""src/possession_model.py - Rule-based ball possession engine.

Implements the possession rules stated in the challenge's CHALLENGE.md:

  * While the ball is alive exactly one team is in possession. "In phases where the ball is
    contested or loose, possession stays with the last team that had it." So a loose ball, a ball
    in flight and a ball the tracker has lost all keep the last possessor (no D).
  * A change of possession needs *controlled* possession by the other team. Proxy: the nearest
    player to the ball is an opponent, within the control radius and clearly closer than the
    current team (margin), for `dwell_frames` consecutive tracked frames.
  * D means the ball is out of play. Evidence used, all from the automatic tracking:
      1. the tracked ball is outside the pitch (plus a margin);
      2. a stoppage: players are near-static (smoothed median speed below a threshold) and the
         ball is not moving, sustained for a minimum duration (free kicks, injuries, VAR, goals);
      3. the ball has been missing for longer than `max_missing_hold_frames`;
      4. the tracked ball sits far from every player for `prolonged_loose_dead_frames`.
  * After a stoppage, possession restarts with the team of the nearest player once play resumes
    and the ball is near a player.

Every threshold is a physical quantity (metres, m/s, frames at 25 fps). The defaults are set from
football reasoning; scripts/external_cv.py evaluates them on the public Metrica sample games.
Nothing here was fitted to the competition match, its ground truth or its video.

Processing is deterministic and isolated per match period. Frames with period 0 (outside the
graded halves) are labelled X by the pipeline, not by this engine.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

from src.tracking import UnifiedTrackingData


@dataclass
class PossessionConfig:
    """Parameters of the possession engine. Units: metres, metres/second, frames (25 fps)."""

    # Geometry
    distance_threshold: float = 2.0  # m: control radius around the ball
    pitch_margin: float = 1.0  # m: ball this far beyond a line counts as out of play
    # (pitch size comes from the tracking data: 105.3 x 68.0 m for the match, from games.py)

    # Possession change (hysteresis)
    enable_hysteresis: bool = True
    dwell_frames: int = 10  # consecutive frames of opponent control needed to switch
    switch_margin_m: float = 0.5  # opponent must be this much closer than the current team

    # Ball-loss handling
    hold_on_missing_ball: bool = True  # keep last possessor while the ball is not tracked
    hold_on_loose_ball: bool = True  # keep last possessor while nobody is near the ball
    max_missing_hold_frames: int = 250  # 10 s without a ball -> treat as dead
    prolonged_loose_dead_frames: int = 125  # 5 s with the ball far from everyone -> dead

    # Stoppage detection from player motion
    enable_dead_ball: bool = True  # master switch for all D rules
    enable_stoppage_speed: bool = True
    stoppage_speed_mps: float = 1.0  # smoothed median player speed below this = near-static
    stoppage_ball_speed_mps: float = 2.0  # ...and the ball not moving faster than this
    stoppage_min_frames: int = 50  # near-static for at least 2 s to count as a stoppage
    speed_smoothing_frames: int = 25  # centred window for speed smoothing
    max_player_speed_mps: float = 12.0  # faster jumps are identity switches, ignored
    min_tracked_players: int = 6  # fewer tracked players -> speed unknown

    # Restarts
    enable_restart_attribution: bool = True
    restart_search_window: float = 3.5  # m: restart taker search radius after a stoppage

    # Post-processing
    enable_stoppage_bridging: bool = True
    bridging_max_gap: int = 25  # fill D gaps up to this long between the same team
    enable_spell_pruning: bool = True
    min_spell_frames: int = 25  # possessions shorter than this are merged away
    smoothing_window: int = 0  # optional majority filter (0 = off)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PossessionConfig":
        """Builds a configuration from a flat dict or a params manifest ({"parameters": {...}})."""
        if not d:
            return cls()
        raw = d.get("parameters", d) if isinstance(d, dict) else d
        if not isinstance(raw, dict):
            return cls()

        alias_map = {
            "proximity_threshold": "distance_threshold",
            "k_dwell": "dwell_frames",
            "delta_d": "switch_margin_m",
            "enable_bridging": "enable_stoppage_bridging",
            "max_bridge_frames": "bridging_max_gap",
            "enable_restart": "enable_restart_attribution",
            "restart_radius": "restart_search_window",
            "enable_pruning": "enable_spell_pruning",
            "enable_min_spell_pruning": "enable_spell_pruning",
            "window_size": "smoothing_window",
        }
        fields = cls.__dataclass_fields__
        kwargs: Dict[str, Any] = {}
        for key, value in raw.items():
            name = alias_map.get(key, key)
            if name not in fields:
                continue
            ftype = fields[name].type
            if ftype in (bool, "bool"):
                kwargs[name] = bool(value)
            elif ftype in (int, "int"):
                kwargs[name] = int(value)
            elif ftype in (float, "float"):
                kwargs[name] = float(value)
            else:
                kwargs[name] = value
        return cls(**kwargs)


def _centred_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean ignoring NaN; NaN where the window holds no data."""
    if window <= 1:
        return values.astype(np.float64)
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(filled)))
    ccnt = np.concatenate(([0], np.cumsum(valid)))
    n = len(values)
    idx = np.arange(n)
    half = window // 2
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n, idx + (window - half))
    total = csum[hi] - csum[lo]
    count = ccnt[hi] - ccnt[lo]
    out = np.full(n, np.nan)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _keep_long_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keeps only runs of True that are at least `min_len` long."""
    if min_len <= 1 or not mask.any():
        return mask.copy()
    out = np.zeros_like(mask)
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start >= min_len:
            out[start:end] = True
    return out


def _runs(labels: np.ndarray) -> List[Tuple[str, int, int]]:
    """Run-length encoding: [(label, start, end_inclusive)]."""
    if len(labels) == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change - 1, [len(labels) - 1]))
    return [(str(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


class ModularPossessionEngine:
    """Deterministic, per-period state machine over tracking-derived features."""

    def __init__(self, config: Optional[PossessionConfig] = None):
        self.config = config or PossessionConfig()

    # ------------------------------------------------------------------ features

    @staticmethod
    def _resolve_periods(
        num_frames: int,
        existing_periods: np.ndarray,
        half_bounds: Optional[Union[Dict[Any, Tuple[int, int]], List[Tuple[int, int]]]] = None,
    ) -> np.ndarray:
        """Overrides periods with explicit [start, end) frame bounds when given."""
        if not half_bounds:
            return existing_periods
        periods = existing_periods.copy()
        items = list(half_bounds.items()) if isinstance(half_bounds, dict) else list(
            enumerate(half_bounds, start=1)
        )
        for key, bounds in items:
            if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
                continue
            start = max(0, min(int(bounds[0]), num_frames))
            end = max(0, min(int(bounds[1]), num_frames))
            if start >= end:
                continue
            if isinstance(key, str):
                digits = re.findall(r"\d+", key)
                key = int(digits[0]) if digits else 1
            periods[start:end] = np.int32(key)
        return periods

    def _distances(
        self, td: UnifiedTrackingData
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Nearest home / away player distance to the ball, ball validity, ball out of bounds."""
        cfg = self.config
        n = td.num_frames
        ball = td.ball_position
        ball_valid = td.ball_visible_mask

        half_l = td.pitch_length / 2.0 + cfg.pitch_margin
        half_w = td.pitch_width / 2.0 + cfg.pitch_margin
        with np.errstate(invalid="ignore"):
            oob = ball_valid & ((np.abs(ball[:, 0]) > half_l) | (np.abs(ball[:, 1]) > half_w))

        dist_h = np.full(n, np.inf, dtype=np.float32)
        dist_a = np.full(n, np.inf, dtype=np.float32)
        idx = np.flatnonzero(ball_valid)
        if len(idx):
            b = ball[idx, np.newaxis, :]
            for pos, out in ((td.home_positions, dist_h), (td.away_positions, dist_a)):
                if pos.shape[1] == 0:
                    continue
                d = np.linalg.norm(pos[idx] - b, axis=2)
                d = np.where(np.isnan(d), np.inf, d)
                out[idx] = d.min(axis=1)
        return dist_h, dist_a, ball_valid, oob

    def _stoppage_mask(self, td: UnifiedTrackingData) -> np.ndarray:
        """Frames inside a sustained near-static phase (players slow and the ball not moving)."""
        cfg = self.config
        n = td.num_frames
        if n < 2 or not (cfg.enable_dead_ball and cfg.enable_stoppage_speed):
            return np.zeros(n, dtype=bool)

        fps = float(td.fps)
        pos = np.concatenate([td.home_positions, td.away_positions], axis=1)
        if pos.shape[1] == 0:
            return np.zeros(n, dtype=bool)

        step = np.linalg.norm(pos[1:] - pos[:-1], axis=2) * fps  # (n-1, P) m/s
        step[step > cfg.max_player_speed_mps] = np.nan
        tracked = (~np.isnan(step)).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_speed = np.nanmedian(step, axis=1)
        median_speed[tracked < cfg.min_tracked_players] = np.nan
        median_speed = np.concatenate(([np.nan], median_speed))
        player_speed = _centred_nanmean(median_speed, cfg.speed_smoothing_frames)

        ball_step = np.linalg.norm(td.ball_position[1:] - td.ball_position[:-1], axis=1) * fps
        ball_step = np.concatenate(([np.nan], ball_step))
        ball_speed = _centred_nanmean(ball_step, cfg.speed_smoothing_frames)

        with np.errstate(invalid="ignore"):
            slow = player_speed < cfg.stoppage_speed_mps  # NaN compares False
            ball_moving = ball_speed > cfg.stoppage_ball_speed_mps  # NaN (no ball) -> not moving
        return _keep_long_runs(slow & ~ball_moving, cfg.stoppage_min_frames)

    # ------------------------------------------------------------------ state machine

    def _process_period(
        self,
        dh: np.ndarray,
        da: np.ndarray,
        valid: np.ndarray,
        oob: np.ndarray,
        stopped: np.ndarray,
    ) -> np.ndarray:
        """Labels one period frame by frame. Inputs are already sliced to the period."""
        cfg = self.config
        n = len(dh)
        labels = np.full(n, "D", dtype="<U1")
        dead_rules = cfg.enable_dead_ball

        state = "D"  # every period starts from a stoppage (kickoff)
        dwell = 0
        missing_run = 0
        loose_run = 0

        for t in range(n):
            # --- dead-ball evidence -------------------------------------------------------
            if valid[t]:
                missing_run = 0
            else:
                missing_run += 1

            if dead_rules and (oob[t] or stopped[t]):
                state, dwell, loose_run = "D", 0, 0
                labels[t] = "D"
                continue

            if not valid[t]:
                if dead_rules and missing_run > cfg.max_missing_hold_frames:
                    state, dwell = "D", 0
                elif not cfg.hold_on_missing_ball:
                    state, dwell = "D", 0
                labels[t] = state
                continue

            d_h, d_a = float(dh[t]), float(da[t])
            d_min = min(d_h, d_a)
            nearest = "H" if d_h <= d_a else "A"

            if d_min > 2.0 * cfg.distance_threshold:
                loose_run += 1
            else:
                loose_run = 0
            if dead_rules and loose_run >= cfg.prolonged_loose_dead_frames:
                state, dwell = "D", 0
                labels[t] = "D"
                continue

            # --- possession logic ---------------------------------------------------------
            if state == "D":
                radius = (
                    cfg.restart_search_window
                    if cfg.enable_restart_attribution
                    else cfg.distance_threshold
                )
                if d_min <= radius:
                    state, dwell = nearest, 0

            elif not cfg.enable_hysteresis:
                if d_min <= cfg.distance_threshold:
                    state = nearest
                elif not cfg.hold_on_loose_ball:
                    state = "D"

            else:
                other = "A" if state == "H" else "H"
                d_own = d_h if state == "H" else d_a
                d_opp = d_a if state == "H" else d_h
                if d_opp <= cfg.distance_threshold and d_opp < d_own - cfg.switch_margin_m:
                    dwell += 1
                    if dwell >= cfg.dwell_frames:
                        state, dwell = other, 0
                else:
                    dwell = 0
                if state in ("H", "A") and d_min > cfg.distance_threshold and not cfg.hold_on_loose_ball:
                    state, dwell = "D", 0

            labels[t] = state

        return labels

    # ------------------------------------------------------------------ post-processing

    def _apply_stoppage_bridging(self, labels: np.ndarray, hard_dead: np.ndarray) -> np.ndarray:
        """Fills short D gaps between two runs of the same team unless the ball was clearly dead."""
        cfg = self.config
        if not cfg.enable_stoppage_bridging or cfg.bridging_max_gap <= 0:
            return labels
        out = labels.copy()
        runs = _runs(out)
        for i in range(1, len(runs) - 1):
            lbl, s, e = runs[i]
            if lbl != "D":
                continue
            left, right = runs[i - 1][0], runs[i + 1][0]
            if (
                left == right
                and left in ("H", "A")
                and e - s + 1 <= cfg.bridging_max_gap
                and not hard_dead[s : e + 1].any()
            ):
                out[s : e + 1] = left
        return out

    def _apply_spell_pruning(self, labels: np.ndarray) -> np.ndarray:
        """Merges possessions shorter than `min_spell_frames` into their neighbours.

        A short run between two runs of the same team is absorbed by that team (a touch, not a
        change of possession). Otherwise it is absorbed by the preceding possession, or by the
        following one at the start of a play phase; only an isolated short run becomes D.
        """
        cfg = self.config
        if not cfg.enable_spell_pruning or cfg.min_spell_frames <= 1:
            return labels
        out = labels.copy()
        runs = _runs(out)
        for i, (lbl, s, e) in enumerate(runs):
            if lbl not in ("H", "A") or e - s + 1 >= cfg.min_spell_frames:
                continue
            prev = str(out[s - 1]) if s > 0 else "D"
            nxt = runs[i + 1][0] if i + 1 < len(runs) else "D"
            if prev in ("H", "A"):
                out[s : e + 1] = prev
            elif nxt in ("H", "A"):
                out[s : e + 1] = nxt
            else:
                out[s : e + 1] = "D"
        return out

    @staticmethod
    def _apply_rolling_smoothing(labels: np.ndarray, window_size: int) -> np.ndarray:
        """Centred majority vote over H / A / D (ties keep the current label)."""
        if window_size <= 1:
            return labels
        codes = {"H": 0, "A": 1, "D": 2}
        onehot = np.zeros((len(labels), 3))
        for lbl, k in codes.items():
            onehot[labels == lbl, k] = 1.0
        csum = np.vstack([np.zeros(3), np.cumsum(onehot, axis=0)])
        n = len(labels)
        idx = np.arange(n)
        half = window_size // 2
        lo = np.maximum(0, idx - half)
        hi = np.minimum(n, idx + (window_size - half))
        counts = csum[hi] - csum[lo]
        best = counts.max(axis=1)
        out = labels.copy()
        names = np.array(["H", "A", "D"])
        for k in range(3):
            unique_best = (counts[:, k] == best) & ((counts == best[:, None]).sum(axis=1) == 1)
            out[unique_best] = names[k]
        return out

    # ------------------------------------------------------------------ public API

    def predict(
        self,
        tracking_data: UnifiedTrackingData,
        half_bounds: Optional[Union[Dict[Any, Tuple[int, int]], List[Tuple[int, int]]]] = None,
    ) -> np.ndarray:
        """Returns one label per tracking frame: 'H', 'A' or 'D' (period-0 frames get 'D')."""
        n = tracking_data.num_frames
        labels = np.full(n, "D", dtype="<U1")
        if n == 0:
            return labels

        dist_h, dist_a, valid, oob = self._distances(tracking_data)
        stopped = self._stoppage_mask(tracking_data)
        periods = self._resolve_periods(n, tracking_data.periods, half_bounds)

        for p in np.unique(periods):
            if p == 0:
                continue
            idx = np.flatnonzero(periods == p)
            p_labels = self._process_period(
                dist_h[idx], dist_a[idx], valid[idx], oob[idx], stopped[idx]
            )
            hard_dead = oob[idx] | stopped[idx] if self.config.enable_dead_ball else np.zeros(len(idx), bool)
            p_labels = self._apply_stoppage_bridging(p_labels, hard_dead)
            p_labels = self._apply_spell_pruning(p_labels)
            if self.config.smoothing_window > 1:
                p_labels = self._apply_rolling_smoothing(p_labels, self.config.smoothing_window)
            labels[idx] = p_labels
        return labels


def predict_possession(
    tracking_data: UnifiedTrackingData,
    params: Optional[Dict[str, Any]] = None,
    half_bounds: Optional[Union[Dict[Any, Tuple[int, int]], List[Tuple[int, int]]]] = None,
) -> np.ndarray:
    """predict_possession(tracking_data, params, half_bounds) -> np.ndarray of 'H'/'A'/'D'."""
    config = PossessionConfig.from_dict(params)
    engine = ModularPossessionEngine(config=config)
    return engine.predict(tracking_data, half_bounds=half_bounds)

"""src/metrica_loader.py - Metrica Sports CSV Tracking Data Parser.

Parses Home and Away CSV tracking data (e.g., Metrica Sample Games 1 & 2),
extracts player jersey identifiers and coordinate pairs, synchronizes tables by frame index,
transforms normalized coordinates [0, 1] x [0, 1] to centered metric pitch coordinates (metres),
and packages the result into the standardized UnifiedTrackingData structure.
"""

from pathlib import Path
import csv
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.tracking import UnifiedTrackingData


def _detect_header_and_entities(
    filepath: Path,
) -> Tuple[int, Optional[List[str]], List[str]]:
    """Scans the beginning of a Metrica CSV file to dynamically determine:
    1. The 0-based row index where 'Period' and 'Frame' column descriptors appear.
    2. The raw entity labels (player numbers, ball) in the preceding row (if present).
    3. The column names list at the descriptor row.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        preview_rows = []
        for _ in range(12):
            try:
                preview_rows.append(next(reader))
            except StopIteration:
                break

    header_row_idx = -1
    for idx, row in enumerate(preview_rows):
        row_lower = [c.strip().lower() for c in row]
        if "period" in row_lower and "frame" in row_lower:
            header_row_idx = idx
            break

    if header_row_idx == -1:
        header_row_idx = 0

    entity_row = None
    if header_row_idx > 0:
        entity_row = preview_rows[header_row_idx - 1]

    col_names = preview_rows[header_row_idx] if header_row_idx < len(preview_rows) else []
    return header_row_idx, entity_row, col_names


def _extract_column_mapping(
    header_row_idx: int,
    entity_row: Optional[List[str]],
    col_names: List[str],
) -> Tuple[int, int, int, Dict[int, Tuple[int, int]], Tuple[int, int]]:
    """Maps player jerseys and ball channels to column indices.

    Returns:
        (period_col, frame_col, time_col, player_col_map, ball_cols)
        where player_col_map is {jersey_num: (x_col, y_col)},
        and ball_cols is (ball_x_col, ball_y_col).
    """
    period_col = 0
    frame_col = 1
    time_col = 2

    for idx, name in enumerate(col_names):
        nl = name.strip().lower()
        if nl == "period":
            period_col = idx
        elif nl == "frame":
            frame_col = idx
        elif "time" in nl:
            time_col = idx

    player_col_map: Dict[int, Tuple[int, int]] = {}
    ball_cols: Tuple[int, int] = (-1, -1)

    if entity_row is not None:
        c = 0
        while c < len(entity_row):
            cell = entity_row[c].strip()
            if not cell and c < len(col_names):
                cell = col_names[c].strip()

            cell_lower = cell.lower()
            if "ball" in cell_lower:
                ball_cols = (c, c + 1)
                c += 2
                continue

            match = re.search(r"(\d+)", cell)
            if match and ("player" in cell_lower or "home" in cell_lower or "away" in cell_lower or cell.isdigit()):
                jersey = int(match.group(1))
                player_col_map[jersey] = (c, c + 1)
                c += 2
                continue
            c += 1
    else:
        # Flat header structure (e.g. Home_11_x, Home_11_y, Ball_x, Ball_y)
        for idx, col in enumerate(col_names):
            col_clean = col.strip()
            cl = col_clean.lower()
            if "ball" in cl and cl.endswith(("_x", "x")):
                ball_x = idx
                ball_y = idx + 1 if idx + 1 < len(col_names) else idx
                ball_cols = (ball_x, ball_y)
            match = re.search(r"(?:player|home|away)?_?(\d+)", cl)
            if match and cl.endswith(("_x", "x")):
                jersey = int(match.group(1))
                player_col_map[jersey] = (idx, idx + 1)

    if ball_cols == (-1, -1) and len(col_names) >= 2:
        ball_cols = (len(col_names) - 2, len(col_names) - 1)

    return period_col, frame_col, time_col, player_col_map, ball_cols


def _load_single_team_csv(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    """Loads a single Metrica team CSV file into memory."""
    header_idx, entity_row, col_names = _detect_header_and_entities(filepath)
    period_col, frame_col, time_col, player_map, ball_cols = _extract_column_mapping(
        header_idx, entity_row, col_names
    )

    df = pd.read_csv(filepath, skiprows=header_idx + 1, header=None, dtype=str)

    periods = pd.to_numeric(df[period_col], errors="coerce").fillna(1).astype(np.int32).values
    frames = pd.to_numeric(df[frame_col], errors="coerce").fillna(0).astype(np.int64).values

    # Ball
    bx = pd.to_numeric(df[ball_cols[0]], errors="coerce").values.astype(np.float32)
    by = pd.to_numeric(df[ball_cols[1]], errors="coerce").values.astype(np.float32)
    ball_coords = np.column_stack([bx, by])

    # Players
    player_coords: Dict[int, np.ndarray] = {}
    for jersey, (cx, cy) in player_map.items():
        px = pd.to_numeric(df[cx], errors="coerce").values.astype(np.float32)
        py = pd.to_numeric(df[cy], errors="coerce").values.astype(np.float32)
        player_coords[jersey] = np.column_stack([px, py])

    return periods, frames, player_coords, ball_coords


def _normalize_to_metric(
    coords: np.ndarray,
    pitch_length: float,
    pitch_width: float,
    is_normalized: Optional[bool] = True,
) -> np.ndarray:
    """Transforms normalized [0, 1] coords to centered pitch coords in metres.

    Preserves NaNs.

    Args:
        coords: Spatial coordinates array of shape (..., 2).
        pitch_length: Field length in metres (default: 105.0).
        pitch_width: Field width in metres (default: 68.0).
        is_normalized:
            - True: Unconditionally apply affine transformation.
            - False: Coordinates are already metric metres; return unmodified.
            - None: Robustly auto-detect scale using dataset-level 95th percentile.
              If the 95th percentile of absolute coordinates > 10.0m, coordinates
              are treated as already metric. Otherwise, they are transformed.
    """
    out = coords.copy()
    valid_mask = ~np.isnan(coords[..., 0])
    if not np.any(valid_mask):
        return out.astype(np.float32)

    if is_normalized is False:
        # Caller explicitly indicated coordinates are already metric
        return out.astype(np.float32)

    if is_normalized is None:
        # Robust scale sniffing: check 95th percentile of valid absolute values.
        # In normalized coordinates (even with out-of-bounds ball at x=1.6), p95 <= 2.0.
        # In metric coordinates on a 105x68m pitch, p95 is typically > 20.0m.
        valid_vals = np.abs(coords[valid_mask])
        p95 = float(np.nanpercentile(valid_vals, 95))
        if p95 > 10.0:
            return out.astype(np.float32)

    # Affine transform:
    # x_m = (x_norm - 0.5) * pitch_length
    # y_m = -(y_norm - 0.5) * pitch_width
    out[..., 0] = (coords[..., 0] - 0.5) * pitch_length
    out[..., 1] = -1.0 * (coords[..., 1] - 0.5) * pitch_width
    return out.astype(np.float32)


def _detect_tracking_role(filename: str) -> Optional[str]:
    """Detects whether a tracking filename designates a Home or Away feed.

    Handles standard naming (e.g., 'Sample_Game_1_RawTrackingData_Away_Team.csv'),
    case variations ('AWAY', 'away', 'Away'), and compound fixture names
    (e.g., 'Home_vs_Away_Away.csv', 'Away_vs_Home_Home.csv').

    Returns:
        'home', 'away', or None if neither or indeterminable.
    """
    matches = list(re.finditer(r"(home|away)", filename, flags=re.IGNORECASE))
    if not matches:
        return None
    # In compound fixture names like Home_vs_Away_Away.csv, the last occurrence
    # designates the specific feed role rather than the match fixture prefix.
    return matches[-1].group(1).lower()


def _infer_counterpart_path(path_input: Path, target_role: str) -> Path:
    """Infers the companion tracking path (Home <-> Away) for a given tracking file.

    Crucially, string replacement is strictly scoped to `path_input.name`,
    preserving `path_input.parent` untouched to prevent parent directory mutations
    (e.g., '/home/user/' -> '/away/user/' or 'away_matches/' -> 'home_matches/').

    Handles case variations ('Away' -> 'Home', 'away' -> 'home', 'AWAY' -> 'HOME')
    and compound names ('Home_vs_Away_Away.csv' -> 'Home_vs_Away_Home.csv') by
    modifying only the role-determining occurrence in the filename.

    Args:
        path_input: Path to the provided tracking CSV file.
        target_role: The role to infer ('home' or 'away').

    Returns:
        The inferred companion Path (which may or may not exist on disk).
    """
    parent_dir = path_input.parent
    name = path_input.name

    matches = list(re.finditer(r"(home|away)", name, flags=re.IGNORECASE))
    if not matches:
        fallback_word = "Home" if target_role == "home" else "Away"
        return parent_dir / f"{path_input.stem}_{fallback_word}{path_input.suffix}"

    # Target the last occurrence, which designates the team feed in compound names
    last_match = matches[-1]
    matched_text = last_match.group(1)

    if target_role == "home":
        replacement = "Home" if matched_text.istitle() else ("HOME" if matched_text.isupper() else "home")
    else:
        replacement = "Away" if matched_text.istitle() else ("AWAY" if matched_text.isupper() else "away")

    # Replace strictly the last occurrence in the filename string
    new_name = name[:last_match.start()] + replacement + name[last_match.end():]
    primary_candidate = parent_dir / new_name

    # Check if primary candidate exists
    if primary_candidate.exists():
        return primary_candidate

    # Fallback 1: Case-insensitive scan in parent directory for cross-platform resilience (e.g. Linux ext4)
    if parent_dir.is_dir():
        target_lower = new_name.lower()
        try:
            for f in parent_dir.iterdir():
                if f.is_file() and f.name.lower() == target_lower:
                    return f
        except OSError:
            pass

    # Fallback 2: Global case-preserving replacement if single-token replacement did not find file
    if target_role == "home":
        global_target = name.replace("Away", "Home").replace("away", "home").replace("AWAY", "HOME")
    else:
        global_target = name.replace("Home", "Away").replace("home", "away").replace("HOME", "AWAY")
    global_candidate = parent_dir / global_target
    if global_candidate.exists():
        return global_candidate

    # Return primary candidate so standard FileNotFoundError displays the expected path
    return primary_candidate


def load_metrica_csv(
    game_dir_or_home_file: Union[str, Path],
    away_file: Optional[Union[str, Path]] = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    fps: float = 25.0,
    is_normalized: Optional[bool] = True,
    **kwargs,
) -> UnifiedTrackingData:
    """Loads and synchronizes Metrica Home and Away tracking CSV data into UnifiedTrackingData.

    Args:
        game_dir_or_home_file: Directory containing Home & Away CSVs or Home CSV file path.
        away_file: Away CSV file path (optional if game_dir_or_home_file is directory).
        pitch_length: Pitch length in metres (default: 105.0).
        pitch_width: Pitch width in metres (default: 68.0).
        fps: Sampling rate (default: 25.0).
        is_normalized: Flag indicating coordinate normalization regime.
        **kwargs: Additional keyword arguments ignored or passed through.

    Returns:
        UnifiedTrackingData object with pitch-centered metric coordinates.
    """
    path_input = Path(game_dir_or_home_file)

    if path_input.is_dir():
        all_csvs = sorted([f for f in path_input.glob("*.csv") if f.is_file()])
        home_candidates = [f for f in all_csvs if _detect_tracking_role(f.name) == "home"]
        away_candidates = [f for f in all_csvs if _detect_tracking_role(f.name) == "away"]

        # Fallback to substring matching if role detector found no matches
        if not home_candidates:
            home_candidates = [f for f in all_csvs if "home" in f.name.lower()]
        if not away_candidates:
            away_candidates = [f for f in all_csvs if "away" in f.name.lower()]

        if not home_candidates:
            raise FileNotFoundError(f"No Home tracking CSV found in directory: {path_input}")
        if not away_candidates:
            raise FileNotFoundError(f"No Away tracking CSV found in directory: {path_input}")

        home_path = home_candidates[0]
        away_path = away_candidates[0]
    else:
        # Single file provided: detect whether home or away was passed
        input_role = _detect_tracking_role(path_input.name)

        if input_role == "away":
            away_path = path_input
            if away_file is None:
                inferred_home = _infer_counterpart_path(away_path, target_role="home")
                if not inferred_home.exists():
                    raise FileNotFoundError(
                        f"Home tracking file not specified and inferred path does not exist: {inferred_home}"
                    )
                home_path = inferred_home
            else:
                home_path = Path(away_file)
        else:
            home_path = path_input
            if away_file is None:
                inferred_away = _infer_counterpart_path(home_path, target_role="away")
                if not inferred_away.exists():
                    raise FileNotFoundError(
                        f"Away tracking file not specified and inferred path does not exist: {inferred_away}"
                    )
                away_path = inferred_away
            else:
                away_path = Path(away_file)

    if not home_path.exists():
        raise FileNotFoundError(f"Home tracking file not found: {home_path}")
    if not away_path.exists():
        raise FileNotFoundError(f"Away tracking file not found: {away_path}")

    # Load separate feeds
    periods_h, frames_h, players_h, ball_h = _load_single_team_csv(home_path)
    periods_a, frames_a, players_a, ball_a = _load_single_team_csv(away_path)

    # Harmonize frame indices (union and sort)
    all_frames = np.unique(np.concatenate([frames_h, frames_a]))
    all_frames.sort()
    n_frames = len(all_frames)

    # Fast row lookup mapping
    frame_to_row = {f: idx for idx, f in enumerate(all_frames)}
    rows_h = np.array([frame_to_row[f] for f in frames_h], dtype=np.int64)
    rows_a = np.array([frame_to_row[f] for f in frames_a], dtype=np.int64)

    # Aligned periods
    periods = np.ones(n_frames, dtype=np.int32)
    periods[rows_h] = periods_h
    periods[rows_a] = periods_a

    # Harmonize ball: prefer home; fall back to away when home is NaN
    aligned_ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    aligned_ball[rows_a] = ball_a
    ball_h_valid = ~np.isnan(ball_h[:, 0])
    aligned_ball[rows_h[ball_h_valid]] = ball_h[ball_h_valid]

    # Harmonize players into (N, P_max, 2)
    p_home_keys = sorted(players_h.keys())
    p_away_keys = sorted(players_a.keys())
    p_max = max(len(p_home_keys), len(p_away_keys), 11)

    home_positions = np.full((n_frames, p_max, 2), np.nan, dtype=np.float32)
    away_positions = np.full((n_frames, p_max, 2), np.nan, dtype=np.float32)

    for slot, jersey in enumerate(p_home_keys):
        home_positions[rows_h, slot, :] = players_h[jersey]

    for slot, jersey in enumerate(p_away_keys):
        away_positions[rows_a, slot, :] = players_a[jersey]

    # Convert coordinates to pitch-centered metres synchronously across all feeds.
    # By specification, Metrica Sports CSV tracking is normalized in [0, 1].
    if is_normalized is None:
        all_valid_coords = []
        for arr in (home_positions, away_positions, aligned_ball):
            valid = ~np.isnan(arr[..., 0])
            if np.any(valid):
                all_valid_coords.append(np.abs(arr[valid]))
        if all_valid_coords:
            combined = np.concatenate(all_valid_coords)
            p95 = float(np.nanpercentile(combined, 95))
            should_normalize = bool(p95 <= 10.0)
        else:
            should_normalize = True
    else:
        should_normalize = bool(is_normalized)

    if should_normalize:
        home_positions = _normalize_to_metric(home_positions, pitch_length, pitch_width, is_normalized=True)
        away_positions = _normalize_to_metric(away_positions, pitch_length, pitch_width, is_normalized=True)
        aligned_ball = _normalize_to_metric(aligned_ball, pitch_length, pitch_width, is_normalized=True)
    else:
        home_positions = home_positions.astype(np.float32)
        away_positions = away_positions.astype(np.float32)
        aligned_ball = aligned_ball.astype(np.float32)

    home_track_ids = [f"Player{j}" for j in p_home_keys]
    away_track_ids = [f"Player{j}" for j in p_away_keys]

    return UnifiedTrackingData(
        frame_indices=all_frames,
        periods=periods,
        home_positions=home_positions,
        away_positions=away_positions,
        ball_position=aligned_ball,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        fps=fps,
        home_track_ids=home_track_ids,
        away_track_ids=away_track_ids,
    )

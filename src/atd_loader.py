"""src/atd_loader.py - Loader for FIFA/EPTS tracking (the competition ATD and Metrica Game 3).

The competition ATD is Metrica's FIFA/EPTS export: a metadata XML plus a raw text file, at the
paths games.asset_paths() builds:

    data/tactical/BAR-ATH_tactical_atd.xml
    data/tactical/BAR-ATH_tactical_atd.txt

Format facts this loader relies on (official DATA.md and starter/baseline.py):

  * Raw line: "<frame_id>:<slot>;<slot>;...;<slot>:<ball>", each group "x,y[,speed]", "NaN" for a
    missing observation. Coordinates are normalised 0-1.
  * One record per video frame: video_frame = frame_id - 1. No offsets are needed for the ATD.
  * The XML lists two <Team> elements (the first is the home side) and <Player> entries with a
    teamId.
  * Which player each slot belongs to is given by <DataFormatSpecification> blocks (one per frame
    range, each listing PlayerChannelRef x/y pairs in slot order). When the XML has no such
    blocks, slot i is the i-th <Player>, the convention the official starter/baseline.py uses.
    Metrica's public Sample Game 3 uses the same format with 11 specification blocks
    (substitutions), which is how this loader is exercised on real files.

The raw file is parsed directly instead of through kloppy: the official starter notes that loading
the competition file through kloppy "costs minutes and gigabytes", and the format is simple.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np

import games
from src.tracking import UnifiedTrackingData, normalized_to_metric_coordinates

FIELD_SEP = ":"
GROUP_SEP = ";"


def _local(tag: str) -> str:
    """Tag name without an XML namespace."""
    return tag.rsplit("}", 1)[-1]


@dataclass
class EPTSLayout:
    """What the metadata says about columns: one column per <Player>, plus slot->column maps."""

    player_ids: List[str]
    is_home: np.ndarray  # (n_players,) bool, per column
    specs: List[Tuple[int, int, np.ndarray]]  # (start_frame_id, end_frame_id, slot->column)
    frame_rate: Optional[float]
    provider_params: Dict[str, str]
    field_size: Optional[Tuple[float, float]]


def parse_epts_metadata(meta_path: Union[str, Path]) -> EPTSLayout:
    """Reads teams, players and slot layouts from an EPTS metadata XML."""
    root = ET.parse(str(meta_path)).getroot()
    elems = list(root.iter())

    team_ids = [e.get("id") for e in elems if _local(e.tag) == "Team" and e.get("id")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 <Team> entries, found {len(team_ids)}")
    players = [e for e in elems if _local(e.tag) == "Player" and e.get("id")]
    if not players:
        raise ValueError(f"{meta_path}: no <Player> entries")
    player_ids = [p.get("id") for p in players]
    column_of = {pid: i for i, pid in enumerate(player_ids)}
    is_home = np.array([p.get("teamId") == team_ids[0] for p in players], dtype=bool)

    # PlayerChannel id -> player id, x channels only (one per slot)
    x_channel_player = {
        e.get("id"): e.get("playerId")
        for e in elems
        if _local(e.tag) == "PlayerChannel" and e.get("channelId") == "x"
    }

    specs: List[Tuple[int, int, np.ndarray]] = []
    for spec in (e for e in elems if _local(e.tag) == "DataFormatSpecification"):
        cols = []
        for ref in spec.iter():
            if _local(ref.tag) != "PlayerChannelRef":
                continue
            pid = x_channel_player.get(ref.get("playerChannelId"))
            if pid is not None:
                cols.append(column_of[pid])
        start = int(spec.get("startFrame", 0) or 0)
        end = int(spec.get("endFrame", 10**12) or 10**12)
        specs.append((start, end, np.array(cols, dtype=np.int64)))

    frame_rate = None
    params: Dict[str, str] = {}
    field_size = None
    for e in elems:
        name = _local(e.tag)
        if name == "FrameRate" and e.text:
            frame_rate = float(e.text)
        elif name == "ProviderParameter":
            key = next((c.text for c in e if _local(c.tag) == "Name"), None)
            val = next((c.text for c in e if _local(c.tag) == "Value"), None)
            if key:
                params[key] = val or ""
        elif name == "FieldSize":
            dims = {_local(c.tag): c.text for c in e}
            try:
                field_size = (float(dims["Width"]), float(dims["Height"]))
            except (KeyError, TypeError, ValueError):
                field_size = None

    return EPTSLayout(player_ids, is_home, specs, frame_rate, params, field_size)


def _parse_xy(group: str) -> Tuple[float, float]:
    """(x, y) from an "x,y,..." group, NaN when the observation is missing."""
    if not group or group[0] == "N":
        return np.nan, np.nan
    fields = group.split(",")
    if len(fields) < 2:
        return np.nan, np.nan
    try:
        return float(fields[0]), float(fields[1])
    except ValueError:
        return np.nan, np.nan


def read_epts_raw(
    raw_path: Union[str, Path], layout: EPTSLayout, n_frames: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Streams a raw EPTS file. Row r holds frame_id r + 1 (video_frame = frame_id - 1).

    Returns:
        players_xy: (n_frames, n_players, 2) float32, normalised, NaN where missing.
        ball_xy:    (n_frames, 2) float32, normalised, NaN where missing.
        present:    (n_frames,) bool, True where the raw file had a record for that frame.
    """
    if n_frames is None:
        last_id = 0
        with open(raw_path, "r", encoding="utf-8") as handle:
            for line in handle:
                sep = line.find(FIELD_SEP)
                if sep > 0:
                    last_id = max(last_id, int(line[:sep]))
        n_frames = last_id

    n_players = len(layout.player_ids)
    players_xy = np.full((n_frames, n_players, 2), np.nan, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    present = np.zeros(n_frames, dtype=bool)

    specs = sorted(layout.specs, key=lambda s: s[0])
    default_cols = np.arange(n_players, dtype=np.int64)
    spec_i = 0

    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame_id = int(parts[0])
            row = frame_id - 1
            if row < 0 or row >= n_frames:
                continue

            if specs:
                while spec_i + 1 < len(specs) and frame_id > specs[spec_i][1]:
                    spec_i += 1
                if not (specs[spec_i][0] <= frame_id <= specs[spec_i][1]):
                    match = [s for s in specs if s[0] <= frame_id <= s[1]]
                    if not match:
                        continue
                    cols = match[0][2]
                else:
                    cols = specs[spec_i][2]
            else:
                cols = default_cols

            groups = parts[1].split(GROUP_SEP)
            if len(groups) != len(cols):
                raise ValueError(
                    f"{raw_path}: frame {frame_id} has {len(groups)} slot groups, "
                    f"metadata declares {len(cols)}"
                )
            for col, group in zip(cols, groups):
                if group and group[0] != "N":
                    players_xy[row, col] = _parse_xy(group)
            ball_xy[row] = _parse_xy(parts[2])
            present[row] = True

    return players_xy, ball_xy, present


def to_unified(
    players_xy: np.ndarray,
    ball_xy: np.ndarray,
    is_home: np.ndarray,
    periods: np.ndarray,
    pitch: Tuple[float, float],
    fps: float,
    player_ids: Optional[List[str]] = None,
) -> UnifiedTrackingData:
    """Normalised EPTS arrays -> UnifiedTrackingData in pitch-centred metres."""
    length, width = pitch
    ids = player_ids or [f"slot_{i + 1}" for i in range(len(is_home))]

    def metric(arr: np.ndarray) -> np.ndarray:
        x, y = normalized_to_metric_coordinates(arr[..., 0], arr[..., 1], length, width)
        return np.stack([x, y], axis=-1).astype(np.float32)

    n = len(ball_xy)
    return UnifiedTrackingData(
        frame_indices=np.arange(n, dtype=np.int64),
        periods=periods.astype(np.int32),
        home_positions=metric(players_xy[:, is_home, :]),
        away_positions=metric(players_xy[:, ~is_home, :]),
        ball_position=metric(ball_xy),
        fps=float(fps),
        pitch_length=float(length),
        pitch_width=float(width),
        home_track_ids=[ids[i] for i in np.flatnonzero(is_home)],
        away_track_ids=[ids[i] for i in np.flatnonzero(~is_home)],
    )


def periods_for_video_frames(n_frames: int, game: str = games.GAME) -> np.ndarray:
    """Period id per video frame: 1 / 2 inside the graded halves, 0 elsewhere."""
    periods = np.zeros(n_frames, dtype=np.int32)
    for half, first, last in games.half_video_blocks(game):
        periods[max(0, first) : min(n_frames, last + 1)] = half
    return periods


def load_atd(
    data_dir: Optional[str] = None,
    game: str = games.GAME,
    source: str = "tactical",
) -> UnifiedTrackingData:
    """Loads the competition ATD into a UnifiedTrackingData indexed by video frame 0..N-1.

    Periods are 1 / 2 on the graded halves (games.half_video_blocks) and 0 elsewhere.
    Raises IOError (via games.require) with a readable message when the data is missing. There is
    deliberately no synthetic fallback: a missing file must stop the pipeline.
    """
    if source != "tactical":
        raise ValueError(
            "Only the tactical source is supported: the submission is indexed to the tactical "
            "video and the broadcast feed is not frame-aligned to it."
        )
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir, source)
    n_frames = games.source_frames(source, game)
    layout = parse_epts_metadata(meta_path)
    players_xy, ball_xy, present = read_epts_raw(raw_path, layout, n_frames)
    if not present.any():
        raise ValueError(f"{raw_path}: no tracking records parsed")
    return to_unified(
        players_xy,
        ball_xy,
        layout.is_home,
        periods_for_video_frames(n_frames, game),
        games.GAMES[game]["pitch"],
        float(games.GAMES[game]["fps"]),
        layout.player_ids,
    )

"""src/tracking.py - Unified Tracking Data Representation and Coordinate Transforms.

Defines the standardized container for match tracking data across all formats
(Metrica CSV, FIFA EPTS XML/TXT) and affine coordinate transformations between
normalized pitch space [0, 1] x [0, 1] and Cartesian pitch-centered metric space
[-L/2, L/2] x [-W/2, W/2] in metres.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass
class UnifiedTrackingData:
    """Standardized, in-memory representation of match tracking data.

    All spatial coordinates are in metres, centered at the pitch center (0.0, 0.0):
      - x in [-pitch_length/2, pitch_length/2] (lengthwise: default [-52.5, 52.5])
      - y in [-pitch_width/2, pitch_width/2]   (widthwise:  default [-34.0, 34.0])

    Missing or occluded ball positions are strictly represented as np.nan.
    Unoccupied or benched player slots are filled with np.nan.
    """

    frame_indices: np.ndarray  # Shape (N,), dtype int64
    periods: np.ndarray  # Shape (N,), dtype int32 (1 or 2)
    home_positions: np.ndarray  # Shape (N, P_home_max, 2), dtype float32 (metres)
    away_positions: np.ndarray  # Shape (N, P_away_max, 2), dtype float32 (metres)
    ball_position: np.ndarray  # Shape (N, 2), dtype float32 (metres, np.nan when missing)
    fps: float = 25.0
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    home_track_ids: List[str] = field(default_factory=list)
    away_track_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Type enforcement
        if not isinstance(self.frame_indices, np.ndarray) or self.frame_indices.dtype != np.int64:
            self.frame_indices = np.asarray(self.frame_indices, dtype=np.int64)

        if not isinstance(self.periods, np.ndarray) or self.periods.dtype != np.int32:
            self.periods = np.asarray(self.periods, dtype=np.int32)

        if not isinstance(self.home_positions, np.ndarray) or self.home_positions.dtype != np.float32:
            self.home_positions = np.asarray(self.home_positions, dtype=np.float32)

        if not isinstance(self.away_positions, np.ndarray) or self.away_positions.dtype != np.float32:
            self.away_positions = np.asarray(self.away_positions, dtype=np.float32)

        if not isinstance(self.ball_position, np.ndarray) or self.ball_position.dtype != np.float32:
            self.ball_position = np.asarray(self.ball_position, dtype=np.float32)

        # Dimension validation
        n = len(self.frame_indices)
        if len(self.periods) != n:
            raise ValueError(
                f"periods length {len(self.periods)} does not match frame_indices length {n}"
            )
        if self.home_positions.ndim != 3 or self.home_positions.shape[0] != n or self.home_positions.shape[2] != 2:
            raise ValueError(
                f"home_positions shape {self.home_positions.shape} must be ({n}, P_max, 2)"
            )
        if self.away_positions.ndim != 3 or self.away_positions.shape[0] != n or self.away_positions.shape[2] != 2:
            raise ValueError(
                f"away_positions shape {self.away_positions.shape} must be ({n}, P_max, 2)"
            )
        if self.ball_position.ndim != 2 or self.ball_position.shape != (n, 2):
            raise ValueError(
                f"ball_position shape {self.ball_position.shape} must be ({n}, 2)"
            )

    @property
    def num_frames(self) -> int:
        """Total number of tracking frames."""
        return len(self.frame_indices)

    @property
    def ball_visible_mask(self) -> np.ndarray:
        """Boolean mask of shape (N,) indicating frames where 2D ball position is valid (neither x nor y is NaN)."""
        return ~np.isnan(self.ball_position).any(axis=1)

    @property
    def ball_missing_mask(self) -> np.ndarray:
        """Boolean mask of shape (N,) indicating frames where 2D ball position is missing (either x or y is NaN)."""
        return np.isnan(self.ball_position).any(axis=1)


def normalized_to_metric_coordinates(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    invert_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms normalized [0, 1] coordinates to pitch-centered metric coordinates in metres.

    By Metrica convention:
      - (0, 0) is top-left corner
      - (0.5, 0.5) is pitch center spot
      - (1, 1) is bottom-right corner

    Transformed Cartesian coordinates:
      - (0.0, 0.0) is pitch center
      - (-L/2, +W/2) is top-left corner (with invert_y=True)
      - (+L/2, -W/2) is bottom-right corner (with invert_y=True)

    Args:
        x: Normalized x coordinate(s) in [0, 1].
        y: Normalized y coordinate(s) in [0, 1].
        pitch_length: Pitch length in metres (default: 105.0).
        pitch_width: Pitch width in metres (default: 68.0).
        invert_y: Whether to invert screen-downwards y-axis so positive y points up.

    Returns:
        Tuple of (x_metric, y_metric) as np.float32 arrays.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)

    x_metric = (x_arr - 0.5) * pitch_length
    if invert_y:
        y_metric = -1.0 * (y_arr - 0.5) * pitch_width
    else:
        y_metric = (y_arr - 0.5) * pitch_width

    return x_metric.astype(np.float32), y_metric.astype(np.float32)


def metric_to_normalized_coordinates(
    x_metric: Union[float, np.ndarray],
    y_metric: Union[float, np.ndarray],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    invert_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms pitch-centered metric coordinates in metres back to normalized [0, 1].

    Args:
        x_metric: Centered metric x coordinate(s) in metres.
        y_metric: Centered metric y coordinate(s) in metres.
        pitch_length: Pitch length in metres (default: 105.0).
        pitch_width: Pitch width in metres (default: 68.0).
        invert_y: Whether y was inverted during forward transformation.

    Returns:
        Tuple of (x_normalized, y_normalized) as np.float32 arrays.
    """
    xm_arr = np.asarray(x_metric, dtype=np.float32)
    ym_arr = np.asarray(y_metric, dtype=np.float32)

    x_norm = (xm_arr / pitch_length) + 0.5
    if invert_y:
        y_norm = (-1.0 * ym_arr / pitch_width) + 0.5
    else:
        y_norm = (ym_arr / pitch_width) + 0.5

    return x_norm.astype(np.float32), y_norm.astype(np.float32)


def load_tracking(
    game_dir_or_path: Union[str, Path],
    format_type: str = "auto",
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    fps: float = 25.0,
    **kwargs,
) -> "UnifiedTrackingData":
    """Convenience alias importing and executing load_tracking from src.data_loader."""
    from src.data_loader import load_tracking as _loader

    return _loader(
        game_dir_or_path=game_dir_or_path,
        format_type=format_type,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        fps=fps,
        **kwargs,
    )


def load_tracking_data(
    path: Union[str, Path],
    format_type: str = "auto",
    pitch_dimensions: Optional[Tuple[float, float]] = None,
    fps: float = 25.0,
    **kwargs,
) -> "UnifiedTrackingData":
    """Convenience alias importing and executing load_tracking_data from src.data_loader.

    """
    from src.data_loader import load_tracking_data as _loader

    return _loader(
        path=path,
        format_type=format_type,
        pitch_dimensions=pitch_dimensions,
        fps=fps,
        **kwargs,
    )

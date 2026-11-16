"""src/data_loader.py - Unified Tracking Data Loader Factory.

Provides a unified interface for loading tracking datasets across all formats
(Metrica CSV, FIFA EPTS XML/TXT) with automated format detection and
centered metric coordinate normalization.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

from src.tracking import UnifiedTrackingData


def load_tracking(
    game_dir_or_path: Union[str, Path],
    format_type: str = "auto",
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    fps: float = 25.0,
    **kwargs,
) -> UnifiedTrackingData:
    """Loads tracking dataset with automatic or explicit format resolution.

    Args:
        game_dir_or_path: Path to directory containing match files or specific tracking file.
        format_type: 'auto', 'metrica_csv', or 'epts'.
        pitch_length: Field length in metres (default: 105.0).
        pitch_width: Field width in metres (default: 68.0).
        fps: Sampling rate (default: 25.0).
        **kwargs: Additional parameters forwarded to specific loaders
            (e.g., away_file, txt_file, half_offsets, is_normalized).

    Returns:
        UnifiedTrackingData object in centered metric pitch coordinates (metres).

    Raises:
        ValueError: If format_type is unsupported or cannot be detected.
        FileNotFoundError: If target path or required files do not exist.
    """
    path = Path(game_dir_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Specified tracking path does not exist: {path}")

    fmt = format_type.strip().lower()
    valid_formats = {"auto", "metrica_csv", "epts"}
    if fmt not in valid_formats:
        raise ValueError(
            f"Unsupported format_type: '{format_type}'. Expected one of {sorted(valid_formats)}"
        )

    # Automatic format detection
    if fmt == "auto":
        if path.is_dir():
            csv_files = list(path.glob("*.csv"))
            has_home_csv = any("home" in f.name.lower() for f in csv_files)
            has_away_csv = any("away" in f.name.lower() for f in csv_files)
            xml_files = list(path.glob("*.xml"))
            txt_files = list(path.glob("*.txt"))
            has_xml = bool(xml_files)
            has_txt = bool(txt_files)

            # Rigorous disambiguation:
            # 1. Metrica CSV check first: presence of Home or Away CSV files
            if has_home_csv or has_away_csv:
                fmt = "metrica_csv"
            # 2. EPTS check: requires BOTH metadata XML and tracking TXT
            elif has_xml and has_txt:
                fmt = "epts"
            # 3. Fallback: CSV files present without EPTS XML
            elif bool(csv_files) and not has_xml:
                fmt = "metrica_csv"
            # 4. Partial EPTS directory (XML or TXT only, without CSV)
            elif has_xml or has_txt:
                fmt = "epts"
            else:
                raise ValueError(
                    f"Unable to auto-detect tracking format in directory: {path}. "
                    "No valid Metrica CSV (*Home*.csv) or EPTS (.xml and .txt) files found."
                )
        else:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                fmt = "metrica_csv"
            elif suffix in {".xml", ".txt"}:
                fmt = "epts"
            else:
                raise ValueError(
                    f"Unable to auto-detect tracking format from file suffix '{suffix}': {path}. "
                    "Expected '.csv' for Metrica or '.xml'/'.txt' for EPTS."
                )

    # Route to specialized loaders with kwargs passthrough
    if fmt == "metrica_csv":
        from src.metrica_loader import load_metrica_csv

        return load_metrica_csv(
            game_dir_or_home_file=path,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            fps=fps,
            **kwargs,
        )
    elif fmt == "epts":
        import numpy as np
        from src.atd_loader import parse_epts_metadata, read_epts_raw, to_unified

        folder = path if path.is_dir() else path.parent
        xml_files = sorted(folder.glob("*.xml"))
        txt_files = sorted(folder.glob("*.txt"))
        if len(xml_files) != 1 or len(txt_files) != 1:
            raise ValueError(f"{folder}: expected exactly one EPTS .xml and one .txt")
        layout = parse_epts_metadata(xml_files[0])
        players, ball, present = read_epts_raw(txt_files[0], layout)
        periods = np.where(present, 1, 0).astype(np.int32)
        return to_unified(players, ball, layout.is_home, periods,
                          (pitch_length, pitch_width), layout.frame_rate or fps,
                          layout.player_ids)
    else:
        raise ValueError(f"Unrecognized resolved format: {fmt}")


def load_tracking_data(
    path: Union[str, Path],
    format_type: str = "auto",
    pitch_dimensions: Optional[Tuple[float, float]] = None,
    fps: float = 25.0,
    **kwargs,
) -> UnifiedTrackingData:
    """Loads tracking from a path with explicit pitch dimensions.

    Adapts (path, format_type, pitch_dimensions) to load_tracking.
    """
    length = pitch_dimensions[0] if pitch_dimensions is not None else 105.0
    width = pitch_dimensions[1] if pitch_dimensions is not None else 68.0
    return load_tracking(
        game_dir_or_path=path,
        format_type=format_type,
        pitch_length=length,
        pitch_width=width,
        fps=fps,
        **kwargs,
    )

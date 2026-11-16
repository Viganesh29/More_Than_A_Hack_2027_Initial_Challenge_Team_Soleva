"""src/params_loader.py - Loads the frozen parameter manifest (params_tuned.json).

The manifest is written by scripts/external_cv.py. Only its "parameters" block drives the model;
the "external_cv" block is informational. Validation checks that every parameter is a known
PossessionConfig field with a sensible type and value. It deliberately does NOT require any score
to be present: scores are reported, never demanded.
"""

import json
from pathlib import Path
from typing import Any, Dict, Union

from src.possession_model import PossessionConfig


class ParameterValidationError(ValueError):
    """Raised when params_tuned.json is malformed."""


# (min, max) for numeric parameters that must be bounded; others only need the right type.
BOUNDS = {
    "distance_threshold": (0.3, 10.0),
    "pitch_margin": (0.0, 10.0),
    "dwell_frames": (1, 250),
    "switch_margin_m": (0.0, 5.0),
    "max_missing_hold_frames": (0, 10000),
    "prolonged_loose_dead_frames": (1, 10000),
    "stoppage_speed_mps": (0.0, 5.0),
    "stoppage_min_frames": (1, 10000),
    "bridging_max_gap": (0, 1000),
    "min_spell_frames": (0, 1000),
    "smoothing_window": (0, 1001),
    "ball_interp_max_gap": (0, 10000),
}


def validate_parameters(params: Dict[str, Any]) -> None:
    """Raises ParameterValidationError on unknown keys, wrong types or out-of-range values."""
    if not isinstance(params, dict):
        raise ParameterValidationError("'parameters' must be a JSON object")
    fields = PossessionConfig.__dataclass_fields__
    for key, value in params.items():
        if key not in fields:
            raise ParameterValidationError(f"unknown parameter '{key}'")
        ftype = fields[key].type
        if ftype in (bool, "bool"):
            if not isinstance(value, bool):
                raise ParameterValidationError(f"'{key}' must be true/false, got {value!r}")
        elif ftype in (int, "int"):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ParameterValidationError(f"'{key}' must be an integer, got {value!r}")
        elif ftype in (float, "float"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ParameterValidationError(f"'{key}' must be a number, got {value!r}")
        if key in BOUNDS:
            lo, hi = BOUNDS[key]
            if not lo <= value <= hi:
                raise ParameterValidationError(f"'{key}'={value} outside [{lo}, {hi}]")


def load_params_tuned(manifest_path: Union[str, Path] = "params_tuned.json") -> Dict[str, Any]:
    """Returns the validated "parameters" block of the manifest."""
    path = Path(manifest_path)
    if not path.exists():
        candidate = Path(__file__).resolve().parent.parent / manifest_path
        if not candidate.exists():
            raise FileNotFoundError(f"parameter manifest not found: {path.resolve()}")
        path = candidate
    try:
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as error:
        raise ParameterValidationError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(manifest, dict) or "parameters" not in manifest:
        raise ParameterValidationError(f"{path}: missing top-level 'parameters'")
    validate_parameters(manifest["parameters"])
    return manifest["parameters"]

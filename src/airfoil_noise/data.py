"""Download, parse, clean and validate the UCI Airfoil Self-Noise dataset."""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

RAW_COLUMNS = [
    "frequency_hz",
    "angle_of_attack_deg",
    "chord_length_m",
    "free_stream_velocity_ms",
    "suction_side_displacement_thickness_m",
    "scaled_sound_pressure_level_db",
]
TARGET = RAW_COLUMNS[-1]
FEATURES = RAW_COLUMNS[:-1]
LOG_FREQUENCY = "log_frequency_hz"
FEATURES_WITH_LOG = [*FEATURES, LOG_FREQUENCY]
EXPECTED_ROWS = 1503

DATA_URL = "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip"
FALLBACK_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "airfoil_self_noise.dat"

FEATURE_META = {
    "frequency_hz": {"label": "Frequency", "unit": "Hz"},
    "angle_of_attack_deg": {"label": "Angle of attack", "unit": "deg"},
    "chord_length_m": {"label": "Chord length", "unit": "m"},
    "free_stream_velocity_ms": {"label": "Free-stream velocity", "unit": "m/s"},
    LOG_FREQUENCY: {"label": "log(Frequency)", "unit": "log(Hz)"},
    "suction_side_displacement_thickness_m": {
        "label": "Displacement thickness",
        "unit": "m",
    },
}
TARGET_META = {"label": "Scaled sound pressure level", "unit": "dB"}

# Broad physical bounds; anything outside is treated as an invalid record.
VALID_RANGES = {
    "frequency_hz": (1.0, 100_000.0),
    "angle_of_attack_deg": (-30.0, 30.0),
    "chord_length_m": (1e-3, 2.0),
    "free_stream_velocity_ms": (1.0, 250.0),
    "suction_side_displacement_thickness_m": (1e-6, 1.0),
    "scaled_sound_pressure_level_db": (60.0, 180.0),
}


def download_raw(dest: Path = RAW_PATH, force: bool = False) -> Path:
    """Download the dataset to ``dest`` unless it is already cached."""
    dest = Path(dest)
    if dest.exists() and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    try:
        _fetch(DATA_URL, partial)
    except Exception as primary_error:
        LOGGER.warning("Primary download failed (%s). Trying fallback URL.", primary_error)
        try:
            _fetch(FALLBACK_URL, partial)
        except Exception as fallback_error:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the dataset automatically. Download it manually "
                f"and save it as {dest}."
            ) from fallback_error

    partial.replace(dest)
    return dest


def _fetch(url: str, target: Path) -> None:
    if url.endswith(".zip"):
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [name for name in archive.namelist() if name.endswith(".dat")]
            if not members:
                raise ValueError("Dataset archive does not contain a .dat file.")
            target.write_bytes(archive.read(members[0]))
    else:
        urllib.request.urlretrieve(url, target)


def load_data(path: Path = RAW_PATH, force_download: bool = False) -> pd.DataFrame:
    """Download if needed, parse, and validate the raw dataset."""
    raw_path = download_raw(path, force=force_download)
    frame = pd.read_csv(raw_path, sep=r"\s+", header=None, names=RAW_COLUMNS)
    validate_schema(frame)
    return frame


def validate_schema(frame: pd.DataFrame) -> None:
    if list(frame.columns) != RAW_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(frame.columns)}")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(frame)}.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError("All columns must be numeric.")


def clean_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Coerce values, enforce physical ranges and drop duplicate records.

    Returns the cleaned frame and a row-count log for the report and web page.
    """
    report = {"raw_rows": int(len(frame))}

    numeric = frame.apply(lambda column: pd.to_numeric(column, errors="coerce"))
    invalid = numeric.isna().any(axis=1)
    for column, (low, high) in VALID_RANGES.items():
        invalid = invalid | ~numeric[column].between(low, high)
    report["invalid_or_missing"] = int(invalid.sum())

    cleaned = numeric.loc[~invalid]
    duplicates = cleaned.duplicated(keep="first")
    report["duplicates"] = int(duplicates.sum())
    cleaned = cleaned.loc[~duplicates].reset_index(drop=True)
    report["final_rows"] = int(len(cleaned))

    if report["final_rows"] != report["raw_rows"]:
        removed = report["raw_rows"] - report["final_rows"]
        LOGGER.info("Cleaning removed %d rows: %s", removed, report)
    return cleaned, report


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.loc[:, FEATURES].copy(), frame.loc[:, TARGET].copy()


def add_log_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(**{LOG_FREQUENCY: np.log(frame["frequency_hz"])})

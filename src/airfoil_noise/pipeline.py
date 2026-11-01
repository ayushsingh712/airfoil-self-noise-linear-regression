"""End-to-end pipeline: data, models, reports and the static web payload."""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from pathlib import Path

import joblib
import pandas as pd

from airfoil_noise.data import (
    FEATURES,
    FEATURES_WITH_LOG,
    FEATURE_META,
    PROJECT_ROOT,
    RAW_COLUMNS,
    RAW_PATH,
    TARGET,
    TARGET_META,
    add_log_frequency,
    clean_data,
    load_data,
    split_xy,
)
from airfoil_noise.modeling import (
    SEED,
    coefficient_table,
    cross_validate_r2,
    evaluate,
    make_pipeline,
    split,
    tune_regularized,
)
from airfoil_noise.plots import (
    plot_actual_vs_predicted,
    plot_coefficients,
    plot_correlation_heatmap,
    plot_feature_vs_target,
    plot_model_comparison,
    plot_residual_distribution,
    plot_residuals_vs_fitted,
)

LOGGER = logging.getLogger(__name__)

PREVIEW_ROWS = 12
PREVIEW_DECIMALS = {
    "frequency_hz": 0,
    "angle_of_attack_deg": 1,
    "chord_length_m": 4,
    "free_stream_velocity_ms": 1,
    "suction_side_displacement_thickness_m": 5,
    "scaled_sound_pressure_level_db": 1,
}


def run(
    data_path: Path = RAW_PATH,
    reports_dir: Path = PROJECT_ROOT / "reports",
    models_dir: Path = PROJECT_ROOT / "models",
    web_dir: Path = PROJECT_ROOT / "web",
    seed: int = SEED,
    force_download: bool = False,
) -> dict:
    """Clean the data, fit and compare models, then write every artifact."""
    reports_dir = Path(reports_dir)
    models_dir = Path(models_dir)
    web_dir = Path(web_dir)

    raw = load_data(data_path, force_download=force_download)
    frame, cleaning = clean_data(raw)
    LOGGER.info(
        "Loaded %d raw rows; %d remain after cleaning",
        cleaning["raw_rows"],
        cleaning["final_rows"],
    )

    features, target = split_xy(frame)
    x_train, x_test, y_train, y_test = split(features, target, random_state=seed)

    ols = make_pipeline().fit(x_train, y_train)
    ols_metrics = evaluate(ols, x_test, y_test)
    cv_mean, cv_std = cross_validate_r2(ols, features, target, random_state=seed)
    coefficients = coefficient_table(ols, FEATURES)

    comparison = tune_regularized(x_train, y_train, x_test, y_test, seed=seed)
    comparison = pd.concat(
        [
            pd.DataFrame([{"model": "OLS", "alpha": math.nan, **ols_metrics}]),
            comparison,
        ],
        ignore_index=True,
    )

    log_frame = add_log_frequency(frame)
    log_features = log_frame.loc[:, FEATURES_WITH_LOG]
    x_log_train, x_log_test, y_log_train, y_log_test = split(
        log_features, target, random_state=seed
    )
    log_model = make_pipeline().fit(x_log_train, y_log_train)
    log_metrics = evaluate(log_model, x_log_test, y_log_test)
    log_cv_mean, log_cv_std = cross_validate_r2(
        log_model, log_features, target, random_state=seed
    )

    predictions = ols.predict(x_test)
    residuals = y_test.to_numpy() - predictions

    figure_dir = reports_dir / "figures"
    figures = {
        "feature_overview": plot_feature_vs_target(
            frame, figure_dir / "feature_vs_target.png"
        ),
        "correlation": plot_correlation_heatmap(
            frame, figure_dir / "correlation_heatmap.png"
        ),
        "actual_vs_predicted": plot_actual_vs_predicted(
            y_test, predictions, figure_dir / "actual_vs_predicted.png"
        ),
        "residuals_vs_fitted": plot_residuals_vs_fitted(
            predictions, residuals, figure_dir / "residuals_vs_fitted.png"
        ),
        "residual_distribution": plot_residual_distribution(
            residuals, figure_dir / "residual_distribution.png"
        ),
        "coefficients": plot_coefficients(
            coefficients, figure_dir / "coefficients.png"
        ),
        "model_comparison": plot_model_comparison(
            comparison, figure_dir / "model_comparison.png"
        ),
    }

    comparison_payload = _comparison_payload(comparison)
    log_gain = log_metrics["r2"] - ols_metrics["r2"]
    metrics = {
        "dataset": {
            "rows_raw": cleaning["raw_rows"],
            "rows_clean": cleaning["final_rows"],
            "features": FEATURES,
            "target": TARGET,
        },
        "seed": seed,
        "ols": {**ols_metrics, "cv_r2_mean": cv_mean, "cv_r2_std": cv_std},
        "log_frequency": {
            "features": FEATURES_WITH_LOG,
            **log_metrics,
            "cv_r2_mean": log_cv_mean,
            "cv_r2_std": log_cv_std,
            "r2_gain": log_gain,
        },
        "comparison": comparison_payload,
        "cleaning": cleaning,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "linear_ols.joblib"
    joblib.dump(
        {"pipeline": ols, "features": list(FEATURES), "target": TARGET, "seed": seed},
        model_path,
    )

    payload = _web_payload(
        frame=frame,
        cleaning=cleaning,
        coefficients=coefficients,
        comparison=comparison_payload,
        ols=ols,
        ols_metrics=ols_metrics,
        cv_mean=cv_mean,
        cv_std=cv_std,
        log_metrics=log_metrics,
        log_cv_mean=log_cv_mean,
        log_cv_std=log_cv_std,
        seed=seed,
    )
    assets_dir = web_dir / "assets" / "figures"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for figure in figures.values():
        shutil.copy2(figure, assets_dir / figure.name)
    web_data_path = web_dir / "model-data.js"
    web_data_path.write_text(
        "// Generated by `uv run airfoil-run`; do not edit by hand.\n"
        f"window.AIRFOIL_MODEL = {json.dumps(payload, indent=2)};\n",
        encoding="utf-8",
    )

    return {
        "cleaning": cleaning,
        "ols": {**ols_metrics, "cv_r2_mean": cv_mean, "cv_r2_std": cv_std},
        "log_frequency": {**log_metrics, "r2_gain": log_gain},
        "figures": {name: str(path) for name, path in figures.items()},
        "metrics_path": str(reports_dir / "metrics.json"),
        "model_path": str(model_path),
        "web_data_path": str(web_data_path),
    }


def _web_payload(
    *,
    frame: pd.DataFrame,
    cleaning: dict[str, int],
    coefficients: pd.DataFrame,
    comparison: list[dict],
    ols,
    ols_metrics: dict[str, float],
    cv_mean: float,
    cv_std: float,
    log_metrics: dict[str, float],
    log_cv_mean: float,
    log_cv_std: float,
    seed: int,
) -> dict:
    scaler = ols.named_steps["scaler"]
    linear = ols.named_steps["model"]

    feature_payload = []
    for feature, mean, scale, coef in zip(
        FEATURES, scaler.mean_, scaler.scale_, linear.coef_
    ):
        low = float(frame[feature].min())
        high = float(frame[feature].max())
        step = _nice_step(low, high)
        feature_payload.append(
            {
                "key": feature,
                "label": FEATURE_META[feature]["label"],
                "unit": FEATURE_META[feature]["unit"],
                "mean": float(mean),
                "scale": float(scale),
                "coef": float(coef),
                "min": low,
                "max": high,
                "step": step,
                "decimals": _display_decimals(low, step),
            }
        )

    coefficient_payload = [
        {
            "feature": row.feature,
            "label": FEATURE_META[row.feature]["label"],
            "unit": FEATURE_META[row.feature]["unit"],
            "coefficient": float(row.coefficient),
        }
        for row in coefficients.itertuples(index=False)
    ]

    preview_columns = [
        {
            "key": column,
            "label": FEATURE_META.get(column, TARGET_META)["label"],
            "unit": FEATURE_META.get(column, TARGET_META)["unit"],
            "decimals": PREVIEW_DECIMALS[column],
        }
        for column in RAW_COLUMNS
    ]
    preview_rows = [
        [
            round(float(value), PREVIEW_DECIMALS[column])
            for column, value in zip(RAW_COLUMNS, row)
        ]
        for row in frame.loc[: PREVIEW_ROWS - 1, RAW_COLUMNS].itertuples(
            index=False, name=None
        )
    ]

    return {
        "meta": {
            "title": "Airfoil Self-Noise",
            "rows": cleaning["final_rows"],
            "seed": seed,
        },
        "target": {
            "key": TARGET,
            **TARGET_META,
            "min": float(frame[TARGET].min()),
            "max": float(frame[TARGET].max()),
            "mean": float(frame[TARGET].mean()),
        },
        "features": feature_payload,
        "prediction": {
            "intercept": float(linear.intercept_),
            "rmse": ols_metrics["rmse"],
        },
        "metrics": {
            "test": ols_metrics,
            "cv_r2_mean": cv_mean,
            "cv_r2_std": cv_std,
        },
        "log_frequency": {
            "test": log_metrics,
            "cv_r2_mean": log_cv_mean,
            "cv_r2_std": log_cv_std,
            "r2_gain": log_metrics["r2"] - ols_metrics["r2"],
        },
        "comparison": comparison,
        "coefficients": coefficient_payload,
        "cleaning": cleaning,
        "preview": {"columns": preview_columns, "rows": preview_rows},
    }


def _comparison_payload(comparison: pd.DataFrame) -> list[dict]:
    payload = []
    for row in comparison.itertuples(index=False):
        payload.append(
            {
                "model": row.model,
                "alpha": None if pd.isna(row.alpha) else float(row.alpha),
                "r2": float(row.r2),
                "rmse": float(row.rmse),
                "mae": float(row.mae),
            }
        )
    return payload


def _nice_step(low: float, high: float) -> float:
    span = high - low
    if span <= 0:
        return 1.0
    raw = span / 100
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 5, 10):
        if raw <= multiple * magnitude:
            return multiple * magnitude
    return 10 * magnitude


def _decimals(step: float) -> int:
    if step >= 1:
        return 0
    return min(8, max(0, -math.floor(math.log10(step))))


def _display_decimals(low: float, step: float) -> int:
    decimals = _decimals(step)
    if 0 < low < 1:
        decimals = max(decimals, min(6, -math.floor(math.log10(low))))
    return decimals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=RAW_PATH)
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--web-dir", type=Path, default=PROJECT_ROOT / "web")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = run(
        data_path=args.data_path,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        web_dir=args.web_dir,
        seed=args.seed,
        force_download=args.force_download,
    )

    cleaning = summary["cleaning"]
    ols = summary["ols"]
    log_info = summary["log_frequency"]
    print()
    print(
        f"Rows      : {cleaning['raw_rows']} raw -> {cleaning['final_rows']} clean "
        f"({cleaning['duplicates']} duplicates, "
        f"{cleaning['invalid_or_missing']} invalid/missing)"
    )
    print(
        f"Test R²   : {ols['r2']:.3f}   RMSE: {ols['rmse']:.2f} dB   "
        f"MAE: {ols['mae']:.2f} dB"
    )
    print(f"5-fold CV : R² {ols['cv_r2_mean']:.3f} ± {ols['cv_r2_std']:.3f}")
    print(
        f"log(freq) : test R² {log_info['r2']:.3f} "
        f"({log_info['r2_gain']:+.3f} vs OLS)"
    )
    print(f"Artifacts : {summary['metrics_path']}, {summary['model_path']}")
    print(f"Web data  : {summary['web_data_path']}")
    return 0

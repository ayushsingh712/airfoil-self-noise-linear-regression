"""Dark-themed figure builders shared by the notebook and the web gallery."""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from airfoil_noise.data import FEATURES, FEATURE_META, TARGET, TARGET_META

BACKGROUND = "#0d1117"
TEXT = "#e6edf3"
MUTED = "#8b949e"
GRID = "#21262d"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
CORAL = "#f78166"
GOLD = "#d29922"

SHORT_LABELS = {
    "frequency_hz": "Frequency",
    "angle_of_attack_deg": "AoA",
    "chord_length_m": "Chord",
    "free_stream_velocity_ms": "Velocity",
    "suction_side_displacement_thickness_m": "Thickness",
    "scaled_sound_pressure_level_db": "SPL",
}
MODEL_COLORS = {"OLS": ACCENT, "Ridge": GREEN, "Lasso": GOLD}


def _pyplot():
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.alpha": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": TEXT,
        }
    )
    return plt


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    _pyplot().close(fig)
    return path


def _feature_label(key: str) -> str:
    meta = FEATURE_META[key]
    return f"{meta['label']} ({meta['unit']})"


def plot_feature_vs_target(frame: pd.DataFrame, path: Path) -> Path:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8))

    for ax, column in zip(axes.flat[:5], FEATURES):
        ax.scatter(frame[column], frame[TARGET], s=7, alpha=0.55, color=ACCENT, linewidths=0)
        if column == "frequency_hz":
            ax.set_xscale("log")
        ax.set_title(_feature_label(column), loc="left")
        ax.set_ylabel("SPL (dB)")
        ax.grid(True, linewidth=0.5)

    target_ax = axes.flat[5]
    target_ax.hist(frame[TARGET], bins=30, color=ACCENT, alpha=0.85, edgecolor=BACKGROUND)
    target_ax.set_title(f"{TARGET_META['label']} (dB)", loc="left")
    target_ax.set_xlabel("SPL (dB)")
    target_ax.set_ylabel("Count")
    target_ax.grid(True, axis="y", linewidth=0.5)

    fig.suptitle("How each input relates to sound pressure level", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def plot_correlation_heatmap(frame: pd.DataFrame, path: Path) -> Path:
    plt = _pyplot()
    correlation = frame.corr(numeric_only=True)
    labels = [SHORT_LABELS.get(column, column) for column in correlation.columns]

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    heatmap = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)

    for row in range(len(labels)):
        for column in range(len(labels)):
            value = correlation.iat[row, column]
            color = BACKGROUND if abs(value) < 0.55 else TEXT
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors=MUTED)
    colorbar.outline.set_visible(False)
    ax.set_title("Pearson correlation", loc="left")
    fig.tight_layout()
    return _save(fig, path)


def plot_actual_vs_predicted(
    target: pd.Series, predictions: np.ndarray, path: Path
) -> Path:
    plt = _pyplot()
    actual = np.asarray(target)
    predicted = np.asarray(predictions)
    low = min(actual.min(), predicted.min()) - 1
    high = max(actual.max(), predicted.max()) + 1

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.scatter(actual, predicted, s=12, alpha=0.6, color=ACCENT, linewidths=0)
    ax.plot([low, high], [low, high], linestyle="--", color=MUTED, linewidth=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Actual SPL (dB)")
    ax.set_ylabel("Predicted SPL (dB)")
    ax.set_title("Held-out predictions vs. actual values", loc="left")
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_residuals_vs_fitted(
    predictions: np.ndarray, residuals: np.ndarray, path: Path
) -> Path:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(predictions, residuals, s=12, alpha=0.6, color=CORAL, linewidths=0)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel("Fitted SPL (dB)")
    ax.set_ylabel("Residual (actual − predicted, dB)")
    ax.set_title("Residuals vs. fitted values", loc="left")
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_residual_distribution(residuals: np.ndarray, path: Path) -> Path:
    plt = _pyplot()
    residuals = np.asarray(residuals)
    fig, (hist_ax, qq_ax) = plt.subplots(1, 2, figsize=(9.0, 3.8))

    hist_ax.hist(residuals, bins=30, density=True, color=CORAL, alpha=0.75, edgecolor=BACKGROUND)
    grid = np.linspace(residuals.min(), residuals.max(), 200)
    normal = NormalDist(mu=float(residuals.mean()), sigma=float(residuals.std()))
    hist_ax.plot(grid, [normal.pdf(value) for value in grid], color=TEXT, linewidth=1.2)
    hist_ax.set_xlabel("Residual (dB)")
    hist_ax.set_ylabel("Density")
    hist_ax.set_title("Residual distribution", loc="left")
    hist_ax.grid(True, linewidth=0.5)

    ordered = np.sort(residuals)
    probabilities = (np.arange(1, len(ordered) + 1) - 0.5) / len(ordered)
    normal_dist = NormalDist()
    theoretical = np.array([normal_dist.inv_cdf(float(p)) for p in probabilities])
    slope, intercept = np.polyfit(theoretical, ordered, 1)
    qq_ax.scatter(theoretical, ordered, s=8, alpha=0.6, color=ACCENT, linewidths=0)
    qq_ax.plot(theoretical, slope * theoretical + intercept, color=TEXT, linewidth=1)
    qq_ax.set_xlabel("Theoretical normal quantile")
    qq_ax.set_ylabel("Ordered residual (dB)")
    qq_ax.set_title("Normal Q-Q plot", loc="left")
    qq_ax.grid(True, linewidth=0.5)

    fig.tight_layout()
    return _save(fig, path)


def plot_coefficients(table: pd.DataFrame, path: Path) -> Path:
    plt = _pyplot()
    ordered = table.iloc[::-1]
    labels = [
        f"{FEATURE_META[key]['label']} ({FEATURE_META[key]['unit']})"
        for key in ordered["feature"]
    ]
    colors = [ACCENT if value >= 0 else CORAL for value in ordered["coefficient"]]

    values = ordered["coefficient"]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    bars = ax.barh(labels, values, color=colors, height=0.6)
    ax.axvline(0, color=MUTED, linewidth=1)
    span = max(float(values.max() - values.min()), 1e-9)
    ax.set_xlim(float(values.min()) - 0.24 * span, float(values.max()) + 0.24 * span)
    ax.set_xlabel("Change in SPL per 1 standard deviation (dB)")
    ax.set_title("Standardised OLS coefficients", loc="left")
    ax.grid(True, axis="x", linewidth=0.5)
    ax.bar_label(bars, fmt="%+.2f", padding=5, color=TEXT, fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> Path:
    plt = _pyplot()
    metrics = [
        ("r2", "Test R²", "{:.3f}"),
        ("rmse", "RMSE (dB)", "{:.2f}"),
        ("mae", "MAE (dB)", "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.4))

    for ax, (column, title, fmt) in zip(axes, metrics):
        colors = [MODEL_COLORS.get(name, ACCENT) for name in comparison["model"]]
        bars = ax.bar(comparison["model"], comparison[column], color=colors, width=0.55)
        ax.set_title(title, loc="left")
        ax.grid(True, axis="y", linewidth=0.5)
        ax.margins(y=0.2)
        ax.bar_label(bars, fmt=fmt, padding=3, color=TEXT, fontsize=9)
        if column == "r2":
            ax.set_ylim(0, 1)

    fig.suptitle("Regularised models vs. plain OLS", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)

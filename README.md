# Airfoil Self-Noise — Linear Regression

Predict the **scaled sound pressure level (SPL, dB)** of an aerofoil from five aerodynamic
parameters using ordinary least squares, with a full audit trail from raw data to a static
in-browser demo.

This is a deliberately small, honest regression project: real wind-tunnel measurements, a
baseline that works reasonably well, and residual diagnostics that show exactly where a linear
model stops being enough.

## Dataset

[UCI Machine Learning Repository — Airfoil Self-Noise (ID 291)](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)

- 1,503 wind-tunnel measurements from NASA (Brooks, Pope & Marcolini).
- Five numeric inputs plus the target:

| Column | Meaning | Unit |
| --- | --- | --- |
| `frequency_hz` | Frequency | Hz |
| `angle_of_attack_deg` | Angle of attack | degrees |
| `chord_length_m` | Chord length | m |
| `free_stream_velocity_ms` | Free-stream velocity | m/s |
| `suction_side_displacement_thickness_m` | Suction-side displacement thickness | m |
| `scaled_sound_pressure_level_db` | **Target:** scaled SPL | dB |

- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Data is downloaded
  from the UCI repository at runtime and is not redistributed in this repository.

## Results

80/20 hold-out split (seed 42), features standardised inside a scikit-learn pipeline.
5-fold cross-validation uses shuffled folds with the same seed.

| Model | Test R² | RMSE (dB) | MAE (dB) | 5-fold CV R² |
| --- | --- | --- | --- | --- |
| **OLS, 5 features (baseline)** | **0.558** | **4.70** | **3.67** | **0.505 ± 0.052** |
| Ridge (α = 10) | 0.557 | 4.71 | 3.69 | — |
| Lasso (α = 0.001) | 0.558 | 4.70 | 3.67 | — |
| OLS + `log(frequency_hz)` | 0.566 | 4.66 | 3.64 | 0.517 ± 0.047 |

The model explains roughly 56% of hold-out variance. The gap between test and CV R² is about
0.05, consistent with measurement-level noise in the target rather than an unstable fit.

### What the coefficients say

Because inputs are standardised, each coefficient is the SPL change per one-standard-deviation
increase, holding the others fixed:

| Feature | Coefficient (dB / SD) |
| --- | --- |
| Frequency | −4.06 |
| Chord length | −3.22 |
| Angle of attack | −2.37 |
| Displacement thickness | −1.82 |
| Free-stream velocity | +1.53 |

Higher frequency, chord length, angle of attack and thickness all reduce SPL in the fitted
linear term; velocity increases it. Angle of attack is the least linear relationship in the
data (non-monotonic in the scatter plots), so its coefficient is best read as an average slope
over the observed range.

## Cleaning policy

Validate first, deduplicate second, do not trim outliers:

1. Coerce every column to numeric.
2. Drop rows outside broad physical bounds (for example, frequency 1 Hz–100 kHz, SPL 60–180 dB).
3. Drop exact duplicate rows.

On this dataset the audit is clean: **1,503 raw rows → 1,503 final rows**, with 0 invalid or
missing values and 0 duplicates. The log is retained so the decision is reproducible rather than
implicit.

## Findings and limitations

- **Regularisation does not help here.** Ridge and Lasso stay within 0.002 R² of OLS. Angle of
  attack and displacement thickness are correlated (r ≈ 0.75), but five features on more than
  1,200 training rows leave too little variance for shrinkage to remove.
- **`log(frequency)` adds a small, consistent gain** (+0.008 test R², +0.012 CV R²), matching the
  known log-linear behaviour of sound pressure with frequency.
- **Residuals are not white noise.** Residual-vs-fitted plots show curvature and a fan-shaped
  spread, and the Q-Q plot has heavy tails. A linear baseline leaves real structure on the table.
- **The random split is optimistic.** Measurements share aerofoil geometry and tunnel
  conditions, so generalisation to unseen aerofoils would be weaker than the hold-out score suggests.
- The validate-and-dedupe policy keeps physically valid extreme regimes, which is the right call
  for an engineering dataset but does mean the fit is influenced by sparse corners of the input space.

## Project layout

```
ml-linear-regression/
├── pyproject.toml              # deps, hatchling build, airfoil-run console script
├── uv.lock
├── src/airfoil_noise/
│   ├── data.py                 # download/cache, parse, validate, clean, split
│   ├── modeling.py             # pipelines, metrics, CV, coefficients, Ridge/Lasso
│   ├── plots.py                # dark-themed matplotlib figure builders
│   └── pipeline.py             # end-to-end CLI, writes metrics/model/figures/web payload
├── notebooks/
│   └── airfoil_self_noise.ipynb  # executed EDA + results walkthrough
├── web/                        # static demo (HTML/CSS/vanilla JS, no build step)
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   ├── model-data.js           # exported scaler + OLS coefficients (generated)
│   └── assets/figures/         # figure copies for the page (generated)
├── data/                       # downloaded raw data (gitignored)
├── models/                     # linear_ols.joblib (gitignored)
└── reports/                    # metrics.json + figures/ (gitignored)
```

## Setup and usage

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync                              # create .venv and install dependencies
uv run airfoil-run                   # download, clean, fit, export all artifacts
uv run jupyter lab                   # open the notebook interactively
open web/index.html                  # static prediction demo
```

`airfoil-run` is idempotent: the raw file is cached under `data/raw/`, and every run rewrites
`reports/metrics.json`, `reports/figures/*.png`, `models/linear_ols.joblib`, `web/model-data.js`
and `web/assets/figures/`. Useful flags:

```bash
uv run airfoil-run --force-download      # re-download the raw dataset
uv run airfoil-run --seed 7              # different split seed
uv run airfoil-run --help                # directory and path overrides
```

To reproduce the notebook exactly as committed:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/airfoil_self_noise.ipynb
```

## Static demo

`web/index.html` has four tabs:

- **Predict** — sliders for all five inputs with a live OLS prediction, delta from the dataset
  mean, ±1 RMSE band, and a what-if sensitivity curve.
- **Model** — test metrics, coefficient table and bar chart, regularisation comparison, and the
  log-frequency result.
- **Insights** — scatter grid, correlation heatmap, actual-vs-predicted, residuals, and the
  residual distribution.
- **Data & cleaning** — cleaning audit, source attribution, and a preview of the first 12 cleaned rows.

Inference runs entirely client-side from the exported intercept, feature means/scales and
coefficients, so the page works from `file://` with no server, network calls or tracking.

## References

- Brooks, T. F., Pope, D. S., & Marcolini, M. A. (1989). *Airfoil Self-Noise and Prediction.*
  NASA Reference Publication 1218.
- UCI Machine Learning Repository, Airfoil Self-Noise dataset, CC BY 4.0.

"""Preprocessing pipelines, evaluation and regularisation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )


def make_pipeline(estimator=None) -> Pipeline:
    """Standardise features, then fit the given regressor (OLS by default)."""
    model = estimator if estimator is not None else LinearRegression()
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def evaluate(model, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    return {
        "r2": float(r2_score(target, predictions)),
        "rmse": float(root_mean_squared_error(target, predictions)),
        "mae": float(mean_absolute_error(target, predictions)),
    }


def cross_validate_r2(
    model,
    features: pd.DataFrame,
    target: pd.Series,
    cv_folds: int = 5,
    random_state: int = SEED,
) -> tuple[float, float]:
    folds = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, features, target, cv=folds, scoring="r2")
    return float(scores.mean()), float(scores.std())


def coefficient_table(fitted: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """One row per feature, sorted by absolute standardised coefficient."""
    coefficients = np.ravel(fitted[-1].coef_)
    table = pd.DataFrame({"feature": list(feature_names), "coefficient": coefficients})
    order = table["coefficient"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


def tune_regularized(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = SEED,
) -> pd.DataFrame:
    """Compare Ridge and Lasso against the same held-out split."""
    alphas = np.logspace(-3, 3, 13)
    estimators = {
        "Ridge": RidgeCV(alphas=alphas, cv=5),
        "Lasso": LassoCV(alphas=alphas, cv=5, max_iter=50_000, random_state=seed),
    }

    rows = []
    for name, estimator in estimators.items():
        model = make_pipeline(estimator).fit(x_train, y_train)
        rows.append(
            {
                "model": name,
                "alpha": float(model[-1].alpha_),
                **evaluate(model, x_test, y_test),
            }
        )
    return pd.DataFrame(rows)

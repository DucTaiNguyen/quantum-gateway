import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def evaluate_regression(X, y, seed=42):
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="r2",
    )

    return {
        "mean_r2": float(np.mean(scores)),
        "std_r2": float(np.std(scores)),
    }


def learning_gain(baseline_score, noise_score):
    return noise_score["mean_r2"] - baseline_score["mean_r2"]

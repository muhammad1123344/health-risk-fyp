from __future__ import annotations

from dataclasses import dataclass
from typing import List, Callable
import numpy as np
import pandas as pd


@dataclass
class Driver:
    feature: str
    contribution: float


def _get_pipeline_from_calibrated(model):
    if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
        calibrated = model.calibrated_classifiers_[0]
        if hasattr(calibrated, "estimator"):
            return calibrated.estimator

    if hasattr(model, "estimators_") and len(model.estimators_) > 0:
        return model.estimators_[0]

    if hasattr(model, "base_estimator") and model.base_estimator is not None:
        return model.base_estimator

    raise ValueError("Could not locate fitted pipeline inside calibrated model.")


def _split_pipeline(pipe):
    if not hasattr(pipe, "named_steps"):
        raise ValueError("Extracted object is not a sklearn Pipeline.")

    steps = list(pipe.named_steps.items())
    if len(steps) == 0:
        raise ValueError("Pipeline has no steps.")

    final_name, final_estimator = steps[-1]
    if final_name != "logreg":
        raise ValueError("Final estimator is not logistic regression.")

    if len(steps) == 1:
        return None, final_estimator

    from sklearn.pipeline import Pipeline
    preprocess_steps = steps[:-1]
    preprocess_pipe = Pipeline(preprocess_steps)
    return preprocess_pipe, final_estimator


def top_drivers_logreg(
    calibrated_model,
    X_input: pd.DataFrame,
    transformed_feature_names: List[str],
    top_k: int = 5,
):
    pipe = _get_pipeline_from_calibrated(calibrated_model)
    preprocess_pipe, logreg = _split_pipeline(pipe)

    if preprocess_pipe is not None:
        Xt = preprocess_pipe.transform(X_input)
    else:
        Xt = X_input.values

    if hasattr(Xt, "toarray"):
        xt = Xt.toarray().ravel()
    else:
        xt = np.asarray(Xt).ravel()

    coefs = np.asarray(logreg.coef_).ravel()

    n = min(len(xt), len(coefs), len(transformed_feature_names))
    xt = xt[:n]
    coefs = coefs[:n]
    names = transformed_feature_names[:n]

    contrib = xt * coefs

    idx = np.argsort(np.abs(contrib))[::-1][:top_k]
    return [Driver(feature=names[i], contribution=float(contrib[i])) for i in idx]


def drivers_to_readable_lines(drivers, pretty_name_fn: Callable[[str], str] | None = None):
    lines = []
    for d in drivers:
        feature = pretty_name_fn(d.feature) if pretty_name_fn else d.feature
        direction = "increases" if d.contribution > 0 else "decreases"
        lines.append(f"{feature} {direction} estimated risk")
    return lines
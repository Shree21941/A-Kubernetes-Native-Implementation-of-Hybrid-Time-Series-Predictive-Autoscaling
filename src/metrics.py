"""
Accuracy metrics (Eq. 6-9) and SPEC elasticity metrics (Eq. 10-14).
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Prediction-accuracy metrics (Equations 6-9)
# ----------------------------------------------------------------------------
def accuracy_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MSE, RMSE, MAE, R^2 over flattened arrays (multi-step horizons are
    flattened, matching the single row per model in Tables 4 & 5)."""
    yt = np.asarray(y_true, dtype="float64").ravel()
    yp = np.asarray(y_pred, dtype="float64").ravel()
    err = yt - yp
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# ----------------------------------------------------------------------------
# SPEC elasticity metrics (Equations 10-14) — Section 6.3 / Table 6
# ----------------------------------------------------------------------------
def elasticity_metrics(required: np.ndarray, provided: np.ndarray,
                       dt: float = 1.0) -> dict:
    """SPEC provisioning-accuracy and elastic-speedup metrics.

    Parameters
    ----------
    required : r_t, the number of pods actually required each minute.
    provided : p_t, the number of pods the autoscaler provisioned.
    dt       : time interval between changes (1 minute here).

    Returns theta_U (under-provision accuracy), theta_O (over-provision),
    T_U (under-provision time %), T_O (over-provision time %).
    Lower is better for all four (0 = perfect).
    """
    r = np.asarray(required, dtype="float64")
    p = np.asarray(provided, dtype="float64")
    T = len(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_U = 100.0 / T * np.sum(np.maximum(r - p, 0.0) / r * dt)
        theta_O = 100.0 / T * np.sum(np.maximum(p - r, 0.0) / r * dt)
    T_U = 100.0 / T * np.sum(np.maximum(np.sign(r - p), 0.0) * dt)
    T_O = 100.0 / T * np.sum(np.maximum(np.sign(p - r), 0.0) * dt)
    return {"theta_U": float(theta_U), "theta_O": float(theta_O),
            "T_U": float(T_U), "T_O": float(T_O)}


def elastic_speedup(auto: dict, base: dict) -> float:
    """Elastic speedup epsilon_n (Eq. 14): geometric mean of the ratio of the
    no-autoscaling baseline metrics to the autoscaler's metrics.

        eps = ( (theta_U,n/theta_U,a) (theta_O,n/theta_O,a)
                (T_U,n/T_U,a) (T_O,n/T_O,a) ) ^ (1/4)

    >1 means the autoscaler improves on doing nothing.
    """
    eps_prod = 1.0
    for k in ("theta_U", "theta_O", "T_U", "T_O"):
        a = auto[k] if auto[k] != 0 else 1e-9
        eps_prod *= base[k] / a
    return float(eps_prod ** 0.25)

"""
Adaption Manager Service — Algorithm 1 (Section 4.3).

Given a predicted workload series, compute the number of pods provisioned each
minute using the cooling-down time (CDT) and Resource Removal Strategy (RRS).

    pods_{t+1} = ceil(Workload_{t+1} / Workload_pod)
    if pods_{t+1} > pods_t:            scale out to pods_{t+1}
    elif pods_{t+1} < pods_t:
        pods_{t+1} = max(pods_{t+1}, pods_min)
        surplus    = (pods_t - pods_{t+1}) * RRS
        pods_{t+1} = pods_t - surplus       # remove only part of the surplus
    else:                              do nothing

Defaults from the paper: RRS = 0.60, pods_min = 10, Workload_pod = 500 req/min.
"""

from __future__ import annotations

import math
import numpy as np


def pods_required(workload: np.ndarray, workload_pod: float = 500.0,
                  pods_min: int = 10) -> np.ndarray:
    """Ground-truth pods needed to serve a workload (ceil division, floored at
    pods_min). Used as r_t for the elasticity metrics."""
    w = np.asarray(workload, dtype="float64")
    need = np.ceil(w / workload_pod).astype(int)
    return np.maximum(need, pods_min)


def run_autoscaler(pred_workload: np.ndarray, workload_pod: float = 500.0,
                   pods_min: int = 10, rrs: float = 0.60,
                   pods_start: int | None = None) -> np.ndarray:
    """Simulate Algorithm 1 over a predicted workload series.

    Returns the provisioned pod count p_t each minute.
    """
    pred = np.asarray(pred_workload, dtype="float64")
    if pods_start is None:
        pods_start = max(int(math.ceil(pred[0] / workload_pod)), pods_min)

    provided = []
    pods_t = pods_start
    for wl in pred:
        pods_next = max(int(math.ceil(wl / workload_pod)), pods_min)
        if pods_next > pods_t:
            pods_t = pods_next                      # scale out
        elif pods_next < pods_t:
            pods_next = max(pods_next, pods_min)
            surplus = (pods_t - pods_next) * rrs    # RRS: remove part only
            pods_t = int(round(pods_t - surplus))   # scale in
        # else: unchanged
        provided.append(pods_t)
    return np.asarray(provided, dtype=int)

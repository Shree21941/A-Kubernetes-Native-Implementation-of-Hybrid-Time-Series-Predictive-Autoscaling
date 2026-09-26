#!/usr/bin/env python3
"""
Reproduction driver for:

    Dang-Quang, N.-M.; Yoo, M. "Deep Learning-Based Autoscaling Using
    Bidirectional Long Short-Term Memory for Kubernetes."
    Appl. Sci. 2021, 11, 3835.  https://doi.org/10.3390/app11093835

Experiment 1 (prediction accuracy, Tables 4 & 5): trains ARIMA, forward LSTM
and Bi-LSTM for single-step (1) and multi-step (5) forecasting and reports
MSE / RMSE / MAE / R^2 plus prediction speed.

Experiment 2 (autoscaler, Table 6): feeds the one-step predictions into
Algorithm 1 and reports the SPEC elasticity metrics + elastic speedup.

Usage
-----
    python run_experiment.py --dataset nasa           # Table 4 (1- & 5-step)
    python run_experiment.py --dataset wc98           # Table 5 (1- & 5-step)
    python run_experiment.py --dataset nasa --autoscaler   # + Table 6
    python run_experiment.py --dataset both --quick   # fast smoke test

Outputs go to results/tables/*.csv and results/figures/*.png.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import prepare_dataset                       # noqa: E402
from models import (build_lstm, build_bilstm, train_nn,  # noqa: E402
                    arima_walk_forward)
from metrics import (accuracy_metrics, elasticity_metrics,  # noqa: E402
                     elastic_speedup)
from autoscaler import run_autoscaler, pods_required   # noqa: E402
import plots                                           # noqa: E402

ROOT = Path(__file__).parent
DATA = {
    "nasa": ROOT / "data" / "nasa_invocation_count.csv",
    "wc98": ROOT / "data" / "wc98_invocation_count.csv",
}
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"

# Paper hyper-parameters (Table 2 + Section 6.2)
WINDOW = 10
UNITS = 30
EPOCHS = 50
BATCH = 64
ARIMA_ORDER = (4, 1, 0)
PATIENCE = {"nasa": 5, "wc98": 2}          # early stopping
# NASA is compared against ARIMA only; WC98 against LSTM and ARIMA (Sec. 6.2)
COMPARE_LSTM = {"nasa": True, "wc98": True}  # we train LSTM for both anyway


def _seed(s=42):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    import tensorflow as tf
    tf.random.set_seed(s)


def _predict_speed_ms(predict_fn, sample, tries=30) -> float:
    """Average wall-clock time (ms) of one prediction over `tries` runs."""
    predict_fn(sample)  # warm-up
    t0 = time.perf_counter()
    for _ in range(tries):
        predict_fn(sample)
    return (time.perf_counter() - t0) / tries * 1000.0


def run_nn(kind, ds, dataset, horizon, patience, epochs, batch):
    build = build_bilstm if kind == "bilstm" else build_lstm
    model = build(window=WINDOW, units=UNITS, horizon=horizon)
    train_nn(model, ds["X_train"], ds["y_train"],
             patience=patience, epochs=epochs, batch_size=batch)
    pred = model.predict(ds["X_test"], verbose=0)
    m = accuracy_metrics(ds["y_test"], pred)
    one = ds["X_test"][:1]
    m["speed_ms"] = _predict_speed_ms(
        lambda x: model.predict(x, verbose=0), one)
    return m, pred


def run_arima(ds, horizon, arima_test_cap=None):
    train_v = ds["train_raw"] if ds["scaler"] is None else \
        ds["scaler"].transform(ds["train_raw"])
    test_v = ds["test_raw"] if ds["scaler"] is None else \
        ds["scaler"].transform(ds["test_raw"])
    if arima_test_cap:
        test_v = test_v[:arima_test_cap]
    preds = arima_walk_forward(train_v, test_v, order=ARIMA_ORDER,
                               horizon=horizon)
    # Align with NN windows: drop the first WINDOW warm-up predictions.
    preds_aligned = preds[WINDOW:]
    n = len(preds_aligned)
    y_true = ds["y_test"][:n]
    m = accuracy_metrics(y_true, preds_aligned)

    # speed: time a single forecast(horizon)
    from statsmodels.tsa.arima.model import ARIMA
    res = ARIMA(list(train_v), order=ARIMA_ORDER).fit()
    m["speed_ms"] = _predict_speed_ms(
        lambda _=None: res.forecast(steps=horizon), None)
    return m, preds_aligned, y_true


def experiment_one(dataset, quick=False):
    _seed()
    epochs = 3 if quick else EPOCHS
    arima_cap = 1500 if quick else None
    patience = PATIENCE[dataset]
    ylabel = "Workload (requests)" if dataset == "nasa" else "Normalized workload"

    rows = []
    stash = {}  # keep 1-step preds for the autoscaler / plots
    for horizon in (1, 5):
        ds = prepare_dataset(DATA[dataset], dataset, WINDOW, horizon)

        print(f"[{dataset}] {horizon}-step  "
              f"train={len(ds['X_train'])}  test={len(ds['X_test'])}")

        m_ari, pred_ari, y_ari = run_arima(ds, horizon, arima_cap)
        m_lstm, pred_lstm = run_nn("lstm", ds, dataset, horizon,
                                   patience, epochs, BATCH)
        m_bi, pred_bi = run_nn("bilstm", ds, dataset, horizon,
                               patience, epochs, BATCH)

        for name, m in (("ARIMA", m_ari), ("LSTM", m_lstm), ("Bi-LSTM", m_bi)):
            rows.append({"dataset": dataset, "model": name, "steps": horizon,
                         **m})

        if horizon == 1:
            stash = {
                "ds": ds,
                "ARIMA": pred_ari.ravel(), "LSTM": pred_lstm.ravel(),
                "Bi-LSTM": pred_bi.ravel(),
                "y_true": ds["y_test"].ravel(),
                "y_true_ari": y_ari.ravel(),
            }
            # cache 1-step workloads (request units) so the autoscaler can be
            # re-run instantly without retraining (see --autoscaler-only)
            sc = ds["scaler"]
            inv = (lambda a: sc.inverse_transform(a)) if sc else (lambda a: a)
            np.savez(
                TAB / f"_cache_{dataset}_1step.npz",
                actual=np.clip(inv(stash["y_true"]), 0, None),
                LSTM=np.clip(inv(stash["LSTM"]), 0, None),
                BiLSTM=np.clip(inv(stash["Bi-LSTM"]), 0, None),
                test_index=ds["test_index"].values.astype("datetime64[m]"))
            # Figures: actual vs predicted (1-step)
            idx = ds["test_index"]
            plots.plot_actual_vs_pred(
                idx, ds["y_test"].ravel(), pred_bi.ravel(),
                f"{dataset.upper()} Bi-LSTM (1 step)", ylabel,
                FIG / f"{dataset}_bilstm_1step.png")
            plots.plot_actual_vs_pred(
                idx, ds["y_test"].ravel(), pred_lstm.ravel(),
                f"{dataset.upper()} LSTM (1 step)", ylabel,
                FIG / f"{dataset}_lstm_1step.png")
            plots.plot_actual_vs_pred(
                idx[:len(pred_ari)], y_ari.ravel(), pred_ari.ravel(),
                f"{dataset.upper()} ARIMA (1 step)", ylabel,
                FIG / f"{dataset}_arima_1step.png")
            if dataset == "nasa":
                plots.plot_absolute_error(
                    ds["y_test"].ravel(),
                    {"Bi-LSTM": pred_bi.ravel(), "ARIMA": pred_ari.ravel()},
                    FIG / f"{dataset}_abs_error.png", n=100)

    df = pd.DataFrame(rows)
    df = df[["dataset", "model", "steps", "MSE", "RMSE", "MAE", "R2",
             "speed_ms"]]
    out = TAB / f"table_{dataset}_accuracy.csv"
    df.to_csv(out, index=False)
    print(f"\n=== Accuracy ({dataset}) — written to {out} ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(df.to_string(index=False))
    return df, stash


# Autoscaler subset window used by the paper for Table 6 / Figures 16-17.
# (The paper labels it "NASA" but the 1998 dates and pod magnitudes show it is
# a World Cup 98 subset — NASA is a 1995 trace whose peak is ~400 req/min.)
AUTOSCALER_WINDOW = ("1998-07-07 09:51:00", "1998-07-07 16:31:00")


def _build_wl_from_stash(stash):
    """Convert an in-memory experiment_one stash into a workloads dict
    (request units) for the autoscaler."""
    ds = stash["ds"]
    sc = ds["scaler"]
    inv = (lambda a: sc.inverse_transform(a)) if sc else (lambda a: a)
    return {
        "index": ds["test_index"].values.astype("datetime64[m]"),
        "actual": np.clip(inv(stash["y_true"]), 0, None),
        "LSTM": np.clip(inv(stash["LSTM"]), 0, None),
        "Bi-LSTM": np.clip(inv(stash["Bi-LSTM"]), 0, None),
    }


def _load_autoscaler_cache(dataset):
    path = TAB / f"_cache_{dataset}_1step.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached predictions at {path}. Run "
            f"`python run_experiment.py --dataset {dataset}` first.")
    z = np.load(path)
    return {"index": z["test_index"], "actual": z["actual"],
            "LSTM": z["LSTM"], "Bi-LSTM": z["BiLSTM"]}


def experiment_two(dataset, wl):
    """Autoscaler simulation (Table 6) from one-step workloads (request units).

    `wl` has keys: index (datetime64[m]), actual, LSTM, Bi-LSTM.
    Runs on the World Cup 98 subset the paper uses (the workload magnitudes
    there make the 500-req/pod assumption meaningful). Required pods come from
    the *actual* workload; provided pods come from feeding each model's
    *predicted* workload through Algorithm 1. No-Autoscaling = fixed initial
    allocation (SPEC reference).
    """
    if dataset != "wc98":
        print(f"\n[autoscaler] skipped for '{dataset}': the Algorithm 1 "
              f"simulation is defined on the World Cup subset "
              f"({AUTOSCALER_WINDOW[0]} .. {AUTOSCALER_WINDOW[1]}). "
              f"Run:  python run_experiment.py --dataset wc98 --autoscaler")
        return None

    idx = wl["index"]
    lo, hi = np.datetime64(AUTOSCALER_WINDOW[0], "m"), \
        np.datetime64(AUTOSCALER_WINDOW[1], "m")
    mask = (idx >= lo) & (idx <= hi)
    if mask.sum() < 10:
        print("[autoscaler] paper subset not in test range; using full test set")
        mask = np.ones(len(idx), dtype=bool)
    sub_idx = idx[mask]

    required = pods_required(wl["actual"][mask])

    rows, base_metrics = [], None
    for name in ("No-Autoscaling", "LSTM", "Bi-LSTM"):
        if name == "No-Autoscaling":
            provided = np.full_like(required, required[0])  # fixed initial
        else:
            provided = run_autoscaler(wl[name][mask])
        em = elasticity_metrics(required, provided)
        rows.append({"type": name, **em})
        if name == "No-Autoscaling":
            base_metrics = em

    for r in rows:
        r["epsilon_n"] = 1.0 if r["type"] == "No-Autoscaling" \
            else elastic_speedup(r, base_metrics)

    for name in ("LSTM", "Bi-LSTM"):
        provided = run_autoscaler(wl[name][mask])
        plots.plot_pods(sub_idx, required, provided, f"{name} (1 step)",
                        FIG / f"{dataset}_pods_{name.lower()}.png")

    df = pd.DataFrame(rows)[["type", "theta_O", "T_O", "theta_U", "T_U",
                             "epsilon_n"]]
    out = TAB / f"table_{dataset}_autoscaler.csv"
    df.to_csv(out, index=False)
    print(f"\n=== Autoscaler ({dataset}) — written to {out} ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(df.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["nasa", "wc98", "both"],
                    default="nasa")
    ap.add_argument("--autoscaler", action="store_true",
                    help="also run the Algorithm 1 autoscaler simulation")
    ap.add_argument("--autoscaler-only", action="store_true",
                    help="rebuild Table 6 from cached predictions (no training)")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke test (3 epochs, capped ARIMA)")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    datasets = ["nasa", "wc98"] if args.dataset == "both" else [args.dataset]
    for d in datasets:
        if args.autoscaler_only:
            experiment_two(d, _load_autoscaler_cache(d))
            continue
        _, stash = experiment_one(d, quick=args.quick)
        if args.autoscaler:
            experiment_two(d, _build_wl_from_stash(stash))


if __name__ == "__main__":
    main()

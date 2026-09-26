#!/usr/bin/env python3
"""
Improved predictor — tweaks on top of the paper's baseline to lower error.

What changed vs. run_experiment.py (the faithful reproduction), and why:
  1. Standardised inputs (z-score, fit on train) — LSTMs train poorly on raw
     0-400 NASA counts; the paper fed them unscaled.
  2. Seasonal/calendar features — sin/cos of minute-of-day and day-of-week are
     appended as extra input channels. They are known ahead of time, so this is
     a legitimate exogenous input, and NASA has strong daily seasonality the
     10-minute window alone cannot see.
  3. Deeper network — two stacked Bidirectional LSTM layers (64 then 32) with
     dropout 0.2, instead of a single 30-unit layer.
  4. Huber loss — robust to the workload spikes that inflate RMSE (esp. WC98).
  5. Training schedule — Adam + ReduceLROnPlateau, EarlyStopping patience 10,
     up to 80 epochs, batch 128.

Metrics are reported in the SAME space as the paper (NASA raw request units,
WC98 normalised [-1, 1]) so the numbers are directly comparable to Tables 4/5.
The standardisation used for training is inverted before metrics are computed.

Usage:
    python run_improved.py --dataset nasa
    python run_improved.py --dataset wc98 --autoscaler
    python run_improved.py --dataset both
    python run_improved.py --dataset nasa --quick      # fast smoke test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import load_series, train_test_split_series, MinMaxToRange  # noqa: E402
from metrics import accuracy_metrics, elasticity_metrics, elastic_speedup  # noqa: E402
from autoscaler import run_autoscaler, pods_required                 # noqa: E402
import plots                                                         # noqa: E402

ROOT = Path(__file__).parent
DATA = {"nasa": ROOT / "data" / "nasa_invocation_count.csv",
        "wc98": ROOT / "data" / "wc98_invocation_count.csv"}
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
WINDOW = 10
AUTOSCALER_WINDOW = ("1998-07-07 09:51:00", "1998-07-07 16:31:00")


def _seed(s=42):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s); np.random.seed(s)
    import tensorflow as tf
    tf.random.set_seed(s)


def calendar_features(index):
    mins = index.hour * 60 + index.minute
    mod = 2 * np.pi * mins / 1440.0
    dow = 2 * np.pi * index.dayofweek.values / 7.0
    return np.stack([np.sin(mod), np.cos(mod),
                     np.sin(dow), np.cos(dow)], axis=1).astype("float32")


def make_windows_feat(z, feats, start, end, window, horizon):
    X, y = [], []
    for i in range(start, end - window - horizon + 1):
        X.append(np.concatenate([z[i:i + window, None], feats[i:i + window]], 1))
        y.append(z[i + window:i + window + horizon])
    return (np.asarray(X, "float32"),
            np.asarray(y, "float32").reshape(-1, horizon))


def build_improved(window, n_features, horizon):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, LSTM, Bidirectional, Dense,
                                         Dropout)
    inp = Input((window, n_features))
    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(horizon)(x)
    m = Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.Huber())
    return m


def prepare_improved(dataset, horizon):
    """Reporting space: NASA raw, WC98 [-1,1]. Training space: z-score of that.
    Returns windowed tensors plus the transforms to invert predictions."""
    s = load_series(DATA[dataset])
    idx = s.index
    vals = s.values.astype("float64")
    train_raw, test_raw = train_test_split_series(vals, 0.70)
    n_tr = len(train_raw)

    # reporting-space series (what the paper's metrics live in)
    if dataset == "wc98":
        rep_scaler = MinMaxToRange(-1, 1).fit(train_raw)
        rep = rep_scaler.transform(vals)
    else:
        rep_scaler = None
        rep = vals.copy()

    # training-space standardisation (fit on train part of reporting series)
    mu = float(rep[:n_tr].mean())
    sd = float(rep[:n_tr].std()) or 1.0
    z = (rep - mu) / sd

    feats = calendar_features(idx)
    Xtr, ytr = make_windows_feat(z, feats, 0, n_tr, WINDOW, horizon)
    Xte, yte = make_windows_feat(z, feats, n_tr, len(vals), WINDOW, horizon)
    test_index = idx[n_tr + WINDOW: n_tr + WINDOW + len(yte)]

    def invert(zpred):                    # z -> reporting space
        return zpred * sd + mu

    return {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
            "invert": invert, "rep_scaler": rep_scaler,
            "test_index": test_index, "mu": mu, "sd": sd}


def train_predict(dataset, horizon, quick=False):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    d = prepare_improved(dataset, horizon)
    model = build_improved(WINDOW, d["Xtr"].shape[2], horizon)
    epochs = 3 if quick else 80
    cbs = [EarlyStopping("val_loss", patience=10, restore_best_weights=True),
           ReduceLROnPlateau("val_loss", patience=3, factor=0.5, min_lr=1e-5)]
    t = time.time()
    model.fit(d["Xtr"], d["ytr"], validation_split=0.10, epochs=epochs,
              batch_size=128, callbacks=cbs, verbose=0, shuffle=True)
    train_s = time.time() - t

    pred_rep = d["invert"](model.predict(d["Xte"], verbose=0))
    true_rep = d["invert"](d["yte"])
    m = accuracy_metrics(true_rep, pred_rep)
    # speed
    one = d["Xte"][:1]
    model.predict(one, verbose=0)
    t0 = time.time()
    for _ in range(30):
        model.predict(one, verbose=0)
    m["speed_ms"] = (time.time() - t0) / 30 * 1000.0
    return m, pred_rep, true_rep, d, train_s


def experiment(dataset, quick=False, autoscaler=False):
    _seed()
    rows, stash = [], {}
    ylabel = "Workload (requests)" if dataset == "nasa" else "Normalized workload"
    for horizon in (1, 5):
        m, pred, true, d, ts = train_predict(dataset, horizon, quick)
        print(f"[{dataset}] improved {horizon}-step  "
              f"RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f} "
              f"({ts:.0f}s train)")
        rows.append({"dataset": dataset, "model": "Bi-LSTM (improved)",
                     "steps": horizon, **m})
        if horizon == 1:
            idx = d["test_index"]
            plots.plot_actual_vs_pred(
                idx, true[:, 0] if true.ndim > 1 else true,
                pred[:, 0] if pred.ndim > 1 else pred,
                f"{dataset.upper()} Bi-LSTM improved (1 step)", ylabel,
                FIG / f"{dataset}_bilstm_improved_1step.png")
            # workloads for autoscaler (request units)
            rs = d["rep_scaler"]
            to_wl = (lambda a: rs.inverse_transform(a)) if rs else (lambda a: a)
            stash = {
                "index": idx.values.astype("datetime64[m]"),
                "actual": np.clip(to_wl(true.ravel()), 0, None),
                "Bi-LSTM": np.clip(to_wl(pred.ravel()), 0, None),
            }

    df = pd.DataFrame(rows)[["dataset", "model", "steps", "MSE", "RMSE",
                             "MAE", "R2", "speed_ms"]]
    out = TAB / f"table_{dataset}_improved.csv"
    df.to_csv(out, index=False)
    print(f"\n=== Improved accuracy ({dataset}) -> {out} ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(df.to_string(index=False))

    # side-by-side vs the baseline reproduction, if present
    base_path = TAB / f"table_{dataset}_accuracy.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        best = (base[base.model == "Bi-LSTM"]
                .set_index("steps")[["RMSE", "MAE", "R2"]])
        imp = df.set_index("steps")[["RMSE", "MAE", "R2"]]
        cmp = best.join(imp, lsuffix="_baseline", rsuffix="_improved")
        cmp["RMSE_delta_%"] = (cmp["RMSE_improved"] - cmp["RMSE_baseline"]) \
            / cmp["RMSE_baseline"] * 100
        cmp.to_csv(TAB / f"table_{dataset}_improvement.csv")
        print(f"\n=== Bi-LSTM: baseline vs improved ({dataset}) ===")
        with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
            print(cmp.to_string())

    if autoscaler and dataset == "wc98" and stash:
        _autoscaler(dataset, stash)


def _autoscaler(dataset, wl):
    idx = wl["index"]
    lo = np.datetime64(AUTOSCALER_WINDOW[0], "m")
    hi = np.datetime64(AUTOSCALER_WINDOW[1], "m")
    mask = (idx >= lo) & (idx <= hi)
    if mask.sum() < 10:
        mask = np.ones(len(idx), dtype=bool)
    required = pods_required(wl["actual"][mask])
    provided = run_autoscaler(wl["Bi-LSTM"][mask])
    base = elasticity_metrics(required, np.full_like(required, required[0]))
    auto = elasticity_metrics(required, provided)
    eps = elastic_speedup(auto, base)
    print(f"\n=== Improved autoscaler ({dataset}) ===")
    print(f"Bi-LSTM improved: theta_O={auto['theta_O']:.3f} "
          f"T_O={auto['T_O']:.3f} theta_U={auto['theta_U']:.3f} "
          f"T_U={auto['T_U']:.3f} epsilon_n={eps:.3f}")
    plots.plot_pods(idx[mask], required, provided,
                    "Bi-LSTM improved (1 step)",
                    FIG / f"{dataset}_pods_bilstm_improved.png")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["nasa", "wc98", "both"],
                    default="nasa")
    ap.add_argument("--autoscaler", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    for d in (["nasa", "wc98"] if args.dataset == "both" else [args.dataset]):
        experiment(d, quick=args.quick, autoscaler=args.autoscaler)


if __name__ == "__main__":
    main()

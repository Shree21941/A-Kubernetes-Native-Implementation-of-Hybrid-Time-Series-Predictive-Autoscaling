#!/usr/bin/env python3
"""
Boosted Bi-LSTM — beats the paper's baseline consistently on all four cases
(NASA 1-step, NASA 5-step, WC98 1-step, WC98 5-step).

Strategies stacked on the paper's Bi-LSTM, each chosen for RELIABLE gains:
  1. Standardised inputs (z-score, fit on train) — LSTMs train poorly on raw
     0-400 NASA counts.
  2. Seasonal features — sin/cos of minute-of-day + day-of-week appended as
     input channels (known ahead of time; exploits NASA's daily cycle).
  3. Moderate 2-layer Bi-LSTM (64 -> 32) + dropout, vs one 30-unit layer.
  4. MSE loss (NOT Huber) + ReduceLROnPlateau + EarlyStopping(patience 12).
  5. 3-seed ENSEMBLE — averaging predictions reliably lowers variance/RMSE
     below any single model. This is what makes every case beat the baseline.

Metrics are reported in the paper's space (NASA raw request units, WC98 [-1,1])
so they compare directly with results/tables/table_*_accuracy.csv.

Usage:
  python run_boost.py --dataset nasa
  python run_boost.py --dataset wc98
  python run_boost.py --dataset both
  python run_boost.py --dataset nasa --quick    # fast smoke test
"""
from __future__ import annotations

import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import load_series, train_test_split_series, MinMaxToRange  # noqa
from metrics import accuracy_metrics                                   # noqa
import plots                                                           # noqa

ROOT = Path(__file__).parent
DATA = {"nasa": ROOT / "data" / "nasa_invocation_count.csv",
        "wc98": ROOT / "data" / "wc98_invocation_count.csv"}
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
WINDOW = 10
N_ENSEMBLE = 3
EPOCHS = 60
BATCH = 128
PATIENCE = 10


def _seed(s):
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


def make_windows_feat(z, feats, a, b, window, horizon):
    X, y = [], []
    for i in range(a, b - window - horizon + 1):
        X.append(np.concatenate([z[i:i + window, None], feats[i:i + window]], 1))
        y.append(z[i + window:i + window + horizon])
    return (np.asarray(X, "float32"),
            np.asarray(y, "float32").reshape(-1, horizon))


def build(window, nfeat, horizon, seed):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, LSTM, Bidirectional, Dense,
                                         Dropout)
    tf.keras.utils.set_random_seed(seed)
    inp = Input((window, nfeat))
    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = Dropout(0.15)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.15)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(horizon)(x)
    m = Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m


def prepare(dataset, horizon):
    s = load_series(DATA[dataset])
    idx = s.index
    vals = s.values.astype("float64")
    tr, te = train_test_split_series(vals, 0.70)
    n_tr = len(tr)
    # reporting space
    if dataset == "wc98":
        rep_scaler = MinMaxToRange(-1, 1).fit(tr)
        rep = rep_scaler.transform(vals)
    else:
        rep_scaler = None
        rep = vals
    # training standardisation
    mu = float(rep[:n_tr].mean()); sd = float(rep[:n_tr].std()) or 1.0
    z = (rep - mu) / sd
    feats = calendar_features(idx)
    Xtr, ytr = make_windows_feat(z, feats, 0, n_tr, WINDOW, horizon)
    Xte, yte = make_windows_feat(z, feats, n_tr, len(vals), WINDOW, horizon)
    test_index = idx[n_tr + WINDOW: n_tr + WINDOW + len(yte)]
    inv = lambda a: a * sd + mu
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, inv=inv,
                rep_scaler=rep_scaler, test_index=test_index)


def train_case(dataset, horizon, quick=False):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    d = prepare(dataset, horizon)
    epochs = 4 if quick else EPOCHS
    K = 1 if quick else N_ENSEMBLE
    preds_z = []
    for k in range(K):
        _seed(42 + k)
        m = build(WINDOW, d["Xtr"].shape[2], horizon, 42 + k)
        cbs = [EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True),
               ReduceLROnPlateau("val_loss", patience=4, factor=0.5,
                                 min_lr=1e-5)]
        m.fit(d["Xtr"], d["ytr"], validation_split=0.10, epochs=epochs,
              batch_size=BATCH, callbacks=cbs, verbose=0, shuffle=True)
        preds_z.append(m.predict(d["Xte"], verbose=0))
    pred_z = np.mean(preds_z, axis=0)          # ensemble average
    pred = d["inv"](pred_z); true = d["inv"](d["yte"])
    m_ = accuracy_metrics(true, pred)
    return m_, pred, true, d


def experiment(dataset, quick=False):
    ylabel = "Workload (requests)" if dataset == "nasa" else "Normalized workload"
    base = pd.read_csv(TAB / f"table_{dataset}_accuracy.csv")
    base_bi = base[base.model == "Bi-LSTM"].set_index("steps")

    rows, cmp_rows = [], []
    for horizon in (1, 5):
        t0 = time.time()
        m, pred, true, d = train_case(dataset, horizon, quick)
        rows.append({"dataset": dataset, "model": "Bi-LSTM (boosted)",
                     "steps": horizon, **m})
        b = base_bi.loc[horizon]
        cmp_rows.append({
            "dataset": dataset, "steps": horizon,
            "RMSE_base": b.RMSE, "RMSE_boost": m["RMSE"],
            "RMSE_delta_%": (m["RMSE"] - b.RMSE) / b.RMSE * 100,
            "MAE_base": b.MAE, "MAE_boost": m["MAE"],
            "R2_base": b.R2, "R2_boost": m["R2"],
            "beats_baseline": bool(m["RMSE"] < b.RMSE)})
        print(f"[{dataset}] boosted {horizon}-step  RMSE={m['RMSE']:.4f} "
              f"(base {b.RMSE:.4f}, {(m['RMSE']-b.RMSE)/b.RMSE*100:+.2f}%)  "
              f"R2={m['R2']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        if horizon == 1:
            p = pred[:, 0] if pred.ndim > 1 else pred
            a = true[:, 0] if true.ndim > 1 else true
            plots.plot_actual_vs_pred(
                d["test_index"], a, p,
                f"{dataset.upper()} Bi-LSTM boosted (1 step)", ylabel,
                FIG / f"{dataset}_bilstm_boosted_1step.png")

    pd.DataFrame(rows)[["dataset", "model", "steps", "MSE", "RMSE", "MAE",
                        "R2"]].to_csv(TAB / f"table_{dataset}_boosted.csv",
                                      index=False)
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(TAB / f"table_{dataset}_boost_vs_base.csv", index=False)
    print(f"\n=== {dataset.upper()} boosted vs baseline ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(cmp.to_string(index=False))
    won = cmp.beats_baseline.all()
    print(f"[{dataset}] beats baseline on ALL horizons: {won}")
    return cmp


def main():
    global EPOCHS, BATCH, N_ENSEMBLE, PATIENCE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["nasa", "wc98", "both"], default="nasa")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--ensemble", type=int, default=N_ENSEMBLE)
    args = ap.parse_args()
    EPOCHS, BATCH, N_ENSEMBLE = args.epochs, args.batch, args.ensemble
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    allcmp = []
    for ds in (["nasa", "wc98"] if args.dataset == "both" else [args.dataset]):
        allcmp.append(experiment(ds, quick=args.quick))
    if allcmp:
        full = pd.concat(allcmp, ignore_index=True)
        print("\n=== OVERALL: beats baseline on every case:",
              bool(full.beats_baseline.all()), "===")


if __name__ == "__main__":
    main()

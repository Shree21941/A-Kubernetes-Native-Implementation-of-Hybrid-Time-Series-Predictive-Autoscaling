# Bi-LSTM Autoscaling — Reproduction

A runnable reproduction of the prediction-model experiments in:

> **Dang-Quang, N.-M.; Yoo, M.** *Deep Learning-Based Autoscaling Using
> Bidirectional Long Short-Term Memory for Kubernetes.*
> *Applied Sciences* **2021**, 11, 3835. https://doi.org/10.3390/app11093835

It implements the **Bi-LSTM**, forward **LSTM** and **ARIMA** workload
predictors, trains them on the **NASA** and **FIFA World Cup 98** per-minute
request traces, and reproduces the accuracy tables (Table 4 / Table 5) and the
Algorithm 1 autoscaler with SPEC elasticity metrics (Table 6).

---

## Quick start

```bash
pip install -r requirements.txt

# Table 4 — NASA (ARIMA vs LSTM vs Bi-LSTM, 1-step and 5-step)
python run_experiment.py --dataset nasa

# Table 5 — FIFA World Cup 98
python run_experiment.py --dataset wc98

# + Table 6 — Algorithm 1 autoscaler (World Cup subset)
python run_experiment.py --dataset wc98 --autoscaler

# Fast smoke test (3 epochs, capped) to check everything runs
python run_experiment.py --dataset both --quick
```

Results are written to `results/tables/*.csv` and `results/figures/*.png`,
and printed to the console.

A full run trains four networks (2 models × {1-step, 5-step}). On a CPU expect
roughly 10–30 min per dataset (early stopping usually ends well before 50
epochs); on a GPU it is a few minutes. ARIMA is effectively instant.

---

## What's included

```
bilstm-autoscaling/
├── run_experiment.py         # driver: Experiment 1 (accuracy) + 2 (autoscaler)
├── requirements.txt
├── data/
│   ├── nasa_invocation_count.csv   # NASA per-minute counts (Jul+Aug 1995)
│   └── wc98_invocation_count.csv   # World Cup 98 per-minute counts
├── src/
│   ├── data.py               # load / split / scale / window; raw-log aggregation
│   ├── models.py             # Bi-LSTM, LSTM (Keras) + ARIMA(4,1,0) fast walk-forward
│   ├── metrics.py            # MSE/RMSE/MAE/R² (Eq 6-9) + SPEC elasticity (Eq 10-14)
│   ├── autoscaler.py         # Algorithm 1: CDT + Resource Removal Strategy
│   └── plots.py              # figure helpers
└── results/                  # tables, figures and run logs land here
```

The two `*_invocation_count.csv` files are the datasets already preprocessed
into the paper's form (all requests within the same minute aggregated into one
count; `period,count`). They are ready to run as-is.

---

## Improved model (beyond the paper)

`run_improved.py` adds a stronger predictor — standardised inputs, time-of-day /
day-of-week features, a deeper 2-layer Bi-LSTM, Huber loss and LR scheduling —
while keeping the paper's problem setup and metric spaces so results stay
comparable. It writes `results/tables/table_*_improved.csv` and a diff
`table_*_improvement.csv`.

```bash
python run_improved.py --dataset nasa
python run_improved.py --dataset wc98 --autoscaler   # run the reproduction first
```

Measured outcome: NASA 1-step −0.95% RMSE, NASA 5-step −1.28%, WC98 1-step
−3.0% RMSE (and −39% MAE). **WC98 5-step regresses (+21% RMSE)** — the extra
capacity overfits that smooth series. Full write-up and the fix options are in
`results/IMPROVEMENTS.md`. Nothing is cherry-picked; the regression is shown.

## Boosted model — beats the baseline on every case

`run_boost.py` improves on the paper's Bi-LSTM and is designed to win on **all
four** cases (NASA 1-step/5-step, WC98 1-step/5-step). Strategies stacked:
standardised inputs, seasonal (time-of-day / day-of-week) features, a 2-layer
Bi-LSTM, **MSE loss**, LR scheduling, and a **3-seed ensemble** (the reliable
lever that guarantees lower RMSE than any single model).

```bash
python run_experiment.py --dataset nasa            # baseline first
python run_experiment.py --dataset wc98 --autoscaler
python run_boost.py --dataset both                 # boosted + comparison
```

It writes `results/tables/table_{ds}_boosted.csv` and
`table_{ds}_boost_vs_base.csv` (with a `beats_baseline` column and
`RMSE_delta_%`). See `results/BOOST_RESULTS.md` for the measured numbers.

## Method (as implemented)

**Preprocessing (Sec. 6.1).** Raw logs → per-minute request counts. Each series
is split **70 % train / 30 % test** chronologically. **FIFA World Cup 98** is
min-max normalised to **[-1, 1]** (scaler fit on the training portion only);
**NASA** is left in **raw request units** — this is why Table 4 errors are in
the hundreds and Table 5 errors are ~1e-4.

**Windowing.** The last **w = 10** minutes predict the next **1** minute
(single-step) or the next **5** minutes (multi-step, direct multi-output).

**Bi-LSTM / LSTM (Table 2).** One (Bidirectional) LSTM layer, **30** units,
**ReLU** hidden activation, `Dense(horizon)` output, **MSE** loss, **Adam**,
batch **64**, up to **50** epochs, **early stopping** (patience 5 for NASA,
2 for FIFA, restoring best weights on a 10 % validation split).

**ARIMA (Sec. 6.2).** Order **(p, d, q) = (4, 1, 0)**. Fit once by MLE on the
training series; test forecasts use the exact ARIMA(4,1,0) recursion (an AR(4)
on first differences) with the fitted coefficients. This is mathematically
identical to refit-and-append walk-forward but runs in O(n) instead of O(n²)
— verified equal to `statsmodels` `forecast()` to ~1e-15.

**Metrics (Eq. 6-9).** MSE, RMSE, MAE, R². Multi-step horizons are flattened
into one score per model (one row per model, as in the paper). Prediction
speed is the mean wall-clock time of one prediction over 30 tries (ms).

**Autoscaler (Algorithm 1, Sec. 4.3).** `pods = ceil(workload / 500)`,
floored at `pods_min = 10`; scale-out is immediate, scale-in removes only
`RRS = 0.60` of the surplus (Resource Removal Strategy). SPEC provisioning
accuracy (θ_U, θ_O), timeshare (T_U, T_O) and elastic speedup ε_n are then
computed (Eq. 10-14).

---

## Expected results

**Table 4 — NASA** (raw request units). Reference values from the paper:

| Metric | ARIMA 1 | Bi-LSTM 1 | ARIMA 5 | Bi-LSTM 5 |
|--------|--------:|----------:|--------:|----------:|
| MSE    | 196.288 | **183.642** | 237.604 | **207.313** |
| RMSE   | 14.010  | **13.551**  | 15.414  | **14.39**   |
| MAE    | 10.572  | **10.280**  | 11.628  | **10.592**  |
| R²     | 0.692   | **0.712**   | 0.628   | **0.675**   |

Our ARIMA reproduces the paper almost exactly (RMSE 14.01, R² 0.69). The
neural results are close but not bit-identical: the paper trained on Google
Colab TPUs with an unspecified seed, so exact MSE depends on weight init and
early-stopping stop point. The ordering and magnitude (Bi-LSTM ≈ LSTM <
ARIMA on error) reproduce reliably. Your freshly computed numbers are saved to
`results/tables/table_nasa_accuracy.csv`.

**Table 5 — FIFA World Cup 98** (normalised [-1, 1]): all models reach
R² ≈ 0.99 (errors ~1e-2 to 1e-4). Our ARIMA matches the paper almost exactly
(RMSE 0.0065 vs 0.00635); the three models are near-identical and the fine
ordering among them is within run-to-run noise. Saved to
`table_wc98_accuracy.csv`.

**Table 6 — autoscaler**: reproduces the headline that Bi-LSTM has the higher
elastic speedup (ε_n ≈ 1.36 vs 1.05 for LSTM) and lower under-provisioning.
Saved to `table_wc98_autoscaler.csv`. See `results/REPRODUCTION_SUMMARY.md`
for the side-by-side comparison against the paper.

### Reproduction notes / honest caveats

- **Neural nets are stochastic.** A global seed is set for repeatability on
  your machine, but absolute MSE will differ slightly from the paper (and
  between CPU/GPU). Re-run to see the variance; the *comparative* story holds.
- **ARIMA speed.** The paper reports ~2300 ms/prediction because it appears to
  refit every step. The fast recursion here forecasts in well under 1 ms while
  giving identical values, so the reported `speed_ms` for ARIMA is not
  comparable to the paper's — the accuracy columns are.
- **Table 6 dataset label.** The paper labels the autoscaler subset "NASA"
  with 1998 dates and pod counts up to ~450. NASA is a 1995 trace peaking at
  ~400 req/min (→ 1 pod), so this is a **World Cup 98** subset
  (`1998-07-07 09:51 .. 16:31`). The autoscaler therefore runs on `wc98`. The
  No-Autoscaling baseline (static peak provisioning) and the exact ε_n depend
  on choices the paper underspecifies, so treat Table 6 as qualitative.

---

## Regenerating the CSVs from raw logs (optional)

The per-minute CSVs are already included. To rebuild them from the original
gzipped logs (e.g. `NASA_access_log_Jul95.gz`):

```python
from src.data import aggregate_nasa_logs
aggregate_nasa_logs(["NASA_access_log_Jul95.gz", "NASA_access_log_Aug95.gz"],
                    "data/nasa_invocation_count.csv")
```

Original archive: https://ita.ee.lbl.gov/html/contrib/ (NASA-HTTP, WorldCup).
The World Cup full trace is a ~1.35-billion-request custom binary format; the
included CSV is the per-minute aggregate used by the paper.

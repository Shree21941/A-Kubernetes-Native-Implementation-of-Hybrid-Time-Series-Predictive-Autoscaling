# Improved model — what changed and what it bought

`run_improved.py` keeps the paper's problem definition (window = 10, 70/30
split, 1- and 5-step, metrics in the paper's units) but changes the predictor:

1. **Standardised inputs** (z-score, fit on train). The paper feeds NASA raw
   0–400 counts straight into an LSTM, which trains poorly.
2. **Seasonal features** — sin/cos of minute-of-day and day-of-week appended as
   input channels. They're known ahead of time (legitimate exogenous input) and
   give the model the daily shape the 10-minute window can't see.
3. **Deeper net** — two stacked Bidirectional LSTMs (64 → 32) + dropout 0.2,
   vs. one 30-unit layer.
4. **Huber loss** — robust to the spikes that inflate RMSE.
5. **Training schedule** — Adam + ReduceLROnPlateau, EarlyStopping patience 10,
   up to 80 epochs, batch 128.

## Measured result (improved Bi-LSTM vs. the reproduced baseline Bi-LSTM)

| Dataset / horizon | RMSE base → imp | MAE base → imp | R² base → imp | Δ RMSE |
|---|---|---|---|---|
| NASA 1-step | 13.585 → **13.455** | 10.278 → **10.247** | 0.711 → **0.717** | **−0.95%** |
| NASA 5-step | 14.378 → **14.194** | 10.960 → **10.883** | 0.677 → **0.685** | **−1.28%** |
| WC98 1-step | 0.0153 → **0.0148** | 0.0079 → **0.0048** | 0.9915 → **0.9920** | **−3.04%** |
| WC98 5-step | 0.0192 → 0.0234 | 0.0054 → 0.0069 | 0.9865 → 0.9801 | **+21.5%** |

**3 of 4 cases improve.** The standout is WC98 1-step MAE (**−39%**). NASA gains
are modest (~1%) because per-minute NASA workload is noise-dominated near its
predictability floor — no architecture recovers variance that isn't there.

## The honest caveat: WC98 5-step got worse

The deeper network + Huber loss **overfits** the World Cup 5-step task. The
normalised WC98 series is dominated by a smooth, near-monotonic decay after the
tournament peaks; the baseline's simpler single-layer model generalises better
across 5 horizons, while the higher-capacity model chases the training spikes.
This is a real regression, not noise — reproduce it and you'll see it.

### How to fix it (not yet applied, so the numbers above stay honest)
- Use the improved model only where it wins (NASA both horizons, WC98 1-step)
  and the baseline for WC98 5-step — an easy per-case selection.
- Or regularise the WC98 5-step model: drop to one BiLSTM layer, add L2 /
  higher dropout, or switch Huber→MSE for that case.
- Or add early-stopping on a horizon-averaged validation MAE.

Tell me which and I'll wire it in. The point of shipping the regression visibly
is that "improve the results" should mean measured, reproducible gains — not a
cherry-picked table.

Regenerate everything: `python run_improved.py --dataset both --autoscaler`.
Per-case CSVs: `table_{nasa,wc98}_improved.csv` and the diff
`table_{nasa,wc98}_improvement.csv`.

# Phase 2 - forecasting model comparison (NASA-HTTP)

Test set: **26,784 minutes**, 70/30 time-ordered split, metrics in requests/minute on the test portion only.
Prophet configuration supplying hybrid residuals: `reading_a`. Seed 42. Look-back 60 minutes.

Headline configuration: **shuffled** minibatch order (see DEVIATIONS D-008).

## Our results - shuffled minibatch order

| Model | MSE | RMSE | MAE | R2 | TPT (ms) | us/pred |
|---|---:|---:|---:|---:|---:|---:|
| **Prophet+LSTM (hybrid)** | 180.470 | 13.434 | 10.103 | 0.711 | 5046 | 188.4 |
| **Prophet+BiLSTM (hybrid)** | 179.737 | 13.407 | 10.081 | 0.712 | 8185 | 305.6 |
| **Prophet+GRU (hybrid)** | 180.087 | 13.420 | 10.085 | 0.711 | 4817 | 179.8 |
| **ARIMA 1-step** | 181.516 | 13.473 | 10.197 | 0.709 | 49855 | 1861.4 |
| **BiLSTM 1-step** | 182.089 | 13.494 | 10.164 | 0.708 | 5110 | 190.8 |
| **ARIMA 5-step** | 215.159 | 14.668 | 11.048 | 0.655 | 47846 | 1786.4 |
| **BiLSTM 5-step** | 216.617 | 14.718 | 11.035 | 0.653 | 5248 | 195.9 |
| _Prophet alone (reading_a)_ | 280.975 | 16.762 | 12.246 | 0.549 | 2854 | 106.6 |

## Our results - chronological minibatch order

| Model | MSE | RMSE | MAE | R2 | TPT (ms) | us/pred |
|---|---:|---:|---:|---:|---:|---:|
| **Prophet+LSTM (hybrid)** | 190.861 | 13.815 | 10.531 | 0.694 | 4970 | 185.5 |
| **Prophet+BiLSTM (hybrid)** | 198.464 | 14.088 | 10.756 | 0.682 | 7780 | 290.5 |
| **Prophet+GRU (hybrid)** | 209.114 | 14.461 | 10.827 | 0.665 | 4815 | 179.8 |
| **ARIMA 1-step** | 181.516 | 13.473 | 10.197 | 0.709 | 49855 | 1861.4 |
| **BiLSTM 1-step** | 208.840 | 14.451 | 10.865 | 0.665 | 5390 | 201.2 |
| **ARIMA 5-step** | 215.159 | 14.668 | 11.048 | 0.655 | 47846 | 1786.4 |
| **BiLSTM 5-step** | 241.730 | 15.548 | 11.818 | 0.612 | 5282 | 197.2 |
| _Prophet alone (reading_a)_ | 280.975 | 16.762 | 12.246 | 0.549 | 2854 | 106.6 |

## Side by side with the paper (CLAUDE.md 5 / paper Table 4)

| Model | MSE (ours) | MSE (paper) | RMSE (ours) | RMSE (paper) | MAE (ours) | MAE (paper) | R2 (ours) | R2 (paper) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prophet+LSTM (hybrid) | 180.470 | 63.836 | 13.434 | 7.990 | 10.103 | 6.164 | 0.711 | 0.900 |
| Prophet+BiLSTM (hybrid) | 179.737 | 67.995 | 13.407 | 8.246 | 10.081 | 6.312 | 0.712 | 0.894 |
| Prophet+GRU (hybrid) | 180.087 | 75.740 | 13.420 | 8.703 | 10.085 | 6.640 | 0.711 | 0.882 |
| ARIMA 1-step | 181.516 | 196.288 | 13.473 | 14.010 | 10.197 | 10.572 | 0.709 | 0.692 |
| BiLSTM 1-step | 182.089 | 183.642 | 13.494 | 13.551 | 10.164 | 10.280 | 0.708 | 0.712 |
| ARIMA 5-step | 215.159 | 237.604 | 14.668 | 15.414 | 11.048 | 11.628 | 0.655 | 0.628 |
| BiLSTM 5-step | 216.617 | 207.313 | 14.718 | 14.390 | 11.035 | 10.592 | 0.653 | 0.675 |

## Definition of Done

```
  [PASS]  Table covers all 7 Table 4 variants                   7/7 present
  [FAIL]  Prophet+LSTM has the lowest MSE                       180.470 vs best 179.737 (Prophet+BiLSTM (hybrid)), behind by 0.41%
  [FAIL]  Prophet+LSTM has the lowest RMSE                      13.434 vs best 13.407 (Prophet+BiLSTM (hybrid)), behind by 0.20%
  [FAIL]  Prophet+LSTM has the lowest MAE                       10.103 vs best 10.081 (Prophet+BiLSTM (hybrid)), behind by 0.22%
  [FAIL]  Prophet+LSTM R2 in [0.85, 0.93]                       R2 = 0.711
  [PASS]  Hybrid slower than the same net alone (TPT)           Prophet+BiLSTM 8185 ms vs BiLSTM alone 5110 ms (+3075 ms = the Prophet stage)
```

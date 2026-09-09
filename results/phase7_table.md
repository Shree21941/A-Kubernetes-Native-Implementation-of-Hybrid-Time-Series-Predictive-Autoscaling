# Phase 7 results - arms x segments

Mean ± half-width of the 95% confidence interval on the mean (Student's t, n=5, df=4, t=2.776).

**All runs included**, flagged ones among them. See `--exclude-flagged` for the sensitivity check.

## segment: diurnal

| metric | hpa-cpu | keda-rate | predictive-prophet | predictive-hybrid |
|---|---|---|---|---|
| p95 latency (ms) | 432.3 ± 358.1 | 213.8 ± 114.8 | 1957.2 ± 4663.0 | 240.9 ± 47.9 |
| p99 latency (ms) | 805.7 ± 613.3 | 381.0 ± 212.6 | 3916.5 ± 9500.4 | 427.1 ± 77.1 |
| SLO violation % | 3.170 ± 4.944 | 0.329 ± 0.464 | 11.543 ± 29.367 | 0.235 ± 0.236 |
| mean replicas | 3.35 ± 1.07 | 4.81 ± 0.04 | 2.26 ± 0.02 | 4.29 ± 0.07 |
| CPU-seconds | 862.3 ± 168.5 | 795.9 ± 194.4 | 803.6 ± 295.0 | 902.0 ± 94.6 |
| replica-seconds | 9637 ± 3076 | 13848 ± 105 | 6511 ± 70 | 12353 ± 208 |
| scaling actions | 38.0 ± 13.8 | 72.8 ± 2.7 | 7.8 ± 2.5 | 18.8 ± 4.5 |
| replica churn | 44.0 ± 25.5 | 81.0 ± 5.3 | 15.0 ± 7.0 | 29.0 ± 5.8 |
| direction reversals | 28.2 ± 7.2 | 47.0 ± 4.6 | 4.2 ± 1.4 | 11.6 ± 2.7 |
| pending-pod seconds | 32.4 ± 25.2 | 53.3 ± 4.6 | 68.4 ± 160.7 | 21.4 ± 3.4 |
| forecast MAPE % | n/a | n/a | 47.9 ± 6.0 | 28.3 ± 1.4 |
| forecast RMSE | n/a | n/a | 35.67 ± 5.51 | 18.98 ± 0.51 |
| inference p50 (ms) | n/a | n/a | 0.03 ± 0.00 | 3.59 ± 1.09 |
| inference p95 (ms) | n/a | n/a | 0.04 ± 0.02 | 6.78 ± 1.90 |

n per arm: [5]

## segment: burst

| metric | hpa-cpu | keda-rate | predictive-prophet | predictive-hybrid |
|---|---|---|---|---|
| p95 latency (ms) | 205.8 ± 43.2 | 354.5 ± 588.8 | 237.0 ± 214.6 | 141.3 ± 59.0 |
| p99 latency (ms) | 392.9 ± 129.2 | 580.1 ± 924.4 | 405.3 ± 327.0 | 240.2 ± 111.9 |
| SLO violation % | 0.294 ± 0.437 | 3.239 ± 8.927 | 0.777 ± 2.055 | 0.012 ± 0.033 |
| mean replicas | 2.01 ± 0.24 | 3.30 ± 0.11 | 2.48 ± 0.39 | 2.96 ± 0.17 |
| CPU-seconds | 212.0 ± 21.5 | 250.1 ± 99.3 | 242.7 ± 93.8 | 202.1 ± 14.9 |
| replica-seconds | 2169 ± 260 | 3563 ± 122 | 2674 ± 419 | 3192 ± 181 |
| scaling actions | 9.8 ± 5.4 | 21.8 ± 4.7 | 4.4 ± 2.9 | 8.6 ± 3.6 |
| replica churn | 10.0 ± 5.8 | 22.4 ± 4.1 | 5.6 ± 4.4 | 10.8 ± 4.0 |
| direction reversals | 7.8 ± 5.4 | 14.4 ± 2.3 | 1.6 ± 1.9 | 4.0 ± 1.8 |
| pending-pod seconds | 6.3 ± 4.4 | 17.8 ± 2.9 | 5.3 ± 2.0 | 9.0 ± 3.0 |
| forecast MAPE % | n/a | n/a | 42.3 ± 2.1 | 36.2 ± 3.9 |
| forecast RMSE | n/a | n/a | 24.17 ± 0.87 | 16.10 ± 0.96 |
| inference p50 (ms) | n/a | n/a | 0.03 ± 0.01 | 3.15 ± 0.14 |
| inference p95 (ms) | n/a | n/a | 0.05 ± 0.02 | 5.91 ± 0.36 |

n per arm: [5]

## segment: quiet

| metric | hpa-cpu | keda-rate | predictive-prophet | predictive-hybrid |
|---|---|---|---|---|
| p95 latency (ms) | 5328.1 ± 2137.3 | 676.8 ± 644.3 | 1199.2 ± 622.8 | 1966.5 ± 1860.2 |
| p99 latency (ms) | 12747.2 ± 5369.9 | 1722.9 ± 2068.5 | 1979.2 ± 1114.9 | 3454.6 ± 3235.0 |
| SLO violation % | 29.566 ± 10.256 | 5.579 ± 7.595 | 13.747 ± 5.984 | 16.315 ± 14.918 |
| mean replicas | 3.19 ± 0.31 | 5.52 ± 0.21 | 4.24 ± 0.33 | 5.17 ± 0.35 |
| CPU-seconds | 74.7 ± 118.6 | 163.6 ± 44.4 | 169.1 ± 50.3 | 113.5 ± 107.7 |
| replica-seconds | 1723 ± 168 | 2976 ± 113 | 2288 ± 177 | 2787 ± 191 |
| scaling actions | 11.4 ± 3.8 | 25.4 ± 2.6 | 11.4 ± 0.7 | 13.8 ± 2.7 |
| replica churn | 16.6 ± 5.4 | 30.8 ± 3.6 | 27.0 ± 3.0 | 32.2 ± 2.2 |
| direction reversals | 5.8 ± 1.8 | 7.2 ± 2.2 | 5.0 ± 1.8 | 7.0 ± 1.8 |
| pending-pod seconds | 31.3 ± 18.9 | 33.4 ± 19.6 | 24.1 ± 17.6 | 59.3 ± 47.5 |
| forecast MAPE % | n/a | n/a | 60.8 ± 1.7 | 30.7 ± 9.9 |
| forecast RMSE | n/a | n/a | 41.77 ± 1.23 | 18.75 ± 3.54 |
| inference p50 (ms) | n/a | n/a | 0.02 ± 0.00 | 3.77 ± 1.18 |
| inference p95 (ms) | n/a | n/a | 0.03 ± 0.00 | 6.49 ± 3.58 |

n per arm: [5]


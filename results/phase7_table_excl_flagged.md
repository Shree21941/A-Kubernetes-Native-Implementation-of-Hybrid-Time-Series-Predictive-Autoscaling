# Phase 7 results - arms x segments

Mean ± half-width of the 95% confidence interval on the mean (Student's t, n=5, df=4, t=2.776).

**Flagged runs EXCLUDED** (4): hpa-cpu/diurnal/rep5, hpa-cpu/diurnal/rep1, predictive-prophet/diurnal/rep5, predictive-prophet/diurnal/rep3

## segment: diurnal

| metric | hpa-cpu | keda-rate | predictive-prophet | predictive-hybrid |
|---|---|---|---|---|
| p95 latency (ms) | 343.0 ± 494.6 | 213.8 ± 114.8 | 224.2 ± 290.6 | 240.9 ± 47.9 |
| p99 latency (ms) | 647.0 ± 1059.2 | 381.0 ± 212.6 | 413.6 ± 615.7 | 427.1 ± 77.1 |
| SLO violation % | 1.763 ± 5.879 | 0.329 ± 0.464 | 0.523 ± 2.244 | 0.235 ± 0.236 |
| mean replicas | 2.91 ± 0.07 | 4.81 ± 0.04 | 2.25 ± 0.05 | 4.29 ± 0.07 |
| CPU-seconds | 898.3 ± 57.5 | 795.9 ± 194.4 | 886.0 ± 26.8 | 902.0 ± 94.6 |
| replica-seconds | 8390 ± 198 | 13848 ± 105 | 6484 ± 134 | 12353 ± 208 |
| scaling actions | 33.7 ± 11.2 | 72.8 ± 2.7 | 6.7 ± 2.9 | 18.8 ± 4.5 |
| replica churn | 35.3 ± 13.7 | 81.0 ± 5.3 | 12.3 ± 14.3 | 29.0 ± 5.8 |
| direction reversals | 27.0 ± 8.6 | 47.0 ± 4.6 | 3.7 ± 2.9 | 11.6 ± 2.7 |
| pending-pod seconds | 23.8 ± 10.5 | 53.3 ± 4.6 | 9.5 ± 7.0 | 21.4 ± 3.4 |
| forecast MAPE % | n/a | n/a | 50.1 ± 1.0 | 28.3 ± 1.4 |
| forecast RMSE | n/a | n/a | 37.61 ± 1.49 | 18.98 ± 0.51 |
| inference p50 (ms) | n/a | n/a | 0.02 ± 0.00 | 3.59 ± 1.09 |
| inference p95 (ms) | n/a | n/a | 0.03 ± 0.00 | 6.78 ± 1.90 |

n per arm: [3, 5]

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


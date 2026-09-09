# Phase 8 — Findings

Reproduction and live evaluation of Guruge & Priyadarshana (2025), *Time series
forecasting-based Kubernetes autoscaling using Facebook Prophet and LSTM*
(Front. Comput. Sci. 7:1509165), extended with a KEDA baseline the paper does not have.

**Dataset.** 60 runs = 4 arms × 3 trace segments × 5 repetitions, executed as a seeded
shuffle on a 3-node KIND cluster, 2026-08-24 to 2026-08-26.
Sources: `experiments/runs/`, `results/phase7_table.md`, `results/figures/`.

All numbers below are **mean ± half-width of the 95% CI on the mean** (Student's *t*,
n=5, df=4, t=2.776), with the four runs flagged by the Phase 7 integrity pass excluded.
`results/phase7_table.md` gives the all-runs version for comparison.

---

## 1. The one-paragraph answer

The hybrid forecaster **reproduces**: it predicts substantially better than Prophet
alone, and it drives an autoscaler that is markedly more stable than either reactive
baseline. But **the latency advantage the paper implies is not demonstrated here** — at
n=5 every SLO comparison between arms has overlapping confidence intervals. What is
statistically solid is the *forecast quality* improvement and the *oscillation*
advantage. What is not solid, on this evidence, is "predictive autoscaling delivers
better user-facing latency."

---

## 2. What is statistically defensible

Two families of claim survive the confidence intervals.

### 2.1 The LSTM residual measurably fixes Prophet — in every segment

| segment | Prophet-only MAPE | Hybrid MAPE | Prophet bias | Hybrid bias | CIs |
|---|---|---|---|---|---|
| diurnal | 50.1 ± 1.0 % | **28.3 ± 1.4 %** | −32.97 | **−2.18** | separated |
| burst | 42.3 ± 2.1 % | **36.2 ± 3.9 %** | −20.03 | **−2.50** | separated |
| quiet | 60.8 ± 1.7 % | **30.7 ± 9.9 %** | −38.22 | **−0.50** | separated |

The bias column is the mechanism, and it is quantifiable. Across all 15 Prophet-only
runs, `bias / −MAE = 0.983`; for the 15 hybrid runs it is `0.124`. A ratio of 1.0 means
every forecast error has the same sign, so **Prophet-only under-predicts essentially
always**, while the hybrid's errors are scattered around zero. That is precisely what a
residual model is supposed to do, and it is the paper's central mechanism working as
advertised.

`fig5_forecast_vs_actual_burst.png` shows why in one picture: after cycle ~60 Prophet's
forecast **flatlines at 20–27 req/trace-min while actual load climbs past 80.** Prophet
models trend and seasonality; a burst is by construction neither. The LSTM residual is
what recovers it.

### 2.2 Predictive control oscillates far less than KEDA

| segment | HPA | KEDA | Prophet-only | Hybrid |
|---|---|---|---|---|
| diurnal reversals | 27.0 | 47.0 ± 4.7 | **3.7** | 11.6 ± 2.7 |
| burst reversals | 7.8 | 14.4 ± 2.3 | **1.6** | 4.0 ± 1.8 |
| diurnal scaling actions | 33.7 | 72.8 ± 2.7 | **6.7** | 18.8 ± 4.5 |

KEDA/hybrid CIs are **separated** on both segments. KEDA issues roughly **4× the
scaling actions and 4× the direction reversals** of the hybrid. Each reversal costs a
pod cold start (measured: 2.70 s), so this is a real operational cost, not bookkeeping.
`PROJECT_PHASES.md` is right to treat stability as a headline result: KEDA's competitive
latency is *bought* with thrash.

---

## 3. What is NOT defensible — the negative result

**Every single SLO comparison between arms overlaps at n=5.**

| segment | HPA | KEDA | Prophet-only | Hybrid |
|---|---|---|---|---|
| diurnal SLO % | 1.76 ± 5.88 | 0.33 ± 0.46 | 0.52 ± 2.24 | 0.24 ± 0.24 |
| burst SLO % | 0.29 ± 0.44 | 3.24 ± 8.93 | 0.78 ± 2.06 | 0.01 ± 0.03 |
| quiet SLO % | 29.57 ± 10.26 | 5.58 ± 7.60 | 13.75 ± 5.98 | 16.31 ± 14.92 |

The hybrid has the lowest mean on diurnal and burst, and that is worth reporting — but
the intervals are wider than the gaps. **The honest statement is "the hybrid was not
worse, and was probably better, but five runs cannot separate these arms on latency."**
Anyone quoting "the hybrid beats HPA on SLO" from this dataset is over-reading it.

Two consequences worth stating plainly:

- **The Phase 7 Definition-of-Done check ("predictive not worse than HPA on burst
  latency") passes** — hybrid p95 141.3 ± 59.0 ms vs HPA 205.8 ± 43.2 ms — but it
  passes on means, and it too would not survive a strict separation test.
- **KEDA is the strongest arm on the quiet segment** (5.58 % vs the hybrid's 16.31 %),
  though again overlapping. Request-rate-driven reactive scaling is a *strong* baseline.
  "Predictive beats reactive" is only clearly true against **CPU-based** reactive.

### 3.1 The quiet segment defeats everyone

All four arms violate the SLO at double-digit or near-double-digit rates on `quiet`,
where they are near-flawless on `burst`. HPA reaches **29.6 %** with a p99 of 12.7 s.
This is not a forecasting failure — Prophet-only and the hybrid have their *best* bias
figures on this segment. The likely cause is that per-pod capacity (3.135 req/s,
measured under smooth load) is optimistic against Poisson arrivals at low replica
counts, where there is no statistical smoothing across pods and one slow request weighs
heavily in a small denominator. **This is unresolved and is the most important open
question in the dataset.**

---

## 4. What live evaluation showed that offline evaluation could not

This is the section that justifies Phases 3–7 existing at all.

**4.1 Inference latency is ~1000× better live than offline — and the reason matters.**
The paper reports TPT ≈ 3492 ms for the hybrid; our own Phase 2 offline measurement was
**8187 ms**, of which **2854 ms was Prophet's `predict()` call**. Live, the hybrid's
inference is **3.59 ± 1.09 ms p50, 6.78 ms p95.** The difference is not a better model —
it is that Prophet's contribution is precomputed into a lookup table (verified equal to
the live model to 1e-9), leaving only the LSTM forward pass in the control loop.

This directly de-risks the cold-start-aware contribution: **a forecast that arrives late
shrinks its own head start.** At 3.6 ms against a 6000 ms control period, inference
consumes 0.06 % of the budget. Had we deployed the naive 8.2 s pipeline, inference alone
would have exceeded the control period and the forecast would have been stale on
arrival. Offline TPT numbers cannot reveal this; only running the loop can.

**4.2 The cold-start head start is real and measurable.**
Measured pod cold start **2.70 s ± 0.13** (n=5) against a 6.0 s compressed horizon
leaves **+3.30 s of head start — 55 % of the horizon survives.** This was the open risk
in the whole design: a forecast one minute ahead is worthless if a pod takes longer than
a minute to become Ready. It does not, at this compression.

**4.3 Forecast quality and control quality are not the same thing.**
Prophet-only records a *worse* forecast on burst (42.3 % MAPE, badly under-predicting)
yet only 0.78 % SLO violations — because the replica floor and dead-band held it at 2
pods, which was sufficient. A model can forecast badly and control adequately. Offline
MAPE would have predicted a disaster that did not occur.

**4.4 MAPE is a poor proxy for control-relevant accuracy.**
In `fig5`, Prophet-only and the hybrid record similar MAPE on one burst run (40.5 % vs
39.1 %) while the plot shows one tracking the signal and the other flatlined. MAPE
punishes the hybrid's small phase errors as harshly as Prophet's total misses.

---

## 5. Deviations from the paper

Recorded in full in `DEVIATIONS.md`; the ones that affect interpretation:

1. **The paper has no Kubernetes implementation.** It states integration "has been
   hypothesized." Phases 3–7 are our engineering from their MAPE description plus the
   Dang-Quang & Yoo replica formula. We are not reproducing their controller — there
   isn't one.
2. **Equations 8 and 9 are identical in the published text** (both `pods_{t+1} <
   pods_t`). We implemented the evidently-intended asymmetric logic.
3. **Table 2 lists weekly seasonality twice** and never daily. We read it as daily
   Fourier order 20, weekly 50.
4. **10× time compression.** One trace-minute is replayed in 6 wall-seconds so a 60-run
   matrix fits in ~25 h. Every timing constant is compressed consistently; cold start is
   the one quantity that does **not** compress, which is why it is measured separately.
5. **KEDA arm added.** The paper has no request-rate reactive baseline. Adding it is
   what produced the most interesting negative result here.
6. **Prophet precomputed to a lookup table in-cluster.** Verified equal to the live
   model to 1e-9. See §4.1.

---

## 6. Integrity problems found during the sweep — and why they matter

Three measurement bugs were caught during Phase 7. **Each would have produced a
plausible, publishable, wrong conclusion.** They are documented here because the
process is part of the result.

1. **CPU recorded as 0.0 for a run that served 5,898 requests.** A cluster-wide *sum*
   of a per-pod counter was differenced across the run; pod turnover made the delta
   negative and a "clamp negatives" rule turned it into zero. Understated the quiet
   runs by ~50 %. Fixed to per-pod, reset-aware differencing, cross-checked against
   PromQL `increase()` (agreement now 1.06 %).
2. **KEDA counted as two autoscalers.** A ScaledObject creates its own HPA, so the
   exclusivity guard refused to start the arm. Fixed by filtering on `ownerReferences`.
3. **Host suspend mid-run.** The laptop slept during three diurnal runs (3110–5106 s
   gaps). `trace_replay` paces on `CLOCK_MONOTONIC`, which does not advance during
   suspend, so its own log reported a flawless 2879.7 s of 2880 s with zero errors —
   the run looked perfect by every self-check it had. Only the wall-clock sampler saw
   the hole. Because cost metrics are integrals over sampler time, `replica_seconds`
   was overstated **1.9×–2.7×**, and two of HPA's five diurnal reps were affected.
   Fixed by excising >10 s sampler gaps; runs rescored.

A fourth run (`prophet/diurnal/rep3`) is **flagged but deliberately kept**: 299.9 s of
pod-pending time against a 18.9 s median, i.e. pods were requested and could not be
scheduled. Including it moves Prophet's diurnal SLO from 0.52 % to 11.54 % and its p95
from 224 ms to 1957 ms — a 12× swing from one run. It is retained because a slow
scheduler is arguably part of the system under test, and discarding results that look
bad is how p-hacking starts. Both versions of every table are published.

---

## 7. Limitations

- **n=5 is small.** It is what the phase specified, and it is not enough to separate
  the arms on latency. The CIs in §3 are the honest consequence. Roughly n=20+ would be
  needed for the SLO gaps observed here.
- **Single machine, single cluster, single trace.** One laptop, KIND on Docker, the
  NASA-HTTP August 1995 log, one synthetic CPU-bound app. Nothing here establishes
  behaviour on real hardware, on other workloads, or with a different app profile.
- **10× compression is an assumption, not a neutral choice.** It holds only because pod
  cold start (2.70 s) remains small against the compressed horizon (6.0 s). At 60×
  compression the head start would vanish and the conclusions would invert.
- **The sweep was interrupted twice** — a host reboot (13 h lost, resumed cleanly) and
  three suspends. Environmental noise on a laptop is a real threat to a 25 h experiment.
- **Four of 60 runs are flagged.** All results are published both with and without them.
- **The quiet-segment SLO failure is unexplained** (§3.1). Until it is understood, no
  claim about low-traffic behaviour should be made from this dataset.
- **CPU-seconds uses boundary sampling.** A pod created *and* destroyed between the two
  samples is invisible to per-pod differencing; the PromQL cross-check agrees to ~1 %,
  so the effect is small but non-zero.
- **`hpa-cpu` scales on CPU, which the project argues is the wrong signal.** It is
  included as the paper's own baseline, not as a straw man — but it is losing partly by
  construction, and that should temper any "predictive beats reactive" reading.

---

## 8. What this means for the cold-start-aware contribution

1. **The head-start budget is real but finite** — 3.30 s of 6.0 s at C=10. A
   cold-start-aware horizon has something concrete to optimise.
2. **Inference latency must stay inside the budget**, and precomputation is what makes
   that possible (§4.1). This is a concrete, measured argument for the contribution.
3. **KEDA, not HPA, is the baseline to beat.** It matched or beat the predictive arms on
   latency in every segment. The defensible angle is *stability at equal latency* —
   where the hybrid already has a separated result — rather than latency alone.
4. **Fix the quiet-segment question first.** A contribution built on a testbed with an
   unexplained 30 % violation rate at low traffic will not survive review.

---

*Generated by `src/eval/aggregate.py` and `src/eval/figures.py` from the 60 run
directories in `experiments/runs/`. Every number here is reproducible from that data.*

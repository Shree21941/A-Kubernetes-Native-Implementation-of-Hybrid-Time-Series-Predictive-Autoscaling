"""Phase 5 - Analyze: turn a window of observed request rates into a t+1 forecast.

Loads the Phase 2 artifacts. NEVER trains - training in the control loop would put a
30-minute fit inside a 6-second cycle, and the paper's own design has the model trained
offline for exactly this reason (CLAUDE.md §2).

The hybrid at inference time, in the order the controller runs it:

    residual[t-59..t]  = observed[t-59..t] - prophet_yhat[t-59..t]
    residual_hat[t+1]  = net(scale(residual[t-59..t]))          # the only live model call
    forecast[t+1]      = prophet_yhat[t+1] + unscale(residual_hat[t+1])

Two properties of this decomposition matter operationally:

* **Prophet does not read the window.** Its term is a function of the timestamp alone, so
  it can be looked up (see export_bundle.py). The network is the only component that
  consumes live data.
* **The scaler must be the training scaler.** It is carried in the bundle as two floats
  fit on TRAIN residuals only. Re-fitting on the live window would rescale every cycle
  according to whatever the last hour happened to contain, and the network would receive
  inputs from a distribution it never saw.

--------------------------------------------------------------------------------
On live TPT
--------------------------------------------------------------------------------
Phase 2's TPT figures are whole-test-set wall clock over 26,784 predictions (D-005) and
say nothing about the cost of ONE forecast, which is the only number a 6-second control
loop cares about. This module times every cycle and the controller logs it. That is the
live TPT, and it belongs in the results.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Keep TensorFlow single-threaded and quiet before it loads. The controller shares a
# node with the pods it is measuring; letting TF grab every core would make the
# controller a source of the load it is trying to observe.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")


@dataclass
class Forecast:
    value: float                 # predicted requests in the next trace-minute
    prophet_term: float
    residual_term: float
    net_ms: float
    prophet_ms: float
    total_ms: float
    clamped: bool                # forecast was negative and floored at 0


class Forecaster:
    def __init__(self, bundle: Path, prophet_mode: str = "table",
                 threads: int = 1, model: str = "hybrid") -> None:
        """model: 'hybrid' (Prophet + network on residuals) or 'prophet' (Prophet alone).

        The Prophet-only variant is an ABLATION ARM, not a lesser deployment. It answers
        the question a reviewer will certainly ask and that D-009 leaves open in both
        directions: the hybrid's accuracy edge over its alternatives is inside run-to-run
        noise (0.6% spread across five 1-step models), and D-015 removed the cost
        argument against it (7.16 ms, not 3.5 s). So neither "the hybrid is necessary"
        nor "one model would have done" is currently supported by our own evidence.

        The two arms must therefore differ in EXACTLY one thing - whether the residual
        network is consulted - which is why this is a mode of one class rather than a
        second implementation. A separate Prophet-only forecaster would inevitably differ
        in some second detail, and the ablation would stop being an ablation.
        """
        if model not in ("hybrid", "prophet"):
            raise ValueError(f"model must be 'hybrid' or 'prophet', got {model!r}")
        self.model = model
        self.bundle = Path(bundle)
        self.meta = json.loads((self.bundle / "bundle.json").read_text())
        self.look_back: int = self.meta["look_back"]
        self.horizon: int = self.meta["horizon"]

        sc = self.meta["resid_scaler"]
        self._rmin = float(sc["data_min"])
        self._rspan = float(sc["data_max"]) - self._rmin

        # TensorFlow is imported and the network loaded ONLY for the hybrid arm. The
        # Prophet-only arm has no network, and loading one it never calls would put ~2 s
        # of import and a few hundred MB of RSS into an arm whose whole point is being
        # lighter - and would make any resource comparison between the arms meaningless.
        self._infer = None
        self.net = None
        if self.model == "hybrid":
            import tensorflow as tf

            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(threads)
            self.net = tf.keras.models.load_model(self.bundle / "net.keras")
            self._build_graph(tf)

        self._init_prophet(prophet_mode)

    def _build_graph(self, tf) -> None:
        # Compile one graph for exactly the shape the controller uses, and call THAT
        # every cycle. This is not a micro-optimisation - it is 68x:
        #
        #     eager  self.net(x)                        419 ms per forecast
        #     compiled tf.function, fixed signature       6 ms per forecast
        #
        # measured on this machine, single-threaded, batch of 1 (see
        # results/phase5_forecaster_verify.json). The network is 30,651 parameters over
        # 60 timesteps; 419 ms is not arithmetic, it is TensorFlow's per-call dispatch
        # and shape-retracing overhead, paid again on every invocation because a fresh
        # NumPy array gives Keras no reason to believe the shape is stable.
        #
        # Pinning the input_signature tells it the shape never changes, so the graph is
        # traced once here and reused. This matters because the whole point of the
        # forecast is to buy a head start: at C=10 the horizon is 6 wall-seconds, and
        # spending 0.42 s of it on framework overhead would consume 7% of the head start
        # to compute the head start.
        @tf.function(input_signature=[
            tf.TensorSpec([1, self.look_back, 1], tf.float32)])
        def _infer(x):
            return self.net(x, training=False)

        self._infer = _infer
        self._infer(tf.zeros((1, self.look_back, 1), dtype=tf.float32))  # trace now

    def _init_prophet(self, prophet_mode: str) -> None:
        self.prophet_mode = prophet_mode
        self._table: dict[int, float] = {}
        self._prophet = None

        if prophet_mode in ("table", "both"):
            t = np.load(self.bundle / "prophet_table.npz")
            self._table = dict(zip(t["ds"].tolist(), t["yhat"].tolist()))
        if prophet_mode in ("live", "both"):
            from prophet.serialize import model_from_json
            self._prophet = model_from_json(
                (self.bundle / "prophet.json").read_text())

    # -- Prophet term ------------------------------------------------------------

    def _yhat_table(self, ts: np.ndarray) -> np.ndarray:
        keys = ts.astype("datetime64[m]").astype("int64")
        missing = [k for k in keys.tolist() if k not in self._table]
        if missing:
            # Loud, not silent. A missing timestamp means the controller is being run
            # outside the test split the bundle covers - which would mean replaying data
            # the network was trained on, and every accuracy number would be void.
            first = np.datetime64(missing[0], "m")
            raise KeyError(
                f"{len(missing)} timestamp(s) absent from the Prophet table, first "
                f"{first}. The table covers {self.meta['table_covers'][0]} to "
                f"{self.meta['table_covers'][1]} (test split only)."
            )
        return np.array([self._table[k] for k in keys.tolist()], dtype="float64")

    def _yhat_live(self, ts: np.ndarray) -> np.ndarray:
        import pandas as pd
        df = pd.DataFrame({"ds": pd.to_datetime(ts)})
        return self._prophet.predict(df)["yhat"].to_numpy()

    def prophet_yhat(self, ts: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        if self.prophet_mode == "live":
            y = self._yhat_live(ts)
        elif self.prophet_mode == "both":
            y = self._yhat_table(ts)
            live = self._yhat_live(ts)
            # The whole justification for the table is that it is identical, not merely
            # close. Verify rather than assert in prose.
            if not np.allclose(y, live, atol=1e-9):
                raise ValueError(
                    f"Prophet table disagrees with live model by up to "
                    f"{np.abs(y - live).max():.3e} - the table is stale, refuse to run"
                )
        else:
            y = self._yhat_table(ts)
        return y, (time.perf_counter() - t0) * 1000.0

    # -- the forecast ------------------------------------------------------------

    def predict(self, window: np.ndarray, window_ts: np.ndarray,
                target_ts: np.datetime64) -> Forecast:
        """window: the last `look_back` observed counts, oldest first."""
        if len(window) != self.look_back:
            raise ValueError(
                f"need exactly {self.look_back} samples, got {len(window)} - the "
                f"controller must not forecast on a partial window"
            )
        if len(window_ts) != self.look_back:
            raise ValueError("window and window_ts lengths differ")

        all_ts = np.concatenate([window_ts, np.array([target_ts])])
        yhat_all, p_ms = self.prophet_yhat(all_ts)
        yhat_window, yhat_target = yhat_all[:-1], float(yhat_all[-1])

        if self.model == "prophet":
            # The ablation arm. Prophet alone: trend + seasonality at the target
            # timestamp, and nothing else. Note that this arm never consults `window` at
            # all - it is open-loop with respect to observed traffic, which is exactly
            # the property under test. If it performs comparably to the hybrid, then the
            # residual network - and the live Monitor stage that feeds it - is not
            # earning its place.
            #
            # Phase 2 measured Prophet alone at RMSE 16.76 / R^2 0.549 against the
            # hybrid's 13.43 / 0.711, so we expect it to be worse. The question Phase 7
            # asks is whether that accuracy gap survives contact with a control loop -
            # a forecaster feeding ceil() into a clamped replica count may well round
            # away differences that look large in RMSE.
            return Forecast(
                value=max(yhat_target, 0.0), prophet_term=yhat_target,
                residual_term=0.0, net_ms=0.0, prophet_ms=p_ms, total_ms=p_ms,
                clamped=yhat_target < 0.0,
            )

        resid = np.asarray(window, dtype="float64") - yhat_window
        scaled = (resid - self._rmin) / self._rspan

        t0 = time.perf_counter()
        out = self._infer(scaled.reshape(1, self.look_back, 1).astype("float32"))
        net_ms = (time.perf_counter() - t0) * 1000.0

        resid_hat = float(np.asarray(out).reshape(-1)[0]) * self._rspan + self._rmin
        value = yhat_target + resid_hat

        # A request rate cannot be negative. Prophet's linear trend plus a residual can
        # go below zero in the quiet segment, and a negative forecast would produce
        # desired_replicas = 0 through the ceil() - taking the service to zero pods
        # because the model was slightly pessimistic about an already-quiet minute.
        clamped = value < 0.0
        return Forecast(
            value=max(value, 0.0), prophet_term=yhat_target, residual_term=resid_hat,
            net_ms=net_ms, prophet_ms=p_ms, total_ms=p_ms + net_ms, clamped=clamped,
        )

    def predict_horizon(self, window: np.ndarray, window_ts: np.ndarray,
                        first_target_ts: np.datetime64, steps: int) -> Forecast:
        """Recursively forecast `steps` trace-minutes and return the final step.

        Recursive use is for matching the lead time to measured pod readiness when the
        packaged network only has a one-step output. The caller must keep `steps`
        bounded and report the horizon because forecast error compounds at each step.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        values = np.asarray(window, dtype="float64").copy()
        timestamps = np.asarray(window_ts, dtype="datetime64[m]").copy()
        total_net_ms = total_prophet_ms = 0.0
        any_clamped = False
        result = None

        for offset in range(steps):
            target_ts = first_target_ts + np.timedelta64(offset, "m")
            result = self.predict(values, timestamps, target_ts)
            total_net_ms += result.net_ms
            total_prophet_ms += result.prophet_ms
            any_clamped = any_clamped or result.clamped
            values = np.concatenate([values[1:], [result.value]])
            timestamps = np.concatenate([timestamps[1:], [target_ts]])

        return Forecast(
            value=result.value,
            prophet_term=result.prophet_term,
            residual_term=result.residual_term,
            net_ms=total_net_ms,
            prophet_ms=total_prophet_ms,
            total_ms=total_net_ms + total_prophet_ms,
            clamped=any_clamped,
        )

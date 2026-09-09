"""Phase 5 - every constant the controller uses, and the clock they all live in.

This module exists because of a recurring failure in this project: a time constant
derived under one clock and used under another. It has happened three times now -
WORK_ITERATIONS sized in real time for a compressed experiment (D-012), the Prometheus
scrape interval left at 15s while the control loop ticks every 6s (D-014), and the
cooldown question below. Putting every duration in one file, each stated in BOTH clocks,
makes the next instance visible instead of silent.

--------------------------------------------------------------------------------
The two clocks
--------------------------------------------------------------------------------
TRACE TIME is the clock of the 1995 NASA data and of the forecasting model. The model
was trained on per-minute counts and predicts one *trace-minute* ahead.

WALL TIME is the clock of this laptop, of Kubernetes, and of pod startup.

They are related by the compression factor C (D-010):

    1 trace-minute  =  60 / C  wall-seconds        (6.0 s at C=10)

Everything the EXPERIMENT defines - forecast horizon, control period, look-back window,
cooldowns, evaluation windows - is defined in trace time and converted here.

Everything PHYSICAL - pod startup, image pull, probe delay, model inference - is measured
in wall time and does NOT convert. This is the asymmetry that makes compression
interesting rather than merely convenient, and the one place the two clocks must be
compared directly is `scale_up_cooldown_wall_s`, below.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Measured constants, written by experiments/wire_measurements.py from the cluster
# measurements in results/phase3_coldstart.json and results/phase4_capacity.json.
#
# Loaded as the DEFAULT source so that a Config() built with no environment - which is
# what experiments/run_matrix.py does when it records each run's provenance - reports the
# same numbers the in-cluster controller is actually running with. Before this existed,
# run_matrix would have written the hardcoded placeholders (3.245 req/s, 5.0 s cold
# start) into all 60 runs' config.json while the cluster ran on the measured values, and
# every run would have carried a false record of its own configuration.
#
# Precedence: environment variable > measured_constants.json > placeholder literal.
# Env still wins because the Deployment sets it explicitly and a run must be able to
# override for a sensitivity test.
_MEASURED_PATH = REPO / "experiments" / "measured_constants.json"
try:
    _MEASURED: dict = json.loads(_MEASURED_PATH.read_text())
except (OSError, ValueError):
    _MEASURED = {}


def _measured_f(key: str, name: str, placeholder: float) -> float:
    """Env, else the measured file, else the placeholder."""
    if name in os.environ:
        return float(os.environ[name])
    return float(_MEASURED.get(key, placeholder))


def _env_f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    # ---- the clock ---------------------------------------------------------------
    compression: float = field(default_factory=lambda: _env_f("COMPRESSION", 10.0))

    # ---- MAPE loop, defined in TRACE minutes --------------------------------------
    # One decision per trace-minute, forecasting one trace-minute ahead. This is the
    # paper's t+1 horizon; it is 60 trace-seconds regardless of what C is.
    control_period_trace_min: float = 1.0
    horizon_trace_min: int = 1
    # The look-back the Phase 2 networks were trained with. Changing it here without
    # retraining would feed the model a window shape it has never seen.
    look_back_trace_min: int = 60

    # ---- replica bounds ------------------------------------------------------------
    # Identical across every autoscaler arm in Phase 6. Different bounds alone can
    # manufacture a "win", so these belong to the experiment, not to this arm.
    min_replicas: int = field(default_factory=lambda: _env_i("MIN_REPLICAS", 1))
    max_replicas: int = field(default_factory=lambda: _env_i("MAX_REPLICAS", 10))

    # ---- capacity ------------------------------------------------------------------
    # req/s one pod serves inside SLO. MEASURED (D-012), never guessed - it is the
    # denominator of the replica formula, so an error here is a constant multiplier on
    # every decision, presenting as a forecasting error rather than a calibration one.
    per_pod_capacity_rps: float = field(
        default_factory=lambda: _measured_f(
            "per_pod_capacity_rps", "PER_POD_CAPACITY_RPS", 3.245))

    # ---- safety layer, in TRACE minutes --------------------------------------------
    # Asymmetric by design. Scaling up wrongly costs money; scaling down wrongly costs
    # SLO violations, and the pods you removed take real wall-seconds to come back.
    scale_up_cooldown_trace_min: float = 1.0
    scale_down_cooldown_trace_min: float = 5.0
    # Dead-band: ignore a desired change of fewer than this many pods.
    #
    # Applied to SCALE-DOWN ONLY (see planner.plan). Dead-banding the up direction would
    # delay exactly the head start the forecast exists to buy - the whole point of being
    # predictive is to move early, and a controller that waits for a 2-pod deficit
    # before growing has thrown that away to avoid a cost it does not pay.
    #
    # 2, not 1. At 1 this setting is a NO-OP: the test is `|desired - current| < band`,
    # and for integers that is never true when they differ, so a band of 1 filters
    # nothing. It shipped as 1 and was caught by test_planner.py only because the
    # dead-band test used a band of 2 explicitly.
    #
    # 2 is justified by the forecaster's own error: Phase 2 measured RMSE 13.4 req/min
    # (D-009), and at 18.81 req/trace-min per pod (the MEASURED capacity) that is
    # 0.71 pods. A one-pod
    # difference is inside the noise of the thing producing it.
    deadband_replicas: int = 2
    # Scale-down damping: require the LOWER desired count to persist for this many
    # consecutive decisions before acting on it. The quiet segment's 9-minute zero run
    # is exactly what this is for - a dropout that recovers must not cost a full
    # scale-down-and-back-up cycle.
    scale_down_confirmations: int = 3

    # ---- physical, WALL seconds, does not compress ---------------------------------
    # Measured pod cold start, from src/eval/coldstart.py via measured_constants.json.
    # The 5.0 literal below is only the fallback for a repo where that measurement has
    # not been taken; `pod_cold_start_measured` is False in that case and the controller
    # prints a warning on every start, so a run built on a guess cannot become a result.
    pod_cold_start_wall_s: float = field(
        default_factory=lambda: _measured_f(
            "pod_cold_start_wall_s", "POD_COLD_START_WALL_S", 5.0))
    pod_cold_start_measured: bool = field(
        default_factory=lambda: (os.environ.get("POD_COLD_START_MEASURED") == "1")
        or ("pod_cold_start_wall_s" in _MEASURED
            and "POD_COLD_START_MEASURED" not in os.environ))

    # ---- which forecaster this arm uses ---------------------------------------------
    # "hybrid"  = Prophet + residual network (the paper's method, Phase 2's headline)
    # "prophet" = Prophet alone (the Phase 7 ablation arm)
    #
    # Selected by env var rather than by deploying two different controllers, so that the
    # two predictive arms are guaranteed to share every other line of code. An ablation
    # that also differs in its plumbing is not an ablation.
    forecaster: str = field(default_factory=lambda: os.environ.get("FORECASTER", "hybrid"))

    # ---- warm-up -------------------------------------------------------------------
    # Until the history window is full the model cannot be asked for a forecast. Falling
    # back to a REACTIVE rule (scale on observed rate) is better than scaling on a
    # half-empty window, because a padded window is not a shorter history - it is a
    # wrong one, and the model has no way to know.
    warmup_policy: str = "reactive"      # "reactive" | "hold"

    # ---- plumbing ------------------------------------------------------------------
    prometheus_url: str = field(default_factory=lambda: os.environ.get(
        "PROMETHEUS_URL", "http://prometheus-operated.monitoring.svc:9090"))
    namespace: str = field(default_factory=lambda: os.environ.get("NAMESPACE", "aiops"))
    deployment: str = field(default_factory=lambda: os.environ.get(
        "DEPLOYMENT", "target-app"))
    bundle: Path = field(default_factory=lambda: Path(os.environ.get(
        "BUNDLE", str(REPO / "models" / "controller_bundle"))))
    decisions_path: Path = field(default_factory=lambda: Path(os.environ.get(
        "DECISIONS_PATH", "/var/log/controller/decisions.jsonl")))

    # ---- derived: wall-clock values the loop actually uses --------------------------

    @property
    def trace_minute_wall_s(self) -> float:
        return 60.0 / self.compression

    @property
    def control_period_wall_s(self) -> float:
        return self.control_period_trace_min * self.trace_minute_wall_s

    @property
    def look_back_wall_s(self) -> float:
        return self.look_back_trace_min * self.trace_minute_wall_s

    @property
    def scale_up_cooldown_wall_s(self) -> float:
        """Where the two clocks collide, and the compressed one must not win.

        The experiment says "wait one trace-minute between scale-ups" = 6 wall-seconds
        at C=10. But pod startup is PHYSICAL and does not compress. If pods need 8 wall-
        seconds to become Ready and we allow another scale-up after 6, the controller
        acts again on a cluster whose previous decision has not landed yet - it sees the
        old replica count, concludes it is still short, and adds more. That is classic
        integral windup, and it would look like the forecaster over-predicting.

        So the cooldown is the LONGER of the two: the experiment's compressed interval,
        and the physical time for the previous decision to take effect.
        """
        return max(self.scale_up_cooldown_trace_min * self.trace_minute_wall_s,
                   self.pod_cold_start_wall_s)

    @property
    def scale_down_cooldown_wall_s(self) -> float:
        # No physical floor needed: removing pods is fast, and the damping above is what
        # protects against premature scale-down.
        return self.scale_down_cooldown_trace_min * self.trace_minute_wall_s

    @property
    def per_pod_capacity_rpm_trace(self) -> float:
        """Capacity expressed in the units the FORECASTER speaks.

        The model predicts trace-requests-per-trace-minute. The pod serves wall-requests
        per wall-second. One trace-minute of load is delivered in 60/C wall-seconds, so

            offered_rps = predicted_rpm / (60 / C) = predicted_rpm * C / 60

        Rearranged, a pod's capacity in predicted-rpm units is capacity_rps * 60 / C.
        Getting this conversion backwards would scale the whole experiment by C^2 = 100,
        which is why it lives here and not inline in the planner.
        """
        return self.per_pod_capacity_rps * 60.0 / self.compression

    def warnings(self) -> list[str]:
        """Conditions that would make results misleading. Logged loudly at startup."""
        w: list[str] = []
        if not self.pod_cold_start_measured:
            w.append(
                f"pod_cold_start_wall_s={self.pod_cold_start_wall_s}s is a PLACEHOLDER, "
                "not a measurement (Phase 3 / src/eval/coldstart.py has not run). "
                "Cooldowns and any cold-start claim derived from this run are provisional."
            )
        # The whole premise of a t+1 forecast is that a pod can be Ready within the
        # horizon. If it cannot, the forecast is correct and useless (D-004).
        horizon_wall = self.horizon_trace_min * self.trace_minute_wall_s
        if self.pod_cold_start_wall_s >= horizon_wall:
            w.append(
                f"pod cold start ({self.pod_cold_start_wall_s:.1f}s wall) is >= the "
                f"forecast horizon ({horizon_wall:.1f}s wall at C={self.compression:g}). "
                "No t+1 forecaster can be proactive under these conditions - this is "
                "the D-004 regime and must be reported, not averaged away."
            )
        elif self.pod_cold_start_wall_s > 0.5 * horizon_wall:
            w.append(
                f"pod cold start consumes {self.pod_cold_start_wall_s / horizon_wall:.0%} "
                "of the forecast horizon."
            )
        return w

    def describe(self) -> str:
        h = self.horizon_trace_min * self.trace_minute_wall_s
        return "\n".join([
            f"  compression            : {self.compression:g}x  "
            f"(1 trace-minute = {self.trace_minute_wall_s:g} wall-seconds)",
            f"  control period         : {self.control_period_trace_min:g} trace-min "
            f"= {self.control_period_wall_s:g} s wall",
            f"  forecast horizon       : {self.horizon_trace_min} trace-min = {h:g} s wall",
            f"  look-back              : {self.look_back_trace_min} trace-min "
            f"= {self.look_back_wall_s:g} s wall",
            f"  replica bounds         : [{self.min_replicas}, {self.max_replicas}]",
            f"  per-pod capacity       : {self.per_pod_capacity_rps:g} req/s wall "
            f"= {self.per_pod_capacity_rpm_trace:g} req/trace-min",
            f"  scale-up cooldown      : {self.scale_up_cooldown_wall_s:g} s wall "
            f"(compressed {self.scale_up_cooldown_trace_min * self.trace_minute_wall_s:g} s "
            f"vs physical {self.pod_cold_start_wall_s:g} s)",
            f"  scale-down cooldown    : {self.scale_down_cooldown_wall_s:g} s wall, "
            f"{self.scale_down_confirmations} confirmations",
            f"  dead-band              : {self.deadband_replicas} replica(s)",
            f"  warm-up policy         : {self.warmup_policy}",
            f"  forecaster (arm)       : {self.forecaster}",
        ])

    def to_dict(self) -> dict:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        d.update(
            trace_minute_wall_s=self.trace_minute_wall_s,
            control_period_wall_s=self.control_period_wall_s,
            scale_up_cooldown_wall_s=self.scale_up_cooldown_wall_s,
            scale_down_cooldown_wall_s=self.scale_down_cooldown_wall_s,
            per_pod_capacity_rpm_trace=self.per_pod_capacity_rpm_trace,
        )
        return d


if __name__ == "__main__":
    c = Config()
    print("Phase 5 controller configuration\n")
    print(c.describe())
    for line in c.warnings():
        print(f"\n  WARNING: {line}")

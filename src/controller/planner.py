"""Phase 5 - Plan: turn a forecast into a replica decision, safely.

The formula is one line (Dang-Quang & Yoo, and CLAUDE.md §5):

    desired = ceil(predicted_rate / per_pod_capacity)   clamped to [min, max]

Everything else in this file is the safety layer, and the safety layer is most of the
engineering. A forecaster with R^2 = 0.71 is right on average and wrong constantly; the
planner's job is to make being occasionally wrong cheap.

--------------------------------------------------------------------------------
The asymmetry, and why it is not just conservatism
--------------------------------------------------------------------------------
Scaling up wrongly costs money for a few minutes. Scaling down wrongly costs SLO
violations, AND the pods removed take real wall-seconds to come back - time that does
not compress (D-010). So the two directions get different rules:

    up    : act immediately, one cooldown, no confirmation
    down  : require the lower figure to persist for N consecutive decisions

This is deliberately NOT symmetric hysteresis. Symmetric damping would delay scale-up
too, which directly destroys the head start the forecast exists to provide.

--------------------------------------------------------------------------------
The paper's Equations 8 and 9
--------------------------------------------------------------------------------
Both are printed identically in the published text as `if (pods_{t+1} < pods_t)`, one
labelled SCALE UP and one SCALE DOWN (CLAUDE.md §4 item 1, D-001). Taken literally the
service can only ever shrink. The corrected comparison is implemented in `_direction`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from enum import Enum


class Action(str, Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HOLD_DEADBAND = "hold_deadband"
    HOLD_COOLDOWN = "hold_cooldown"
    HOLD_CONFIRMING = "hold_confirming"
    HOLD_NO_CHANGE = "hold_no_change"
    HOLD_WARMUP = "hold_warmup"


@dataclass
class Decision:
    desired_raw: int          # ceil(rate / capacity), before any safety rule
    desired_clamped: int      # after [min, max]
    target: int               # what we will actually ask for (== current if holding)
    current: int
    action: Action
    reason: str
    predicted_rate: float     # requests per trace-minute
    seconds_since_last_scale: float | None
    pending_down_count: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d


class Planner:
    """Stateful across cycles: cooldowns and scale-down confirmations need memory."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._last_scale_at: float | None = None
        self._last_direction: Action | None = None
        self._pending_down: int = 0
        self._pending_down_target: int | None = None

    # -- the corrected Equations 8 / 9 --------------------------------------------

    @staticmethod
    def _direction(desired: int, current: int) -> Action:
        """Which way to move.

        The paper prints BOTH Equation 8 (scale up) and Equation 9 (scale down) as
        `if (pods_{t+1} < pods_t)`. That is a typo: implemented literally, the up branch
        fires only when the desired count is SMALLER, so the deployment can never grow
        and the predictive autoscaler silently becomes a one-way scale-down. We implement
        the evident intent. See DEVIATIONS.md D-001.
        """
        if desired > current:          # Eq. 8, corrected from `<` to `>`
            return Action.SCALE_UP
        if desired < current:          # Eq. 9, as printed
            return Action.SCALE_DOWN
        return Action.HOLD_NO_CHANGE

    # -- the loop's entry point ----------------------------------------------------

    def plan(self, predicted_rate: float, current: int, now: float,
             warming_up: bool = False) -> Decision:
        cfg = self.cfg
        since = None if self._last_scale_at is None else now - self._last_scale_at

        def mk(target: int, action: Action, reason: str, raw: int,
               clamped: int) -> Decision:
            return Decision(
                desired_raw=raw, desired_clamped=clamped, target=target,
                current=current, action=action, reason=reason,
                predicted_rate=round(predicted_rate, 3),
                seconds_since_last_scale=None if since is None else round(since, 2),
                pending_down_count=self._pending_down,
            )

        if warming_up:
            # Never scale on a partial window. A padded window is not a shorter history,
            # it is a wrong one, and the network cannot tell the difference.
            return mk(current, Action.HOLD_WARMUP,
                      "history window not yet full", current, current)

        # ceil() because a fraction of a pod cannot exist: 2.1 pods' worth of load
        # needs 3. Capacity is expressed in the forecaster's units (requests per
        # trace-minute) so the compressed/wall conversion happens once, in config.
        raw = math.ceil(predicted_rate / cfg.per_pod_capacity_rpm_trace)
        clamped = max(cfg.min_replicas, min(cfg.max_replicas, raw))

        direction = self._direction(clamped, current)

        if direction is Action.HOLD_NO_CHANGE:
            self._reset_pending()
            return mk(current, direction, "desired equals current", raw, clamped)

        # Dead-band, on the DOWN direction only. Forecast RMSE is 13.4 req/min (D-009)
        # = 0.69 pods at our capacity, so a one-pod difference is inside the noise of
        # the forecaster itself and not worth acting on when acting means giving up
        # capacity. Upward, the same one-pod difference IS worth acting on: the cost of
        # being wrong is one pod for a few minutes, and the cost of waiting is the head
        # start this entire architecture exists to produce.
        if direction is Action.SCALE_DOWN and abs(clamped - current) < cfg.deadband_replicas:
            self._reset_pending()
            return mk(current, Action.HOLD_DEADBAND,
                      f"scale-down of {current - clamped} < dead-band "
                      f"{cfg.deadband_replicas}", raw, clamped)

        if direction is Action.SCALE_UP:
            self._reset_pending()
            cd = cfg.scale_up_cooldown_wall_s
            if since is not None and since < cd:
                # The physical floor in this cooldown is what prevents integral windup:
                # acting again before the previous pods are Ready means deciding on a
                # cluster that has not yet responded. See config.scale_up_cooldown_wall_s.
                return mk(current, Action.HOLD_COOLDOWN,
                          f"scale-up cooldown {since:.1f}s < {cd:.1f}s", raw, clamped)
            self._mark(now, Action.SCALE_UP)
            return mk(clamped, Action.SCALE_UP,
                      f"forecast {predicted_rate:.1f} req/trace-min needs {clamped}",
                      raw, clamped)

        # -- scale down ------------------------------------------------------------
        cd = cfg.scale_down_cooldown_wall_s
        if since is not None and since < cd:
            return mk(current, Action.HOLD_COOLDOWN,
                      f"scale-down cooldown {since:.1f}s < {cd:.1f}s", raw, clamped)

        # Damping. Require the same-or-lower target repeatedly before shrinking, so a
        # transient dropout does not cost a scale-down and an immediate scale-back-up.
        # The quiet segment's 9-minute run of exact zeros is precisely this case.
        if self._pending_down_target is None or clamped > self._pending_down_target:
            self._pending_down_target = clamped
            self._pending_down = 1
        else:
            self._pending_down_target = min(self._pending_down_target, clamped)
            self._pending_down += 1

        if self._pending_down < cfg.scale_down_confirmations:
            return mk(current, Action.HOLD_CONFIRMING,
                      f"scale-down confirmation {self._pending_down}/"
                      f"{cfg.scale_down_confirmations}", raw, clamped)

        target = self._pending_down_target
        self._reset_pending()
        self._mark(now, Action.SCALE_DOWN)
        return mk(target, Action.SCALE_DOWN,
                  f"confirmed {cfg.scale_down_confirmations}x, shrinking to {target}",
                  raw, clamped)

    # -- state ---------------------------------------------------------------------

    def _mark(self, now: float, direction: Action) -> None:
        self._last_scale_at = now
        self._last_direction = direction

    def _reset_pending(self) -> None:
        self._pending_down = 0
        self._pending_down_target = None

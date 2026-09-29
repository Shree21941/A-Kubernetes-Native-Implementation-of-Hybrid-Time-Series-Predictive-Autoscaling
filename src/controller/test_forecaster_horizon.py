"""Unit checks for readiness-horizon recursive forecasting (no TensorFlow needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from forecaster import Forecast, Forecaster  # noqa: E402


def main() -> int:
    forecaster = Forecaster.__new__(Forecaster)
    seen = []

    def fake_predict(window, window_ts, target_ts):
        seen.append((window.copy(), window_ts.copy(), target_ts))
        value = float(len(seen))
        return Forecast(value, value, 0.0, 1.0, 0.5, 1.5, False)

    forecaster.predict = fake_predict
    initial_window = np.array([10.0, 20.0, 30.0])
    initial_ts = np.array(["2026-01-01T00:00", "2026-01-01T00:01",
                           "2026-01-01T00:02"], dtype="datetime64[m]")
    result = forecaster.predict_horizon(
        initial_window, initial_ts, np.datetime64("2026-01-01T00:03"), 3)

    assert result.value == 3.0
    assert result.net_ms == 3.0 and result.prophet_ms == 1.5
    assert len(seen) == 3
    assert seen[1][0].tolist() == [20.0, 30.0, 1.0]
    assert seen[2][0].tolist() == [30.0, 1.0, 2.0]
    assert seen[2][2] == np.datetime64("2026-01-01T00:05")

    try:
        forecaster.predict_horizon(initial_window, initial_ts,
                                   np.datetime64("2026-01-01T00:03"), 0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-step horizon must be rejected")

    print("PASS recursive horizon and accumulated inference timings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
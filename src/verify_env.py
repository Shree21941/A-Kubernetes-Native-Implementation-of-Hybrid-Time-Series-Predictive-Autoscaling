"""Phase 0 Definition-of-Done check.

Run this after any environment change to confirm the project's foundations still hold.
It is deliberately blunt: it either prints a table of green checks and exits 0, or it
tells you exactly which assumption broke and exits 1.

    ./.venv/bin/python src/verify_env.py

Why a script and not a shell one-liner: this gets re-run at the start of Phases 1, 2 and
5, and after any dependency change. An environment that silently drifts (a numpy
downgrade, a Prophet that imports but cannot fit) is the kind of failure that shows up
as a bad *result* three phases later rather than as an error here.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The two raw traces and their documented line counts from the Internet Traffic
# Archive. Checking the count - not merely that the file exists - is what caught a
# corrupt July download in Phase 0 (see DEVIATIONS.md D-E02).
EXPECTED_LOG_LINES = {
    "NASA_access_log_Jul95.gz": 1_891_714,
    "NASA_access_log_Aug95.gz": 1_569_898,
}

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))


# --- 1. Interpreter -------------------------------------------------------
# Prophet's Stan toolchain is unreliable on 3.12+, so this is a hard requirement,
# not a preference (CLAUDE.md 3).
v = sys.version_info
check(
    "Python is 3.11",
    (v.major, v.minor) == (3, 11),
    f"{v.major}.{v.minor}.{v.micro}",
)

# --- 2. Prophet imports ---------------------------------------------------
try:
    from prophet import Prophet

    import prophet

    check("Prophet imports", True, f"prophet {prophet.__version__}")
except Exception as e:  # noqa: BLE001 - we want any failure reported, not raised
    check("Prophet imports", False, f"{type(e).__name__}: {e}")
    Prophet = None

# --- 3. TensorFlow imports ------------------------------------------------
try:
    import tensorflow as tf

    check("TensorFlow imports", True, f"tensorflow {tf.__version__}")
except Exception as e:  # noqa: BLE001
    check("TensorFlow imports", False, f"{type(e).__name__}: {e}")
    tf = None

# --- 4. The rest of the stack --------------------------------------------
for mod, label in [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("statsmodels", "statsmodels"),     # ARIMA baseline (Phase 2)
    ("matplotlib", "matplotlib"),
    ("sklearn", "scikit-learn"),        # metrics + scalers (Phase 2)
    ("requests", "requests"),
    ("kubernetes", "kubernetes"),       # Execute stage (Phase 5)
    ("prometheus_api_client", "prometheus-api-client"),  # Monitor stage (Phase 5)
]:
    try:
        m = __import__(mod)
        check(f"{label} imports", True, getattr(m, "__version__", "ok"))
    except Exception as e:  # noqa: BLE001
        check(f"{label} imports", False, f"{type(e).__name__}: {e}")

# --- 5. Prophet can actually FIT -----------------------------------------
# Importing Prophet only proves the Python package is present. Fitting is what
# invokes the compiled Stan binary, which is the part that actually breaks.
if Prophet is not None:
    try:
        import logging

        import numpy as np
        import pandas as pd

        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

        n = 120
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2024-01-01", periods=n, freq="h"),
                "y": 10 + np.sin(np.arange(n) * 2 * np.pi / 24) * 5,
            }
        )
        m = Prophet(daily_seasonality=True).fit(df)
        yhat = m.predict(m.make_future_dataframe(periods=3, freq="h"))["yhat"].iloc[-1]
        check("Prophet Stan fit runs", True, f"forecast={yhat:.3f}")
    except Exception as e:  # noqa: BLE001
        check("Prophet Stan fit runs", False, f"{type(e).__name__}: {e}")

# --- 6. Keras can build and train a tiny LSTM ----------------------------
# Same logic: importing TF proves little. Phase 2 needs LSTM/GRU layers to
# actually compile and run a gradient step.
if tf is not None:
    try:
        import numpy as np

        tf.keras.utils.set_random_seed(42)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(10, 1)),
                tf.keras.layers.LSTM(4),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        x = np.random.rand(16, 10, 1).astype("float32")
        y = np.random.rand(16, 1).astype("float32")
        h = model.fit(x, y, epochs=1, verbose=0)
        check("Keras LSTM trains", True, f"loss={h.history['loss'][0]:.4f}")
    except Exception as e:  # noqa: BLE001
        check("Keras LSTM trains", False, f"{type(e).__name__}: {e}")

# --- 7. Raw data present and intact --------------------------------------
for fname, expected in EXPECTED_LOG_LINES.items():
    p = REPO / "data" / "raw" / fname
    if not p.exists():
        check(f"{fname}", False, "MISSING")
        continue
    try:
        # Counting lines also validates the gzip CRC end-to-end: a corrupt member
        # raises here rather than silently yielding a short file.
        #
        # Count newline BYTES, not iterator yields, to match `wc -l` - which is the
        # convention the Internet Traffic Archive's documented counts use. It matters
        # here: the July trace's final record ("alyssa.p") is truncated with no
        # trailing newline, so iterating lines reports 1,891,715 while wc -l reports
        # 1,891,714. Both describe the same intact file; only the convention differs.
        with gzip.open(p, "rb") as fh:
            lines = 0
            while chunk := fh.read(8 << 20):
                lines += chunk.count(b"\n")
        ok = lines == expected
        check(
            f"{fname}",
            ok,
            f"{lines:,} lines" + ("" if ok else f" (expected {expected:,})"),
        )
    except Exception as e:  # noqa: BLE001
        check(f"{fname}", False, f"{type(e).__name__}: {e}")

# --- 8. Directory skeleton ------------------------------------------------
required_dirs = [
    "data/raw",
    "data/processed",
    "src/data",
    "src/models",
    "src/eval",
    "src/controller",
    "k8s",
    "load",
    "experiments",
    "results",
]
missing = [d for d in required_dirs if not (REPO / d).is_dir()]
check("Repo skeleton", not missing, "all present" if not missing else f"missing: {missing}")

# --- Report ---------------------------------------------------------------
width = max(len(n) for n, _, _ in results)
print()
print("Phase 0 - environment verification")
print("=" * (width + 46))
for name, ok, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<{width}}  {detail}")
print("=" * (width + 46))

failed = [n for n, ok, _ in results if not ok]
if failed:
    print(f"\n{len(failed)} CHECK(S) FAILED: {', '.join(failed)}")
    sys.exit(1)
print(f"\nAll {len(results)} checks passed.")
sys.exit(0)

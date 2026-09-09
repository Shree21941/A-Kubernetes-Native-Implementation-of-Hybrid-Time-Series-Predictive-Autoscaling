"""Phase 2 - the recurrent networks, one configuration parameterised three ways.

The paper reuses a single architecture across LSTM, Bi-LSTM and GRU, so this module
does too: swapping the cell type is the only difference between them, which keeps the
comparison honest. Any accuracy gap between the three is attributable to the cell and
not to someone having tuned one of them harder.

Architecture and hyperparameters, from the paper:

    2 recurrent layers x 50 units  ->  Dense(1)
    loss MSE, Adam lr=0.001, batch 16, up to 50 epochs, early stopping patience 5

The same class serves both roles in this phase:

  * hybrid stage - trained on Prophet's RESIDUALS. This is the paper's actual method,
    and training on raw `y` here is the single most common way to accidentally not
    implement it.
  * single-model baseline - trained on raw `y`, which is what the Bi-LSTM 1-step and
    5-step rows of Table 4 are.

Nothing in the class knows which of those it is doing; the caller decides by choosing
what series to hand it. That is deliberate - it means the hybrid and the baseline go
through identical code, so a difference in their scores is a difference in the method.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np

# Quieten TF's C++ logging before it is imported, or every run buries its own output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402

from dataset import Supervised  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"

SEED = 42

# Paper hyperparameters. Named constants rather than call-site literals so that the
# claim "all three cells use one configuration" is checkable by reading this block.
UNITS = 50
N_LAYERS = 2
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
MAX_EPOCHS = 50
PATIENCE = 5
VAL_FRACTION = 0.1

CELLS = ("lstm", "bilstm", "gru")


def set_seeds(seed: int = SEED) -> None:
    """Pin every RNG Keras might consult.

    Note this does NOT make training bit-for-bit reproducible: TensorFlow's CPU kernels
    parallelise reductions, and floating-point addition is not associative, so run-to-run
    results differ in the last few digits. `tf.config.experimental.enable_op_determinism()`
    would close that gap at a large speed cost. We pin seeds, record them, and accept
    third-decimal drift - documented in DEVIATIONS.md D-006.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build(cell: str, look_back: int, units: int = UNITS) -> tf.keras.Model:
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {CELLS}, got {cell!r}")

    layers = [tf.keras.layers.Input(shape=(look_back, 1))]
    for i in range(N_LAYERS):
        last = i == N_LAYERS - 1
        # Only the final recurrent layer collapses the sequence; the earlier one must
        # hand a full sequence forward or the second layer has nothing to recur over.
        if cell == "lstm":
            layers.append(tf.keras.layers.LSTM(units, return_sequences=not last))
        elif cell == "gru":
            layers.append(tf.keras.layers.GRU(units, return_sequences=not last))
        else:
            # Bidirectional reads the window backwards as well as forwards. That is not
            # leakage: every timestep in the window is strictly earlier than the target,
            # so the "future" it sees is still the target's past.
            layers.append(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(units, return_sequences=not last)
                )
            )
    layers.append(tf.keras.layers.Dense(1))

    model = tf.keras.Sequential(layers, name=f"{cell}_{units}x{N_LAYERS}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def train(
    cell: str,
    data: Supervised,
    tag: str,
    epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    outdir: Path = MODELS,
    verbose: int = 2,
    shuffle: bool = False,
    patience: int = PATIENCE,
) -> tuple[tf.keras.Model, dict]:
    """Train one network and persist it. Returns (model, history-and-timing dict)."""
    set_seeds()
    model = build(cell, data.look_back)

    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,  # keep the best epoch, not the last one
        verbose=1,
    )

    t0 = time.perf_counter()
    hist = model.fit(
        data.x_train, data.y_train,
        validation_split=VAL_FRACTION,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[stop],
        # Keras carves the validation set off the END of the arrays BEFORE shuffling,
        # so the validation window is always the most recent slice of training - a
        # time-ordered holdout, whatever `shuffle` is set to.
        #
        # `shuffle` then controls only the order in which training WINDOWS are drawn
        # into minibatches. That is not a time-series shuffle: each window is a
        # self-contained (60 past minutes -> next minute) pair, so every input it holds
        # is strictly earlier than its own target no matter when it is presented.
        # CLAUDE.md 8's rule targets the split, which stays time-ordered either way.
        # Defaulting to False keeps the paper-faithful run maximally conservative; see
        # DEVIATIONS.md D-008 for the measured effect of turning it on.
        shuffle=shuffle,
        verbose=verbose,
    )
    train_s = time.perf_counter() - t0

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{tag}.keras"
    model.save(path)

    meta = dict(
        tag=tag, cell=cell, path=str(path.relative_to(REPO)),
        look_back=data.look_back, horizon=data.horizon,
        epochs_run=len(hist.history["loss"]),
        best_val_loss=float(min(hist.history["val_loss"])),
        final_train_loss=float(hist.history["loss"][-1]),
        train_seconds=round(train_s, 1),
        shuffle=shuffle,
        patience=patience,
        n_train_windows=int(len(data.x_train)),
        n_params=int(model.count_params()),
        seed=SEED,
    )
    return model, meta


def predict_timed(
    model: tf.keras.Model, data: Supervised, batch_size: int = 512
) -> tuple[np.ndarray, float]:
    """Predict the test set and return (unscaled predictions, elapsed ms).

    The warm-up call matters: the first `predict` on a Keras 3 model pays for tracing
    and graph construction, which can dominate a short measurement and would make the
    TPT column measure compilation rather than inference.
    """
    model.predict(data.x_test[:batch_size], batch_size=batch_size, verbose=0)

    t0 = time.perf_counter()
    scaled = model.predict(data.x_test, batch_size=batch_size, verbose=0)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return data.inverse(scaled), elapsed_ms


if __name__ == "__main__":
    # Architecture summary only - training is driven by src/eval/compare.py.
    for c in CELLS:
        m = build(c, 60)
        print(f"{c:<7} {m.count_params():>8,} params   "
              f"output {m.output_shape}")

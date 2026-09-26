"""
Prediction models: forward LSTM, Bi-LSTM and ARIMA.

Bi-LSTM / LSTM configuration follows Table 2 of the paper:
    layers .............. 1 (Bi)LSTM layer (forward + backward for Bi-LSTM)
    input size .......... 10 time steps
    hidden units ........ 30 per LSTM cell
    loss ................ MSE
    batch size .......... 64
    epochs .............. 50
    early stopping ...... patience 5 (NASA) / 2 (FIFA)
    activation .......... ReLU (hidden layer)
    output .............. Dense(horizon)   (1 = single-step, 5 = multi-step)

ARIMA follows Section 6.2: order (p, d, q) = (4, 1, 0), walk-forward one step.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Neural models (Keras / TensorFlow)
# ----------------------------------------------------------------------------
def _compile(model):
    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(), loss="mse")
    return model


def build_lstm(window: int = 10, units: int = 30, horizon: int = 1):
    """Forward (uni-directional) LSTM -> Dense(horizon)."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(units, activation="relu"),
        Dense(horizon),
    ])
    return _compile(model)


def build_bilstm(window: int = 10, units: int = 30, horizon: int = 1):
    """Bidirectional LSTM -> Dense(horizon)."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Input
    model = Sequential([
        Input(shape=(window, 1)),
        Bidirectional(LSTM(units, activation="relu")),
        Dense(horizon),
    ])
    return _compile(model)


def train_nn(model, X_train, y_train, patience: int = 5,
             epochs: int = 50, batch_size: int = 64, verbose: int = 0):
    """Train with MSE loss and early stopping on a 10% validation split
    (restores best weights). Matches the paper's 'stop when validation loss
    does not decrease any further'."""
    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor="val_loss", patience=patience,
                       restore_best_weights=True)
    hist = model.fit(
        X_train, y_train,
        validation_split=0.10,
        epochs=epochs, batch_size=batch_size,
        callbacks=[es], verbose=verbose, shuffle=True,
    )
    return hist


# ----------------------------------------------------------------------------
# ARIMA (statsmodels)
# ----------------------------------------------------------------------------
def fit_arima(train_values: np.ndarray, order=(4, 1, 0)):
    """Fit ARIMA once (MLE) on the training series and return the results."""
    from statsmodels.tsa.arima.model import ARIMA
    return ARIMA(np.asarray(train_values, dtype="float64"), order=order).fit()


def arima_walk_forward(train_values: np.ndarray, test_values: np.ndarray,
                       order=(4, 1, 0), horizon: int = 1, res=None):
    """Walk-forward ARIMA(p, 1, 0) evaluated over the test set.

    The model is fit once (MLE) on the training series. Forecasts are then
    produced by the exact ARIMA(p,1,0) recursion using the fitted AR
    coefficients: a pure AR(p) process on the first-differenced series. This
    gives identical forecasts to refitting/appending at every step, but in
    O(n) instead of O(n^2) (verified against statsmodels `forecast`).

        d y_t = c + sum_k phi_k * d y_{t-k} + e_t ,   d y_t = y_t - y_{t-1}

    Returns
    -------
    preds : (len(test) - horizon + 1, horizon) array. preds[j] is the forecast
            of test[j : j+horizon] using all actual history up to test[j-1]
            (history reaches into the training series for small j). The caller
            slices off the first `window` rows to align with the NN windows.
    """
    p, d, q = order
    assert d == 1 and q == 0, "fast path implements ARIMA(p, 1, 0)"

    train = np.asarray(train_values, dtype="float64")
    test = np.asarray(test_values, dtype="float64")

    if res is None:
        res = fit_arima(train, order)
    phi = np.asarray(res.arparams, dtype="float64")          # (p,)
    const = float(res.params[res.param_names.index("const")]) \
        if "const" in res.param_names else 0.0

    y = np.concatenate([train, test])
    dy = np.diff(y)                       # dy[t] = y[t+1]-y[t]; len = len(y)-1
    n_train = len(train)
    n_test = len(test)

    preds = np.empty((n_test - horizon + 1, horizon), dtype="float64")
    for j in range(n_test - horizon + 1):
        g = n_train + j                   # global index of test[j]
        y_last = y[g - 1]                 # last known actual = test[j-1]
        # last p KNOWN diffs, most recent first. dy[t] = y[t+1]-y[t], so the
        # diff ending at y[g-1] is dy[g-2], then dy[g-3], ...
        buf = [dy[g - 2 - k] for k in range(p)]
        yhat = y_last
        for h in range(horizon):
            dnext = const + float(np.dot(phi, buf))
            yhat = yhat + dnext
            preds[j, h] = yhat
            buf = [dnext] + buf[:-1]      # roll predicted diff in
    return preds

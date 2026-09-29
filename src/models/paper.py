"""Paper reproduction: LSTM, Bi-LSTM and ARIMA predictors."""

from __future__ import annotations

import numpy as np


def _compile(model):
    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(), loss="mse")
    return model


def build_lstm(window: int = 10, units: int = 30, horizon: int = 1):
    """Build forward LSTM followed by Dense(horizon)."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(units, activation="relu"),
        Dense(horizon),
    ])
    return _compile(model)


def build_bilstm(window: int = 10, units: int = 30, horizon: int = 1):
    """Build bidirectional LSTM followed by Dense(horizon)."""
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
    """Train using MSE and early stopping on a 10% validation split."""
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True)
    return model.fit(
        X_train, y_train,
        validation_split=0.10,
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stopping], verbose=verbose, shuffle=True,
    )


def fit_arima(train_values: np.ndarray, order=(4, 1, 0)):
    """Fit ARIMA once by maximum likelihood."""
    from statsmodels.tsa.arima.model import ARIMA
    return ARIMA(np.asarray(train_values, dtype="float64"), order=order).fit()


def arima_walk_forward(train_values: np.ndarray, test_values: np.ndarray,
                       order=(4, 1, 0), horizon: int = 1, res=None):
    """Walk-forward ARIMA(p,1,0) using the fitted first-difference recursion."""
    p, d, q = order
    assert d == 1 and q == 0, "fast path implements ARIMA(p, 1, 0)"

    train = np.asarray(train_values, dtype="float64")
    test = np.asarray(test_values, dtype="float64")
    if res is None:
        res = fit_arima(train, order)
    phi = np.asarray(res.arparams, dtype="float64")
    const = float(res.params[res.param_names.index("const")]) \
        if "const" in res.param_names else 0.0

    values = np.concatenate([train, test])
    differences = np.diff(values)
    n_train, n_test = len(train), len(test)
    predictions = np.empty((n_test - horizon + 1, horizon), dtype="float64")
    for offset in range(n_test - horizon + 1):
        global_index = n_train + offset
        last_value = values[global_index - 1]
        history = [differences[global_index - 2 - k] for k in range(p)]
        forecast = last_value
        for step in range(horizon):
            next_difference = const + float(np.dot(phi, history))
            forecast += next_difference
            predictions[offset, step] = forecast
            history = [next_difference] + history[:-1]
    return predictions
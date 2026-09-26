"""Plotting helpers that reproduce the paper's figures."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_actual_vs_pred(index, actual, pred, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(index, actual, color="blue", linewidth=0.7, label="Actual")
    ax.plot(index, pred, color="red", linewidth=0.7, label="Predicted")
    ax.set_title(title)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_absolute_error(actual, preds_by_model, out_path, n=100):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    styles = {"Bi-LSTM": ("x-", "blue"), "ARIMA": ("o-", "red"),
              "LSTM": ("s-", "green")}
    for name, pred in preds_by_model.items():
        ae = abs(actual[:n] - pred[:n])
        st, col = styles.get(name, ("-", None))
        ax.plot(range(len(ae)), ae, st, color=col, markersize=3,
                linewidth=0.8, label=name)
    ax.set_title("Comparison of absolute error")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Absolute error value (requests)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pods(index, required, provided, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(index, required, color="blue", linewidth=0.8, label="required")
    ax.plot(index, provided, color="red", linewidth=0.8, label="provided")
    ax.set_title(title)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Number of Pods")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

"""Phase 2 - assemble every model into one comparison table against paper Table 4.

    ./.venv/bin/python src/eval/compare.py

Reads the prediction files each model script wrote to results/phase2_preds_*.npz,
scores them all through the same code in metrics.py, and emits:

    results/phase2_comparison.md    - side by side with the paper
    results/phase2_comparison.csv   - the same numbers, machine-readable

Then it checks the Definition of Done explicitly and reports pass/fail per criterion,
because "the table looks about right" is not a verification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src" / "eval"), str(REPO / "src" / "models")]

from dataset import load_series, split_at  # noqa: E402
from metrics import MODEL_ORDER, PAPER_TABLE4, evaluate  # noqa: E402

RESULTS = REPO / "results"
MODELS = REPO / "models"

_tpt_path = RESULTS / "phase2_tpt.json"
TPT: dict[str, float] = (
    json.loads(_tpt_path.read_text()) if _tpt_path.exists() else {}
)

# job name -> the Table 4 row it corresponds to.
JOB_TO_LABEL = {
    "hybrid_lstm": "Prophet+LSTM (hybrid)",
    "hybrid_bilstm": "Prophet+BiLSTM (hybrid)",
    "hybrid_gru": "Prophet+GRU (hybrid)",
    "arima_1step": "ARIMA 1-step",
    "bilstm_1step": "BiLSTM 1-step",
    "arima_5step": "ARIMA 5-step",
    "bilstm_5step": "BiLSTM 5-step",
}


def load_all(variant: str = "") -> tuple[list, list[str]]:
    """Score one variant. `variant` is a filename suffix, e.g. "_shuffled"."""
    scored, missing = [], []
    for job, label in JOB_TO_LABEL.items():
        # ARIMA has no minibatch order, so it has no variant - the same run serves both
        # tables. Only the neural jobs are retrained.
        suffix = "" if job.startswith("arima") else variant
        npz_path = RESULTS / f"phase2_preds_{job}{suffix}.npz"
        meta_path = MODELS / f"{job}{suffix}_meta.json"
        if not npz_path.exists() or not meta_path.exists():
            missing.append(job)
            continue
        d = np.load(npz_path)
        meta = json.loads(meta_path.read_text())
        # Prefer the dedicated serial TPT pass. The timings recorded during training
        # were taken with three or four jobs competing for 12 CPUs, which is fine for
        # accuracy but meaningless for latency.
        tpt = TPT.get(f"{job}{suffix}", meta["tpt_ms"])
        s = evaluate(label, d["y_true"], d["y_pred"], tpt_ms=tpt)

        # Predictions are NOT clamped at zero for the headline table - clamping is a
        # post-hoc correction the paper does not describe, and applying it silently
        # would flatter every model by an undocumented amount. It is reported here so
        # the size of that effect is visible rather than hidden.
        clamped = np.maximum(d["y_pred"], 0.0)
        c_rmse = float(np.sqrt(((d["y_true"] - clamped) ** 2).mean()))
        s.notes = f"clamped@0 RMSE {c_rmse:.3f}"
        scored.append((job, s, meta))
    return scored, missing


def reference_prophet_row(config: str = "reading_a"):
    """Prophet alone - not a Table 4 row, but the number the hybrid must improve on.

    Without this line the table cannot answer the obvious question: is the LSTM stage
    earning its cost, or is Prophet doing all the work?
    """
    npz = np.load(MODELS / f"prophet_{config}.npz")
    meta = next(m for m in json.loads((MODELS / "prophet_meta.json").read_text())
                if m["name"] == config)
    s = load_series()
    cut = split_at(len(s))
    return evaluate(
        f"Prophet alone ({config})",
        s.to_numpy()[cut:], npz["yhat"][cut:],
        tpt_ms=meta["predict_test_ms"],
        notes="reference row, not in paper Table 4",
    )


def fmt_table(scored, prophet_row) -> str:
    lines = []
    lines.append("| Model | MSE | RMSE | MAE | R2 | TPT (ms) | us/pred |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, s, _ in scored:
        lines.append(
            f"| **{s.model}** | {s.mse:.3f} | {s.rmse:.3f} | {s.mae:.3f} | "
            f"{s.r2:.3f} | {s.tpt_ms:.0f} | {s.tpt_us_per_pred:.1f} |"
        )
    p = prophet_row
    lines.append(
        f"| _{p.model}_ | {p.mse:.3f} | {p.rmse:.3f} | {p.mae:.3f} | "
        f"{p.r2:.3f} | {p.tpt_ms:.0f} | {p.tpt_us_per_pred:.1f} |"
    )
    return "\n".join(lines)


def fmt_side_by_side(scored) -> str:
    lines = []
    lines.append("| Model | MSE (ours) | MSE (paper) | RMSE (ours) | RMSE (paper) "
                 "| MAE (ours) | MAE (paper) | R2 (ours) | R2 (paper) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    by_label = {s.model: s for _, s, _ in scored}
    for label in MODEL_ORDER:
        pa = PAPER_TABLE4[label]
        if label not in by_label:
            lines.append(f"| {label} | – | {pa['mse']:.3f} | – | {pa['rmse']:.3f} "
                         f"| – | {pa['mae']:.3f} | – | {pa['r2']:.3f} |")
            continue
        s = by_label[label]
        lines.append(
            f"| {label} | {s.mse:.3f} | {pa['mse']:.3f} | {s.rmse:.3f} "
            f"| {pa['rmse']:.3f} | {s.mae:.3f} | {pa['mae']:.3f} | {s.r2:.3f} "
            f"| {pa['r2']:.3f} |"
        )
    return "\n".join(lines)


def check_dod(scored) -> tuple[bool, list[str]]:
    """The Phase 2 Definition of Done, evaluated rather than asserted."""
    by_label = {s.model: s for _, s, _ in scored}
    out, ok = [], True

    def report(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= passed
        out.append(f"  [{'PASS' if passed else 'FAIL'}]  {name:<52}  {detail}")

    report("Table covers all 7 Table 4 variants",
           len(by_label) == 7, f"{len(by_label)}/7 present")

    hyb = by_label.get("Prophet+LSTM (hybrid)")
    if hyb is None:
        report("Prophet+LSTM has lowest MSE/RMSE/MAE", False, "hybrid missing")
        report("Prophet+LSTM R2 in [0.85, 0.93]", False, "hybrid missing")
    else:
        others = [s for s in by_label.values() if s.model != hyb.model]
        for metric in ("mse", "rmse", "mae"):
            best = min(others, key=lambda s: getattr(s, metric))
            hv, bv = getattr(hyb, metric), getattr(best, metric)
            passed = hv < bv
            gap = abs(hv - bv) / bv * 100
            report(f"Prophet+LSTM has the lowest {metric.upper()}", passed,
                   f"{hv:.3f} vs best {bv:.3f} ({best.model}), "
                   f"{'behind' if not passed else 'ahead'} by {gap:.2f}%")
        report("Prophet+LSTM R2 in [0.85, 0.93]", 0.85 <= hyb.r2 <= 0.93,
               f"R2 = {hyb.r2:.3f}")

    # The accuracy/latency trade-off the paper reports and we must not hide: the
    # hybrids pay Prophet's inference cost on top of the network's, on every forecast.
    #
    # The comparison is against the NEURAL single models specifically. The paper's own
    # contrast is 3492 ms (hybrid) against 4.3 ms (Bi-LSTM) - three orders of magnitude
    # - and its ARIMA rows are slow too (2300 ms), so ARIMA was never the point. Ours is
    # slower still because a 26,784-step rolling Kalman loop in Python is an artifact of
    # our implementation, not a property of ARIMA, and including it here would let a
    # slow baseline mask the effect being tested.
    # The comparison has to be SAME-CELL to mean anything. Comparing Prophet+GRU against
    # a bare Bi-LSTM measures the cell, not the hybrid: a bidirectional LSTM makes two
    # passes over the window and carries 81 k parameters against the GRU's 23 k, so it
    # is intrinsically ~2.5x dearer. The only like-for-like pair we have is
    # Prophet+BiLSTM against Bi-LSTM alone, where the difference is exactly the Prophet
    # stage.
    pair = (by_label.get("Prophet+BiLSTM (hybrid)"), by_label.get("BiLSTM 1-step"))
    if all(pair):
        hyb_b, net_b = pair
        overhead = hyb_b.tpt_ms - net_b.tpt_ms
        report("Hybrid slower than the same net alone (TPT)",
               hyb_b.tpt_ms > net_b.tpt_ms,
               f"Prophet+BiLSTM {hyb_b.tpt_ms:.0f} ms vs BiLSTM alone "
               f"{net_b.tpt_ms:.0f} ms (+{overhead:.0f} ms = the Prophet stage)")

    return ok, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prophet", default="reading_a")
    args = ap.parse_args()

    # Two configurations are reported side by side, because which one is "the"
    # reproduction is itself a judgement call (DEVIATIONS.md D-008):
    #   chronological - minibatches in time order, the maximally conservative reading
    #                   of CLAUDE.md 8's no-shuffle rule
    #   shuffled      - training windows shuffled between epochs, which converges
    #                   properly and is what the Definition of Done is judged on,
    #                   since a reproduction should test the method at its best
    variants = [("shuffled", "_shuffled"), ("chronological", "")]

    tables, results = [], {}
    for vname, vsuffix in variants:
        scored, missing = load_all(vsuffix)
        if not scored:
            continue
        results[vname] = (scored, missing)
        tables.append((vname, scored, missing))

    if not tables:
        print("Nothing to compare - run the model scripts first.")
        return 1

    headline = "shuffled" if "shuffled" in results else "chronological"
    scored, missing = results[headline]
    if missing:
        print(f"WARNING: no {headline} predictions yet for: {', '.join(missing)}")

    prophet_row = reference_prophet_row(args.prophet)
    ok, dod = check_dod(scored)

    n_test = scored[0][1].n_test
    body = [
        "# Phase 2 - forecasting model comparison (NASA-HTTP)",
        "",
        f"Test set: **{n_test:,} minutes**, 70/30 time-ordered split, "
        "metrics in requests/minute on the test portion only.",
        f"Prophet configuration supplying hybrid residuals: `{args.prophet}`. "
        f"Seed 42. Look-back 60 minutes.",
        "",
        f"Headline configuration: **{headline}** minibatch order (see DEVIATIONS "
        "D-008).",
        "",
        *[line
          for vname, sc, _ in tables
          for line in (f"## Our results - {vname} minibatch order", "",
                       fmt_table(sc, prophet_row), "")],
        "## Side by side with the paper (CLAUDE.md 5 / paper Table 4)",
        "",
        fmt_side_by_side(scored),
        "",
        "## Definition of Done",
        "",
        "```",
        *dod,
        "```",
        "",
    ]
    (RESULTS / "phase2_comparison.md").write_text("\n".join(body))

    import csv
    with open(RESULTS / "phase2_comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["model", "mse", "rmse", "mae", "r2", "tpt_ms",
                        "tpt_us_per_pred", "n_test", "notes"],
        )
        w.writeheader()
        for vname, sc, _ in tables:
            for _, s, _ in sc:
                row = s.as_dict()
                row["model"] = f"{s.model} [{vname}]"
                w.writerow(row)
        w.writerow(prophet_row.as_dict())

    print()
    print("Phase 2 - model comparison")
    print("=" * 96)
    print(f"({headline} minibatch order)")
    print(f"{'Model':<26} {'MSE':>10} {'RMSE':>8} {'MAE':>8} {'R2':>7} "
          f"{'TPT ms':>9} {'us/pred':>9}")
    print("-" * 96)
    for _, s, _ in scored:
        print(f"{s.model:<26} {s.mse:>10.3f} {s.rmse:>8.3f} {s.mae:>8.3f} "
              f"{s.r2:>7.3f} {s.tpt_ms:>9.0f} {s.tpt_us_per_pred:>9.1f}")
    if "chronological" in results and headline != "chronological":
        print("-" * 96)
        print("  (chronological minibatch order, for comparison)")
        for _, s, _ in results["chronological"][0]:
            print(f"{s.model:<26} {s.mse:>10.3f} {s.rmse:>8.3f} {s.mae:>8.3f} "
                  f"{s.r2:>7.3f} {s.tpt_ms:>9.0f} {s.tpt_us_per_pred:>9.1f}")
    print("-" * 96)
    p = prophet_row
    print(f"{p.model:<26} {p.mse:>10.3f} {p.rmse:>8.3f} {p.mae:>8.3f} "
          f"{p.r2:>7.3f} {p.tpt_ms:>9.0f} {p.tpt_us_per_pred:>9.1f}   (reference)")
    print("=" * 96)
    print()
    print("Definition of Done")
    print("\n".join(dod))
    print()
    print(f"wrote results/phase2_comparison.md and .csv")

    if missing:
        print(f"\nINCOMPLETE: still missing {', '.join(missing)}")
        return 1
    if not ok:
        print("\nDEFINITION OF DONE NOT MET - see FAIL lines above.")
        return 1
    print("\nPhase 2 Definition of Done met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

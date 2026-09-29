"""Build the evidence deck from the saved NASA result CSV and measurement JSON."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

from analyze_live_pair import score_run


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "cold_start_aware_evidence.pptx"
TABLE = Path(__file__).resolve().parent / "tables" / "table_nasa_accuracy.csv"

INK = RGBColor(24, 42, 58)
TEAL = RGBColor(0, 115, 120)
GOLD = RGBColor(194, 113, 36)
PALE = RGBColor(239, 245, 245)
WHITE = RGBColor(255, 255, 255)
MUTED = RGBColor(93, 108, 119)
RED = RGBColor(161, 55, 45)


def add_text(slide, x, y, w, h, text, size=20, color=INK, bold=False,
             font="Aptos", align=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.font.name = font
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = color
        if align is not None:
            paragraph.alignment = align
    return box


def add_base(prs, title, section):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    add_text(slide, 0.55, 0.28, 10.8, 0.28, section.upper(), 10, TEAL, True)
    add_text(slide, 0.55, 0.62, 12.1, 0.62, title, 27, INK, True)
    line = slide.shapes.add_shape(1, Inches(0.55), Inches(1.35), Inches(12.2), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = PALE
    line.line.fill.background()
    add_text(slide, 0.55, 7.12, 12.0, 0.18,
             "NASA baseline + cold-start-aware controller | 29 Sep 2026", 9, MUTED)
    return slide


def add_bullets(slide, items, x=0.8, y=1.7, w=11.5, size=20, gap=0.82):
    for idx, item in enumerate(items):
        yy = y + idx * gap
        dot = slide.shapes.add_shape(9, Inches(x), Inches(yy + 0.1), Inches(0.13), Inches(0.13))
        dot.fill.solid()
        dot.fill.fore_color.rgb = TEAL
        dot.line.fill.background()
        add_text(slide, x + 0.32, yy, w - 0.32, gap - 0.08, item, size, INK)


def load_results():
    with TABLE.open(newline="") as stream:
        return list(csv.DictReader(stream))


def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = INK
    add_text(slide, 0.8, 0.65, 11.7, 0.35, "RESEARCH PROJECT | EVIDENCE UPDATE", 12,
             RGBColor(102, 207, 197), True)
    add_text(slide, 0.8, 1.55, 11.7, 1.7,
             "Cold-Start-Aware\nKubernetes Autoscaling", 34, WHITE, True)
    add_text(slide, 0.8, 3.72, 11.2, 0.9,
             "Paper baseline reproduction and measured readiness-aware control", 22,
             RGBColor(219, 231, 235))
    add_text(slide, 0.8, 5.55, 11.2, 0.8,
             "Two-pair pilot: mean p95 down 48%; broader replication pending.",
             17, RGBColor(255, 204, 133), True)
    add_text(slide, 0.8, 7.0, 11.2, 0.2,
             "Dang-Quang & Yoo, Applied Sciences 11 (2021), 3835", 10,
             RGBColor(187, 202, 210))
    return slide


def main():
    results = load_results()
    by_key = {(r["model"], int(r["steps"])): r for r in results}
    measurements = json.loads((ROOT / "experiments" / "measured_constants.json").read_text())
    ready = measurements["pod_cold_start_p95_wall_s"]
    interval = 60.0 / 10.0
    slow = json.loads((Path(__file__).resolve().parent / "slow_start_coldstart.json").read_text())
    slow_ready = float(slow["all_ready"]["p95"])
    steps = max(1, math.ceil((slow_ready + 0.01) / interval))
    pilot_runs = [score_run(arm, rep) for arm in ("fixed_horizon", "readiness_aware")
                  for rep in (1, 2)]
    pilot = {(r["arm"], r["rep"]): r for r in pilot_runs}

    def mean(arm, metric):
        return sum(pilot[(arm, rep)][metric] for rep in (1, 2)) / 2

    def pct_change(metric):
        fixed = mean("fixed_horizon", metric)
        return 100 * (mean("readiness_aware", metric) - fixed) / fixed

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "Cold-Start-Aware Kubernetes Autoscaling: Evidence Update"
    prs.core_properties.subject = "NASA paper baseline and KIND optimization status"
    prs.core_properties.author = "Project team"

    slide_title(prs)

    slide = add_base(prs, "Research question and contribution", "01 | Problem")
    add_bullets(slide, [
        "A t+1 workload forecast only helps if additional pods become Ready before the forecasted demand arrives.",
        "The paper uses a fixed one-step forecast, 60-second cooldown, and 60% resource removal in its simulated policy.",
        "This extension selects a forecast lead from measured pod readiness, recursively using the packaged one-step model when needed.",
    ], size=19, gap=1.15)
    add_text(slide, 1.0, 5.45, 11.2, 0.75,
             "required steps = ceil((readiness p95 + controller overhead) / control interval)",
             19, TEAL, True, align=PP_ALIGN.CENTER)

    slide = add_base(prs, "NASA trace: validated before modeling", "02 | Dataset")
    add_text(slide, 0.8, 1.68, 5.4, 0.48, "3,461,612", 32, TEAL, True)
    add_text(slide, 0.8, 2.16, 5.4, 0.4, "requests counted from attached raw logs", 16, MUTED)
    add_text(slide, 0.8, 3.0, 5.4, 0.48, "89,280", 32, TEAL, True)
    add_text(slide, 0.8, 3.48, 5.4, 0.4, "continuous minute bins, including zero-load minutes", 16, MUTED)
    add_text(slide, 0.8, 4.28, 5.4, 0.48, "7,884", 32, GOLD, True)
    add_text(slide, 0.8, 4.76, 5.4, 0.55, "zero-request minutes (8.83% of the timeline)", 16, MUTED)
    image = ROOT / "results" / "figures" / "fig_phase1_full_series.png"
    if image.exists():
        slide.shapes.add_picture(str(image), Inches(6.25), Inches(1.72), width=Inches(6.35))
    add_text(slide, 0.8, 6.15, 11.7, 0.45,
             "Raw July/August archives match the copies in data/raw; all documented requests are accounted for.",
             14, INK)

    slide = add_base(prs, "NASA Table 4: close single-run reproduction", "03 | Baseline")
    headers = ["Horizon", "Model", "Paper RMSE", "Re-run RMSE", "Difference"]
    rows = [
        ["1-step", "ARIMA", "14.010", f"{float(by_key['ARIMA', 1]['RMSE']):.3f}", "+0.4%"],
        ["1-step", "Bi-LSTM", "13.551", f"{float(by_key['Bi-LSTM', 1]['RMSE']):.3f}", "+1.7%"],
        ["5-step", "ARIMA", "15.414", f"{float(by_key['ARIMA', 5]['RMSE']):.3f}", "+0.4%"],
        ["5-step", "Bi-LSTM", "14.390", f"{float(by_key['Bi-LSTM', 5]['RMSE']):.3f}", "+1.4%"],
    ]
    table = slide.shapes.add_table(5, 5, Inches(0.8), Inches(1.8), Inches(11.7), Inches(2.7)).table
    for col, text in enumerate(headers):
        table.cell(0, col).text = text
    for i, row in enumerate(rows, start=1):
        for j, text in enumerate(row):
            table.cell(i, j).text = text
    for r in range(5):
        for c in range(5):
            cell = table.cell(r, c)
            cell.margin_left = Inches(0.12)
            cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = INK if r == 0 else (PALE if r % 2 else WHITE)
            for p in cell.text_frame.paragraphs:
                p.font.name = "Aptos"
                p.font.size = Pt(15 if r else 14)
                p.font.bold = r == 0
                p.font.color.rgb = WHITE if r == 0 else INK
    add_text(slide, 0.8, 5.0, 11.7, 0.9,
             "One CPU run, seed 42, 70/30 chronological split. Neural results are stochastic; prediction speed is not compared across CPU/TPU environments.",
             15, MUTED)
    add_text(slide, 0.8, 6.0, 11.7, 0.5,
             "The refreshed scores use the complete time axis, including zero-request minutes.",
             15, TEAL, True)

    slide = add_base(prs, "Live pilot: readiness-aware planning under slow startup", "04 | Pilot result")
    headers = ["Metric", "Fixed mean", "Readiness-aware mean", "Change"]
    rows = [
        ["p95 latency (ms)", f"{mean('fixed_horizon', 'latency_p95_ms'):.1f}",
         f"{mean('readiness_aware', 'latency_p95_ms'):.1f}", f"{pct_change('latency_p95_ms'):.1f}%"],
        ["p99 latency (ms)", f"{mean('fixed_horizon', 'latency_p99_ms'):.1f}",
         f"{mean('readiness_aware', 'latency_p99_ms'):.1f}", f"{pct_change('latency_p99_ms'):.1f}%"],
        ["SLO violations (%)", f"{mean('fixed_horizon', 'slo_violations_pct'):.1f}",
         f"{mean('readiness_aware', 'slo_violations_pct'):.1f}",
         f"{pct_change('slo_violations_pct'):.1f}% (-14.1 pp)"],
        ["Replica-seconds (est.)", f"{mean('fixed_horizon', 'replica_seconds_est'):.0f}",
         f"{mean('readiness_aware', 'replica_seconds_est'):.0f}",
         f"{pct_change('replica_seconds_est'):.1f}%"],
        ["Pending seconds (est.)", f"{mean('fixed_horizon', 'pending_seconds_est'):.0f}",
         f"{mean('readiness_aware', 'pending_seconds_est'):.0f}",
         f"{pct_change('pending_seconds_est'):.1f}%"],
        ["Scale actions / run", f"{mean('fixed_horizon', 'scale_actions'):.1f}",
         f"{mean('readiness_aware', 'scale_actions'):.1f}", "lower"],
        ["HTTP errors (total)", "0", "1", "one ClientPayloadError"],
    ]
    table = slide.shapes.add_table(8, 4, Inches(0.65), Inches(1.55), Inches(12.0), Inches(3.8)).table
    for col, text in enumerate(headers):
        table.cell(0, col).text = text
    for i, row in enumerate(rows, start=1):
        for j, text in enumerate(row):
            table.cell(i, j).text = text
    for r in range(8):
        for c in range(4):
            cell = table.cell(r, c)
            cell.margin_left = Inches(0.1)
            cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = INK if r == 0 else (PALE if r % 2 else WHITE)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name = "Aptos"
                paragraph.font.size = Pt(12 if r else 13)
                paragraph.font.bold = r == 0
                paragraph.font.color.rgb = WHITE if r == 0 else INK
    add_text(slide, 0.75, 5.55, 11.8, 0.7,
             "Two paired seeds (1001, 1002); each arm received the same replay within each pair. "
             f"Controlled startup delay = 10 s; measured all-Ready p95 = {slow_ready:.3f} s; "
             f"readiness-aware horizon = {steps} steps.", 13, INK)
    add_text(slide, 0.75, 6.2, 11.8, 0.55,
             "Promising two-pair pilot; one optimized request error; n=2 is not enough for statistical significance.",
             14, RED, True)

    slide = add_base(prs, "Optimization: readiness-derived lead time", "05 | Method")
    add_bullets(slide, [
        f"Baseline readiness p95: {ready:.3f} s (five trials); controlled slow-start p95: {slow_ready:.3f} s (20 trials).",
        f"Control interval at compression 10: {interval:.1f} s; controlled condition derives a {steps}-step horizon.",
        "When demand arrives, the controller uses the forecast at the readiness horizon; if that horizon exceeds the cap, it logs the shortfall and uses a conservative recent-demand fallback.",
        "Scale-down is held while requested replicas are still Pending; scale-up remains immediate.",
    ], size=17, gap=0.92)
    add_text(slide, 0.8, 6.15, 11.7, 0.5,
             "The optimization is tested under a controlled 10-second boot delay; natural-startup baseline remains one step.",
             15, GOLD, True)

    slide = add_base(prs, "Validation and evidence boundary", "06 | Status")
    add_bullets(slide, [
        "Passed: 24 planner assertions, recursive-horizon tests, two dataset-loader tests, Python compilation, and measured-config checks.",
        "Full NASA baseline artifacts: dated CSV, prediction cache, and four figures in this presentation directory.",
        "Two matched live pairs completed; both reduced p95 and SLO violations; one optimized request returned ClientPayloadError.",
        "This pilot compares two horizon policies using the same live hybrid forecaster; it is not the paper's Bi-LSTM/HPA comparison.",
    ], size=17, gap=0.98)
    add_text(slide, 0.8, 6.2, 11.6, 0.45,
             "Defensible claim today: the two-pair controlled pilot favors readiness-aware scaling; replicate before claiming significance.",
             16, RED, True)

    slide = add_base(prs, "What is required to establish the win", "07 | Next experiment")
    add_bullets(slide, [
        "Increase repetitions (ideally at least 10 pairs) and report paired confidence intervals.",
        "Repeat under natural startup and additional controlled startup delays to test where the horizon helps.",
        "Keep p95/p99 latency and SLO violations paired with replica-seconds, pending-pod time, scale actions, and reversals.",
        "Add the paper's Bi-LSTM + CDT/RRS policy as a separate baseline before claiming superiority over the paper.",
    ], size=17, gap=0.98)
    add_text(slide, 0.8, 6.2, 11.5, 0.45,
             "Do not substitute the live hybrid controller comparison for the paper's simulated Bi-LSTM + CDT/RRS baseline.",
             14, GOLD, True)

    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
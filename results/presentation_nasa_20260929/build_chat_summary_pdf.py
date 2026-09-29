"""Generate a PDF summary of the project discussion and verified work."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)


OUT = Path(__file__).resolve().parent / "chat_summary_worklog.pdf"
INK = colors.HexColor("#182A3A")
TEAL = colors.HexColor("#007378")
PALE = colors.HexColor("#EEF4F4")
MUTED = colors.HexColor("#5D6C77")
GOLD = colors.HexColor("#C27124")
RED = colors.HexColor("#A1372D")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="DocTitle", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=24, leading=29, textColor=INK, alignment=TA_LEFT,
    spaceAfter=8))
styles.add(ParagraphStyle(
    name="Deck", parent=styles["Normal"], fontName="Helvetica",
    fontSize=11, leading=15, textColor=MUTED, spaceAfter=12))
styles.add(ParagraphStyle(
    name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=15, leading=19, textColor=TEAL, spaceBefore=7, spaceAfter=7,
    keepWithNext=True))
styles.add(ParagraphStyle(
    name="Body", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=6))
styles.add(ParagraphStyle(
    name="Small", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8, leading=10.5, textColor=MUTED, spaceAfter=4))
styles.add(ParagraphStyle(
    name="Cell", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8, leading=10, textColor=INK))
styles.add(ParagraphStyle(
    name="CellHead", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=8, leading=10, textColor=colors.white))
styles.add(ParagraphStyle(
    name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=10, leading=14, textColor=RED, spaceBefore=5, spaceAfter=8))


def p(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str) -> Paragraph:
    return Paragraph(f"- {text}", styles["Body"])


def table(rows, widths, header=True):
    wrapped = []
    for r, row in enumerate(rows):
        wrapped.append([p(str(value), "CellHead" if header and r == 0 else "Cell")
                        for value in row])
    t = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0,
              hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5D8")),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), INK))
        for row_index in range(1, len(rows)):
            if row_index % 2:
                commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE))
    t.setStyle(TableStyle(commands))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7E0E2"))
    canvas.line(0.65 * inch, 0.55 * inch, 7.85 * inch, 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.65 * inch, 0.35 * inch,
                      "Project discussion work log | 29 September 2026")
    canvas.drawRightString(7.85 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_story():
    story = [
        p("Project Discussion Summary and Work Log", "DocTitle"),
        p("Bi-LSTM Kubernetes autoscaling reproduction, NASA trace validation, and the "
          "cold-start-aware optimization. This report distinguishes user-provided "
          "history from code changes and measured evidence.", "Deck"),
        p("Executive summary", "Section"),
        p("The repository implements the core offline predictor comparison and "
          "Algorithm 1 autoscaler simulation from Dang-Quang and Yoo (2021). The "
          "attached NASA logs were verified and the full NASA baseline was rerun on "
          "the corrected continuous minute series. A readiness-aware controller path "
          "was added and unit-tested. A live optimization win has <b>not</b> been "
          "measured: the current readiness p95 is shorter than one control interval, "
          "and the existing KIND API is unavailable.", "Body"),
        p("Materials supplied in the conversation", "Section"),
        bullet("A Kubernetes/Prometheus workflow log describing a recommendationservice "
               "CrashLoopBackOff, its startupProbe fix, kube-prometheus-stack setup, "
               "Prometheus query checks, and CPU metric export."),
        bullet("The 2021 Applied Sciences paper on Bi-LSTM-based proactive Kubernetes "
               "autoscaling, including its workload forecasting, cooling-down period, "
               "resource removal strategy, and evaluation tables."),
        bullet("The NASA July/August 1995 access-log archives used for the paper's "
               "forecasting comparison."),
        p("Conversation and work timeline", "Section"),
        table([
            ["Stage", "Discussion and action", "Outcome"],
            ["Project identification", "Compared the repository README and source "
             "with the attached workflow log and paper.", "Project is a paper reproduction "
             "plus Kubernetes monitoring/autoscaling extensions."],
            ["Baseline assessment", "Inspected predictors, Algorithm 1, saved tables, "
             "and the live controller.", "Offline reproduction is substantial; the "
             "live hybrid controller is not the paper's Bi-LSTM controller."],
            ["Optimization design", "Selected readiness-derived forecast lead time, "
             "recursive use of the one-step model, and scale-down protection while "
             "replicas are Pending.", "A separate fixed-horizon comparison remains "
             "available through configuration."],
            ["Implementation", "Added readiness p95 wiring, horizon selection/logging, "
             "tests, deployment helper, and experiment documentation.", "24 planner "
             "assertions and recursive-forecast tests pass."],
            ["NASA validation", "Validated raw files, repaired package imports, and "
             "made the loader restore missing zero-request minutes.", "3,461,612 "
             "requests verified over 89,280 continuous minutes."],
            ["Presentation package", "Reran the full NASA baseline in isolated Python "
             "3.11 Docker and generated an evidence deck.", "Fresh CSV, figures, PDF work "
             "log, and PowerPoint are saved under results/presentation_nasa_20260929."],
        ], [1.15 * inch, 3.35 * inch, 2.65 * inch]),
        PageBreak(),
        p("Repository findings and implementation", "DocTitle"),
        p("What the repository supports", "Section"),
        bullet("The offline paper driver trains ARIMA, forward LSTM, and Bi-LSTM for "
               "one- and five-step forecasts, and computes prediction and elasticity "
               "metrics."),
        bullet("The live KIND controller uses a Prophet-plus-residual-network hybrid. "
               "Its current experiment is distinct from the paper's Bi-LSTM/HPA "
               "comparison."),
        bullet("Two Python import collisions were found and repaired: the data scripts "
               "resolved an empty src/data package instead of src/data.py, and the "
               "baseline model scripts resolved an empty src/models package instead "
               "of src/models.py."),
        bullet("The shared loader now accepts both period/count and ds/y CSV layouts, "
               "groups duplicate minutes, sorts timestamps, and inserts zero counts "
               "for missing minutes."),
        p("Cold-start-aware controller changes", "Section"),
        bullet("Compute required lead steps from measured readiness p95 plus forecast "
               "overhead, divided by the trace-clock control interval."),
        bullet("Recursively apply the packaged one-step forecaster up to a configured "
               "five-step cap; log required steps, used steps, and any uncovered lead."),
        bullet("If readiness exceeds the forecast cap, log the shortfall and use recent "
               "observed demand as a conservative fallback."),
        bullet("Hold scale-down while requested replicas are not all Ready. A scale-up "
               "cooldown still prevents repeated requests during startup."),
        bullet("The readiness-aware mode is enabled in the controller manifest; setting "
               "READINESS_AWARE_SCALING=0 selects the fixed-horizon arm."),
        p("Key implementation files", "Section"),
        table([
            ["File", "Purpose"],
            ["src/controller/config.py", "Readiness measurements, horizon derivation, "
             "cap, and warning/report fields."],
            ["src/controller/forecaster.py", "Recursive multi-step prediction using "
             "the existing one-step model."],
            ["src/controller/planner.py", "Pending-replica scale-down guard."],
            ["src/controller/main.py", "Horizon selection, safety fallback, and "
             "decision-log instrumentation."],
            ["k8s/controller.yaml", "KIND runtime settings and measured readiness "
             "p95 value."],
            ["COLD_START_OPTIMIZATION.md", "Policy, parameters, and paired-run "
             "instructions."],
        ], [2.45 * inch, 4.7 * inch]),
        p("NASA dataset verification", "Section"),
        p("The raw parser read 3,461,613 lines, skipped one malformed/truncated July "
          "record, and counted 3,461,612 requests. All observed timezone offsets were "
          "-0400. The regenerated full grid spans 1995-07-01 00:00 through 1995-08-31 "
          "23:59, has 89,280 minute bins, and contains 7,884 zero-request minutes. "
          "Both checked-in NASA CSVs match the raw counts at every timestamp they "
          "contain; the shared loader fills the 7,884 omitted bins for model input.",
          "Body"),
        PageBreak(),
        p("Measured results and limits", "DocTitle"),
        p("Refreshed full NASA run", "Section"),
        p("One CPU run, seed 42, chronological 70/30 split, full training mode. "
          "RMSE is in requests per minute; relative difference is versus the cited "
          "paper Table 4 value.", "Body"),
        table([
            ["Horizon", "Model", "Paper RMSE", "Rerun RMSE", "Delta"],
            ["1-step", "ARIMA", "14.010", "14.063", "+0.4%"],
            ["1-step", "Bi-LSTM", "13.551", "13.788", "+1.7%"],
            ["5-step", "ARIMA", "15.414", "15.481", "+0.4%"],
            ["5-step", "Bi-LSTM", "14.390", "14.594", "+1.4%"],
        ], [0.9 * inch, 1.2 * inch, 1.25 * inch, 1.35 * inch, 0.9 * inch]),
        Spacer(1, 8),
        p("Interpretation: the refreshed NASA accuracy is close to the paper. It is "
          "not bit-identical because neural training is stochastic and hardware "
          "differs. The sparse CSV was correctly expanded to include zero-request "
          "minutes; old result tables are not assumed to represent this corrected "
          "input. The FIFA 98 tables were not rerun after the shared-loader change.",
          "Body"),
        p("Readiness measurements and claim boundary", "Section"),
        table([
            ["Quantity", "Current value", "Interpretation"],
            ["Natural-start p95", "2.8662 s", "5 scale-out trials; selects one step."],
            ["Controlled p95", "12.715 s", "20 trials with STARTUP_DELAY_S=10; "
             "selects three steps."],
            ["Control interval", "6.0 s", "One trace-minute at compression 10."],
            ["Pilot lead", "3 steps", "Controlled slow-start condition."],
        ], [1.45 * inch, 1.35 * inch, 4.35 * inch]),
         p("The existing compression-sweep JSON is simulated, uses an unmeasured "
           "placeholder cold-start value, and cannot resolve sub-minute startup effects. "
           "It is not used as live optimization evidence.", "Body"),
        p("KIND experiment status", "Section"),
                p("After recovering the existing KIND control plane, two matched burst pairs "
                    "(seeds 1001 and 1002) completed under the controlled 10-second startup delay:",
           "Body"),
         table([
                        ["Metric", "Fixed one-step mean", "Readiness-aware mean"],
                        ["p95 latency", "3,814.5 ms", "1,970.2 ms (-48.4%)"],
                        ["p99 latency", "5,708.8 ms", "3,241.6 ms (-43.2%)"],
                        ["SLO violations", "36.28%", "22.15% (-14.12 pp)"],
                        ["Replica-seconds (estimated)", "3,276", "3,246 (-0.9%)"],
                        ["Pending seconds (estimated)", "102", "75 (-26.5%)"],
                        ["Scale actions / direction reversals", "8.5 / 4", "7.5 / 3"],
                        ["Request errors", "0", "1 (ClientPayloadError)"],
         ], [2.25 * inch, 2.15 * inch, 2.75 * inch]),
                p("Both pairs reduced p95, p99, and SLO violations; paired p95 deltas were "
                    "-1,335 ms and -2,353 ms, with SLO reductions of 11.19 and 17.06 percentage "
                    "points. This is a consistent pilot direction, but n=2 is not statistical or "
                    "publication-grade evidence. The comparison uses the same live hybrid "
                    "forecaster with different planning horizons; it is not a live reproduction "
                    "of the paper's Bi-LSTM/HPA experiment.", "Callout"),
         p("Next steps", "Section"),
                bullet("Increase paired repetitions (ideally at least 10) and report uncertainty."),
                bullet("Repeat under natural startup and additional controlled startup delays."),
         bullet("Add the paper-faithful Bi-LSTM + CDT/RRS policy as a separate baseline "
             "before claiming superiority over the paper."),
         bullet("Report p95/p99 latency and SLO violations together with replica-seconds, "
             "pending-pod time, scaling actions, and confidence intervals."),
        p("Validation performed", "Section"),
        p("NASA preprocessing sanity gates passed; two dataset regression tests passed; "
          "24 planner assertions passed; recursive horizon tests passed; experiment "
          "CLIs imported; Python compilation and manifest consistency checks passed. "
          "The eight-slide PowerPoint file was reopened and validated with python-pptx.",
          "Body"),
        p("Generated artifacts", "Section"),
        p("results/presentation_nasa_20260929/cold_start_aware_evidence.pptx<br/>"
          "results/presentation_nasa_20260929/EVIDENCE.md<br/>"
                    "results/presentation_nasa_20260929/live_pair/scored_results.md<br/>"
          "results/presentation_nasa_20260929/tables/table_nasa_accuracy.csv<br/>"
          "results/presentation_nasa_20260929/figures/", "Small"),
    ]
    return story


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        rightMargin=0.65 * inch, leftMargin=0.65 * inch,
        topMargin=0.6 * inch, bottomMargin=0.75 * inch,
        title="Project Discussion Summary and Work Log",
        author="GitHub Copilot",
    )
    doc.build(build_story(), onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
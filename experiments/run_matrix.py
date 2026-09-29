"""Phase 7 - run the full experiment matrix, unattended, and survive its own failures.

    ./.venv/bin/python experiments/run_matrix.py --plan          # print, run nothing
    ./.venv/bin/python experiments/run_matrix.py                 # run (resumes)
    ./.venv/bin/python experiments/run_matrix.py --only burst    # one segment
    ./.venv/bin/python experiments/run_matrix.py --status        # progress so far

4 arms x 3 segments x 5 reps = 60 runs. At C=10 the segments take 48 / 18 / 9 minutes of
wall time respectively, so one full rep of all three is ~2.9 h per arm and the matrix is
roughly **23 hours** of pure replay before overheads. That is an overnight-and-then-some
job, which drives every design decision in this file.

--------------------------------------------------------------------------------
Resume, do not restart
--------------------------------------------------------------------------------
A 23-hour sweep WILL be interrupted - laptop sleep, an OOM, a KIND node dying, someone
closing the terminal. Restarting from scratch each time means the matrix never completes.

So completion is defined by a file on disk, not by a counter in memory: a run is done iff
its directory contains `metrics.json`. On startup we scan `experiments/runs/`, skip what
is complete, and continue. A run that crashed halfway leaves its directory WITHOUT
metrics.json and is re-run - into a NEW directory, because overwriting the partial one
would destroy the only evidence of how it failed.

--------------------------------------------------------------------------------
Randomised order, and why it is not cosmetic
--------------------------------------------------------------------------------
Runs are executed in a seeded shuffle rather than arm-by-arm. A laptop running for 23
hours is not a stationary environment: thermal throttling, background updates, and disk
pressure all drift. Running all HPA runs first and all hybrid runs last would confound
the arm with the time of night. Shuffling turns a systematic bias into noise that the
5 repetitions can average over - which is the entire reason for having repetitions.

--------------------------------------------------------------------------------
Never overwrite raw data
--------------------------------------------------------------------------------
Run directories are timestamped and never reused. Every file is written once. If a run is
repeated because it failed, the failure's directory stays on disk with its logs.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src" / "controller")]

RUNS = REPO / "experiments" / "runs"
MEASURED = REPO / "experiments" / "measured_constants.json"


def _default_slo_ms() -> float:
    """The SLO the capacity was measured against. Refuses to invent one."""
    try:
        return float(json.loads(MEASURED.read_text())["slo_ms"])
    except (OSError, ValueError, KeyError):
        # No measurement yet. Return a sentinel the preflight will reject rather than a
        # plausible number that would silently score 60 runs against the wrong target.
        return float("nan")
ARMS = ["hpa-cpu", "keda-rate", "predictive-prophet", "predictive-hybrid"]
SEGMENTS = ["diurnal", "burst", "quiet"]
REPS = 5
SHUFFLE_SEED = 20260823

NS = "aiops"
DEPLOY = "target-app"
SAMPLE_INTERVAL_S = 1.0          # replica/CPU polling during a run

_stop = False


def _sig(signum, _frame):
    global _stop
    _stop = True
    print(f"\n[signal {signum}] finishing the current run, then stopping. "
          f"Re-run to resume.", flush=True)


def sh(cmd: list[str], timeout: float = 300, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])}... exited {p.returncode}\n"
                           f"stdout: {p.stdout[-2000:]}\nstderr: {p.stderr[-2000:]}")
    return p


# ---------------------------------------------------------------------------------
def planned_runs() -> list[tuple[str, str, int]]:
    combos = [(a, s, r) for a in ARMS for s in SEGMENTS for r in range(1, REPS + 1)]
    random.Random(SHUFFLE_SEED).shuffle(combos)
    return combos


def is_complete(arm: str, seg: str, rep: int) -> Path | None:
    """A run is complete iff some directory for it holds metrics.json."""
    for d in sorted(RUNS.glob(f"*_{arm}_{seg}_rep{rep}")):
        if (d / "metrics.json").exists():
            return d
    return None


def failed_attempts(arm: str, seg: str, rep: int) -> int:
    return sum(1 for d in RUNS.glob(f"*_{arm}_{seg}_rep{rep}")
               if not (d / "metrics.json").exists())


# ---------------------------------------------------------------------------------
def sample_cluster(outdir: Path, stop_file: Path) -> subprocess.Popen:
    """Poll replica counts and pod CPU for the duration of the run.

    A separate process, not a thread: the replay is the priority, and a sampler sharing
    the GIL with anything else would skew its own timestamps. It exits when the stop file
    appears, which is more robust than signalling across process boundaries when the
    parent may itself be dying.
    """
    script = f'''
import json, time, subprocess, pathlib, sys
out = pathlib.Path({str(outdir)!r})
stop = pathlib.Path({str(stop_file)!r})
rep_f = (out / "raw" / "replicas.jsonl").open("a", buffering=1)
cpu_f = (out / "raw" / "cpu.jsonl").open("a", buffering=1)
last_cpu = 0.0
while not stop.exists():
    t = time.time()
    try:
        r = subprocess.run(["kubectl","get","deploy",{DEPLOY!r},"-n",{NS!r},"-o",
            "jsonpath={{.spec.replicas}} {{.status.readyReplicas}} {{.status.unavailableReplicas}}"],
            capture_output=True, text=True, timeout=5)
        parts = (r.stdout or "").split()
        spec = int(parts[0]) if len(parts) > 0 and parts[0] else 0
        ready = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        rep_f.write(json.dumps(dict(ts=t, spec=spec, ready=ready)) + "\\n")
    except Exception as e:
        rep_f.write(json.dumps(dict(ts=t, error=str(e))) + "\\n")
    time.sleep({SAMPLE_INTERVAL_S})
rep_f.close(); cpu_f.close()
'''
    return subprocess.Popen([sys.executable, "-c", script],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sample_cpu_once(outdir: Path, prom: str) -> None:
    """One CPU counter reading PER POD, appended. Called before and after the replay.

    WHY PER POD, AND NOT sum() - this was a real bug, found at run 5 of the sweep.

    container_cpu_usage_seconds_total is a per-container COUNTER: it only ever rises,
    and it vanishes when the pod does. The previous version asked Prometheus for
    sum(...) across pods and stored that single number, then scored a run by
    differencing the two boundary samples.

    That is wrong whenever pods turn over during the run - which autoscaling guarantees.
    The sum drops when a pod is deleted, so the difference can be NEGATIVE, and the
    scorer clamped negatives to zero. Observed on hpa-cpu/quiet/rep1:

        start 1076.39   (inherited from the PREVIOUS run's long-lived pods)
        end     31.13   (fresh pods after the arm switch reset the counters)
        delta = -1045.26  ->  clamped  ->  cpu_seconds_total = 0.0

    A run that served 5,898 requests was recorded as costing zero CPU. Silently. CPU
    cost is one of the two axes of this entire experiment, so this would have quietly
    corrupted the cost comparison for every run whose pods turned over.

    Summing BEFORE differencing destroys the information needed to detect a reset.
    Keeping each pod's counter separate lets the scorer difference per pod and treat a
    pod that disappeared as "contributed what it had", which is the truth.
    """
    q = (f'container_cpu_usage_seconds_total{{namespace="{NS}",'
         f'pod=~"{DEPLOY}-.*",container="app"}}')
    per_pod: dict[str, float] | None = {}
    try:
        import requests
        r = requests.get(f"{prom}/api/v1/query", params={"query": q}, timeout=10)
        for s in r.json()["data"]["result"]:
            per_pod[s["metric"]["pod"]] = float(s["value"][1])
    except Exception as e:  # noqa: BLE001
        per_pod = None
        print(f"    (cpu sample failed: {type(e).__name__})", flush=True)
    with (outdir / "raw" / "cpu.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(
            ts=time.time(),
            per_pod=per_pod,
            # Kept only so a human reading the file sees the same number the old format
            # showed. The scorer must NOT use this - see the docstring above.
            cpu_seconds_total_UNSAFE_SUM=None if per_pod is None else sum(per_pod.values()),
        )) + "\n")


# ---------------------------------------------------------------------------------
def one_run(arm: str, seg: str, rep: int, args) -> tuple[bool, str, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = RUNS / f"{stamp}_{arm}_{seg}_rep{rep}"
    (outdir / "raw").mkdir(parents=True, exist_ok=True)

    from config import Config
    cfg = Config()
    seg_meta = json.loads((REPO / "load" / "segments.json").read_text())["segments"][seg]

    # The config is written BEFORE the run, so that a crashed run still records what it
    # was attempting. A config written afterwards documents only successes.
    config = dict(
        run_id=outdir.name, arm=arm, segment=seg, rep=rep,
        segment_sha=seg_meta["sha256_16"], segment_minutes=seg_meta["minutes"],
        trace_start=seg_meta["first_minute"],
        compression=cfg.compression, magnitude=1.0,
        per_pod_capacity_rps=cfg.per_pod_capacity_rps,
        min_replicas=cfg.min_replicas, max_replicas=cfg.max_replicas,
        slo_ms=args.slo_ms,
        pod_cold_start_wall_s=cfg.pod_cold_start_wall_s,
        pod_cold_start_measured=cfg.pod_cold_start_measured,
        controller_config=cfg.to_dict(),
        git_commit=sh(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                      check=False).stdout.strip(),
        git_dirty=bool(sh(["git", "-C", str(REPO), "status", "--porcelain"],
                          check=False).stdout.strip()),
        started_at=time.time(), started_at_iso=datetime.now().isoformat(),
        host_note=args.note,
    )
    (outdir / "config.json").write_text(json.dumps(config, indent=2))

    log = (outdir / "run.log").open("w", buffering=1)

    def say(msg: str) -> None:
        print(f"    {msg}", flush=True)
        log.write(msg + "\n")

    stop_file = outdir / ".stop_sampler"
    sampler = None
    try:
        # 1. Switch arm. This tears down EVERY arm, resets to baseline, and asserts
        #    exactly one is active afterwards - the Phase 6 reset protocol.
        say(f"switching to arm {arm}")
        if arm.startswith("predictive"):
            trace_start = seg_meta["first_minute"]
            sh(["kubectl", "set", "env", f"deploy/{'predictive-autoscaler'}",
                "-n", NS, f"TRACE_START={trace_start}"])
            say(f"controller trace clock aligned to {trace_start}")
        p = sh([str(REPO / "experiments" / "switch_arm.sh"), arm], timeout=600, check=False)
        log.write(p.stdout + p.stderr)
        if p.returncode != 0:
            return False, f"switch_arm.sh failed (rc={p.returncode})", outdir

        # 2. Start sampling the cluster.
        sampler = sample_cluster(outdir, stop_file)
        sample_cpu_once(outdir, args.prometheus)

        # 3. Replay the segment. This is the long part.
        say(f"replaying {seg} ({seg_meta['minutes']} trace-min, "
            f"~{seg_meta['minutes'] * 60 / cfg.compression / 60:.0f} min wall)")
        replay = sh([str(REPO / ".venv" / "bin" / "python"),
                     str(REPO / "load" / "trace_replay.py"),
                     "--segment", seg, "--url", args.url,
                     "--compression", str(cfg.compression),
                     "--seed", str(1000 + rep),      # a different arrival draw per rep
                     "--tag", f"{arm}-rep{rep}"],
                    timeout=seg_meta["minutes"] * 60 / cfg.compression + 900,
                    check=False)
        log.write(replay.stdout + replay.stderr)
        if replay.returncode != 0:
            return False, f"trace_replay failed (rc={replay.returncode})", outdir

        # The replay writes into its own experiments/<id>/ directory; move its raw data
        # into this run so everything about the run is in one place.
        m = [l for l in replay.stdout.splitlines() if "wrote experiments/" in l]
        if m:
            src = REPO / m[-1].split("wrote ")[1].split("/summary.json")[0]
            for f in ("raw/requests.csv", "summary.json"):
                if (src / f).exists():
                    shutil.copy2(src / f, outdir / "raw" / Path(f).name)

        sample_cpu_once(args_outdir := outdir, args.prometheus)

        # 4. Pull the controller's decision log, for predictive arms.
        if arm.startswith("predictive"):
            pod = sh(["kubectl", "get", "pods", "-n", NS, "-l",
                      "app=predictive-autoscaler", "-o",
                      "jsonpath={.items[0].metadata.name}"], check=False).stdout.strip()
            if pod:
                cp = sh(["kubectl", "cp", "-n", NS,
                         f"{pod}:/var/log/controller/decisions.jsonl",
                         str(outdir / "raw" / "decisions.jsonl")],
                        timeout=120, check=False)
                log.write(cp.stdout + cp.stderr)
                if not (outdir / "raw" / "decisions.jsonl").exists():
                    return False, "decision log missing for a predictive arm", outdir
            else:
                return False, "no predictive-autoscaler pod found", outdir
        return True, "ok", outdir

    except subprocess.TimeoutExpired as e:
        return False, f"timeout: {e}", outdir
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}", outdir
    finally:
        stop_file.touch()
        if sampler:
            try:
                sampler.wait(timeout=15)
            except subprocess.TimeoutExpired:
                sampler.kill()
        config["ended_at"] = time.time()
        (outdir / "config.json").write_text(json.dumps(config, indent=2))
        log.close()


# ---------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true", help="print the matrix and exit")
    ap.add_argument("--status", action="store_true", help="progress so far and exit")
    ap.add_argument("--only", default=None, help="restrict to one segment")
    ap.add_argument("--arm", default=None, help="restrict to one arm")
    ap.add_argument("--url", default="http://localhost:30080/")
    ap.add_argument("--prometheus", default="http://localhost:30090")
    # Default comes from the MEASURED capacity run, not a literal. calibrate.py derives
    # the SLO as 5x the unloaded p50 it measured against the real pod (D-012), and that
    # is the same threshold the replica formula was calibrated against - scoring runs at
    # any other value would mean the arms provisioned for one target and were graded on
    # another. A hardcoded 142.7 here came from the LOCAL process and is 4.2x too tight.
    ap.add_argument("--slo-ms", type=float, default=_default_slo_ms(),
                    help="SLO threshold; defaults to the measured value in "
                         "experiments/measured_constants.json")
    ap.add_argument("--note", default="", help="free text recorded in every config.json")
    ap.add_argument("--max-runs", type=int, default=0, help="stop after N runs")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    RUNS.mkdir(parents=True, exist_ok=True)

    combos = planned_runs()
    if args.only:
        combos = [c for c in combos if c[1] == args.only]
    if args.arm:
        combos = [c for c in combos if c[0] == args.arm]

    done = [c for c in combos if is_complete(*c)]
    todo = [c for c in combos if not is_complete(*c)]

    if args.status or args.plan:
        print(f"\nPhase 7 matrix: {len(ARMS)} arms x {len(SEGMENTS)} segments "
              f"x {REPS} reps = {len(ARMS) * len(SEGMENTS) * REPS} runs")
        print(f"  complete : {len(done)}")
        print(f"  remaining: {len(todo)}")
        seg_meta = json.loads((REPO / "load" / "segments.json").read_text())["segments"]
        est = sum(seg_meta[s]["minutes"] * 6 for _, s, _ in todo) / 3600
        print(f"  estimated replay time remaining: {est:.1f} h (excluding overheads)\n")
        print(f"  {'arm':<20} {'segment':<9} {'done':>5} {'failed attempts':>16}")
        print("  " + "-" * 54)
        for a in ARMS:
            for s in SEGMENTS:
                d = sum(1 for r in range(1, REPS + 1) if is_complete(a, s, r))
                f = sum(failed_attempts(a, s, r) for r in range(1, REPS + 1))
                print(f"  {a:<20} {s:<9} {d:>3}/{REPS} {f:>16}")
        if args.plan:
            print(f"\n  next 10 in shuffled order (seed {SHUFFLE_SEED}):")
            for a, s, r in todo[:10]:
                print(f"    {a:<20} {s:<9} rep {r}")
        return 0

    print(f"\nPhase 7 sweep: {len(todo)} runs remaining of {len(combos)}")
    print(f"  resume policy: a run is complete iff its directory holds metrics.json")
    print(f"  order: seeded shuffle ({SHUFFLE_SEED}) so arm is not confounded with time\n")

    from collect import collect
    ok_n = fail_n = 0
    for i, (arm, seg, rep) in enumerate(todo, 1):
        if _stop:
            break
        if args.max_runs and (ok_n + fail_n) >= args.max_runs:
            print(f"\n--max-runs {args.max_runs} reached.")
            break
        prev = failed_attempts(arm, seg, rep)
        print(f"[{i}/{len(todo)}] {arm} / {seg} / rep {rep}"
              + (f"   (previous failed attempts: {prev})" if prev else ""), flush=True)
        t0 = time.time()
        ok, detail, outdir = one_run(arm, seg, rep, args)
        if ok:
            try:
                collect(outdir, args.slo_ms)
                ok_n += 1
                print(f"    done in {(time.time() - t0) / 60:.1f} min -> "
                      f"{outdir.name}/metrics.json", flush=True)
            except Exception as e:  # noqa: BLE001
                fail_n += 1
                print(f"    COLLECT FAILED: {type(e).__name__}: {e}", flush=True)
                (outdir / "FAILED").write_text(f"collect: {e}\n")
        else:
            fail_n += 1
            print(f"    FAILED after {(time.time() - t0) / 60:.1f} min: {detail}",
                  flush=True)
            # A marker file, and NO metrics.json - so the resume logic re-runs this
            # combination into a fresh directory while this one stays as evidence.
            (outdir / "FAILED").write_text(detail + "\n")

    print(f"\nsweep stopped: {ok_n} succeeded, {fail_n} failed this session")
    print(f"re-run to resume; ./experiments/run_matrix.py --status for progress")
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

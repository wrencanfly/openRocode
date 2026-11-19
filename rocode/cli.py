"""Command line: `rocode run`, `rocode bench`, `rocode demo`, `rocode view`, `rocode tasks`, `rocode replay`."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path

from .core.control import ControlChannel
from .core.events import (AssistantReply, EpisodeEnd, EStop, Event, Notice, OperatorInput, Paused, Resumed,
                          ToolDenied, ToolEnd, ToolStart, TurnStart)
from .session import SessionConfig, default_out_dir, run_session
from .tasks import TASKS


class ConsoleView:
    """Prints a readable transcript of the episode as events arrive."""

    def __init__(self, out=sys.stdout):
        self.out = out

    def __call__(self, e: Event) -> None:
        p = lambda s: print(s, file=self.out, flush=True)  # noqa: E731
        if isinstance(e, TurnStart):
            p(f"\n── turn {e.turn}")
        elif isinstance(e, AssistantReply):
            if getattr(e, "reasoning", "").strip():
                for line in e.reasoning.strip().splitlines()[:4]:
                    if line.strip():
                        p(f"   thinks: {line.strip()}")
            if e.text.strip():
                for line in e.text.strip().splitlines()[:6]:
                    p(f"   model: {line}")
        elif isinstance(e, ToolStart):
            indent = "      " if e.parent_id else "   "
            args = e.args.get("code") if e.name == "run_code" else json.dumps(e.args)
            if e.name == "run_code":
                p(f"{indent}▶ run_code")
                for line in str(args).splitlines():
                    p(f"{indent}  │ {line}")
            else:
                p(f"{indent}▶ {e.name} {args}")
        elif isinstance(e, ToolEnd):
            indent = "      " if e.parent_id else "   "
            first = e.text.splitlines()[0] if e.text else ""
            mark = "✗" if e.is_error else "✓"
            p(f"{indent}{mark} {first}  [{', '.join(e.observation_ids)}] {e.duration_s:.1f}s")
        elif isinstance(e, ToolDenied):
            p(f"   ⛔ {e.name}: {e.reason}")
        elif isinstance(e, OperatorInput):
            p(f"   operator: {e.text}")
        elif isinstance(e, (Paused, Resumed, EStop, Notice)):
            p(f"   [{e.type}] {getattr(e, 'text', getattr(e, 'source', ''))}")
        elif isinstance(e, EpisodeEnd):
            s = e.stats
            p(f"\n── ended: {e.reason} after {s['steps']} steps, {s['turns']} turns, {s['seconds']} s; "
              f"tokens in {s['usage']['input_tokens']} (cached {s['usage']['cache_read_tokens']}), "
              f"out {s['usage']['output_tokens']}")
            if s.get("error"):
                p(f"   error: {s['error']}")
            if s.get("finish"):
                p(f"   model says: success={s['finish'].get('success')}: {s['finish'].get('summary')}")


def operator_console(control: ControlChannel) -> None:
    """Read operator lines from the terminal: text goes to the model, /pause /resume /stop control the arm."""
    def loop():
        for line in sys.stdin:
            line = line.strip()
            if line == "/stop":
                control.estop("operator console")
            elif line == "/pause":
                control.pause()
            elif line == "/resume":
                control.resume()
            elif line:
                control.say(line)

    threading.Thread(target=loop, daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rocode", description="Drive a robot with a multimodal model.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run one episode with a real model.")
    run.add_argument("task", choices=sorted(TASKS))
    run.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    run.add_argument("--model", default=None, help="Default: claude-opus-5-5 for anthropic.")
    run.add_argument("--base-url", default=None, help="OpenAI-compatible server URL (vLLM, Qwen, DeepSeek...).")
    run.add_argument("--policy", default="both", choices=["tools", "code", "both"],
                     help="Act through direct tool calls, programs, or both.")
    run.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    run.add_argument("--reasoning", action="store_true",
                     help="openai provider: send --effort as reasoning_effort and record the model's reasoning summary.")
    run.add_argument("--max-steps", type=int, default=60)
    run.add_argument("--keep-images", type=int, default=None, help="Trim pictures older than N tool results.")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--out", type=Path, default=Path("runs"))
    run.add_argument("--view", action="store_true", help="Open the dashboard to watch and steer the run.")
    run.add_argument("--port", type=int, default=8765)
    run.add_argument("--no-open", action="store_true", help="With --view: print the address without opening a browser.")

    demo = sub.add_parser("demo", help="Scripted episode, no API key needed (checks your install).")
    demo.add_argument("--out", type=Path, default=Path("runs"))

    view = sub.add_parser("view", help="Open the dashboard for recorded (and running) episodes.")
    view.add_argument("runs", type=Path, nargs="?", default=Path("runs"))
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--no-open", action="store_true", help="Print the address without opening a browser.")

    bench = sub.add_parser("bench", help="Run tasks x models x seeds and compare them in the dashboard.")
    bench.add_argument("tasks", nargs="+", choices=sorted(TASKS), metavar="TASK")
    bench.add_argument("--model", dest="models", action="append", metavar="PROVIDER:MODEL",
                       help="Repeat for each model, e.g. --model anthropic:claude-opus-5-5 --model openai:gpt-6.1-sol. "
                            "Default: anthropic:claude-opus-5-5.")
    bench.add_argument("--base-url", default=None, help="OpenAI-compatible server URL, used for every openai: model.")
    bench.add_argument("--seeds", type=int, nargs="+", default=None, metavar="N",
                       help="Seeds to run; each moves the objects a little. Default: the unjittered layout once.")
    bench.add_argument("--repeat", type=int, default=None, metavar="K", help="Shorthand for --seeds 0 1 ... K-1.")
    bench.add_argument("--policy", default="both", choices=["tools", "code", "both"])
    bench.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    bench.add_argument("--reasoning", action="store_true",
                       help="openai: models: send --effort as reasoning_effort and record reasoning summaries.")
    bench.add_argument("--max-steps", type=int, default=60)
    bench.add_argument("--keep-images", type=int, default=None)
    bench.add_argument("--out", type=Path, default=Path("runs"))
    bench.add_argument("--yes", action="store_true", help="Start without confirming the number of episodes.")
    bench.add_argument("--view", action="store_true", help="Open the comparison page while the batch runs.")
    bench.add_argument("--port", type=int, default=8765)
    bench.add_argument("--no-open", action="store_true")

    sub.add_parser("tasks", help="List tasks.")
    rep = sub.add_parser("replay", help="Print a recorded episode.")
    rep.add_argument("dir", type=Path)

    args = ap.parse_args(argv)
    if args.cmd == "tasks":
        for t in TASKS.values():
            print(f"{t.id:<20} {t.instruction}")
        return 0
    if args.cmd == "replay":
        view = ConsoleView()
        from .core import events as ev

        for line in (args.dir / "episode.jsonl").read_text().splitlines():
            d = json.loads(line)
            cls = getattr(ev, d.pop("type"), None)
            if cls is not None:
                view(cls(**d))
        evaluation = args.dir / "evaluation.json"
        if evaluation.exists():
            print(f"   ground truth: success={json.loads(evaluation.read_text())['success']}")
        return 0

    if args.cmd == "view":
        from .view.server import ViewerServer

        server = ViewerServer(args.runs, port=args.port)
        print(f"Dashboard: {server.url}  (episodes in {args.runs.resolve()}; Ctrl-C to quit)")
        if not args.no_open:
            webbrowser.open(server.url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
        return 0

    if args.cmd == "bench":
        return bench_command(args, ap)

    if args.cmd == "demo":
        from .demo import scripted_demo

        report = scripted_demo(default_out_dir(args.out, "demo"), subscribers=[ConsoleView()])
        print(f"   ground truth: success={report.success}\n   recording: {report.out_dir}")
        return 0 if report.success else 1

    model = args.model or ("claude-opus-5-5" if args.provider == "anthropic" else None)
    if model is None:
        ap.error("--model is required with --provider openai")
    out = default_out_dir(args.out, args.task)
    cfg = SessionConfig(task=args.task, provider=args.provider, model=model, base_url=args.base_url,
                        policy=args.policy, effort=args.effort, max_steps=args.max_steps,
                        keep_image_turns=args.keep_images, seed=args.seed, out_dir=out, reasoning=args.reasoning)
    control = ControlChannel()
    if sys.stdin.isatty():
        print("Type to talk to the model mid-run. /pause, /resume, /stop (emergency stop).")
        operator_console(control)
    server = None
    if args.view:
        from .view.server import ViewerServer

        server = ViewerServer(args.out, port=args.port, control=control, live_id=out.name).start()
        print(f"Dashboard: {server.url}")
        if not args.no_open:
            webbrowser.open(server.url)
    report = run_session(cfg, control=control, subscribers=[ConsoleView()])
    print(f"   ground truth: success={report.success}\n   recording: {out}")
    if server is not None:
        print(f"Dashboard still open at {server.url} (Ctrl-C to quit)")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            server.shutdown()
    return 0 if report.success else 1


def bench_command(args, ap: argparse.ArgumentParser) -> int:
    import signal

    from .bench import Bench, BenchPlan, ModelSpec

    try:
        models = [ModelSpec.parse(m) for m in (args.models or ["anthropic:claude-opus-5-5"])]
    except ValueError as err:
        ap.error(str(err))
    if args.seeds is not None and args.repeat is not None:
        ap.error("use --seeds or --repeat, not both")
    seeds = args.seeds if args.seeds is not None else list(range(args.repeat)) if args.repeat else [None]
    plan = BenchPlan(tasks=list(dict.fromkeys(args.tasks)), models=models, seeds=seeds, policy=args.policy,
                     effort=args.effort, max_steps=args.max_steps, keep_image_turns=args.keep_images,
                     base_url=args.base_url, reasoning=args.reasoning)
    n = len(plan.cells())
    print(f"{n} episode(s): {len(plan.tasks)} task(s) x {len(models)} model(s) x {len(seeds)} seed(s). "
          "Each one calls the model API.")
    if not args.yes and sys.stdin.isatty() and input("Start? [y/N] ").strip().lower() not in ("y", "yes"):
        return 1

    def started(i, total, e):
        seed = "default" if e["seed"] is None else e["seed"]
        print(f"[{i}/{total}] {e['task']} · {e['model']} · seed {seed} …", end=" ", flush=True)

    def finished(i, total, e):
        if e["status"] == "error":
            print(f"error: {e.get('error', '')}")
        else:
            print(f"{e['status']} in {e.get('turns')} turns, {e.get('seconds')} s")

    bench = Bench(plan, args.out, on_start=started, on_done=finished)
    server = None
    if args.view:
        from .view.server import ViewerServer

        server = ViewerServer(args.out, port=args.port).start()
        url = f"{server.url}#/compare?batch={bench.id}"
        print(f"Comparison: {url}")
        if not args.no_open:
            webbrowser.open(url)

    def interrupt(signum, frame):
        if bench.stopping:
            raise KeyboardInterrupt
        print("\nStopping: emergency stop on the current episode, the rest are skipped. Ctrl-C again to quit now.")
        bench.stop()

    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        manifest = bench.run()
    finally:
        signal.signal(signal.SIGINT, previous)
    print_bench_table(manifest)
    print(f"Batch {bench.id}: {bench.manifest_path}")
    if server is not None:
        print(f"Dashboard still open at {server.url}#/compare?batch={bench.id} (Ctrl-C to quit)")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            server.shutdown()
    return 0


def print_bench_table(manifest: dict) -> None:
    from statistics import median

    rows: dict[tuple[str, str], list[dict]] = {}
    for e in manifest["episodes"]:
        rows.setdefault((e["task"], e["model"]), []).append(e)
    print(f"\n{'task':<20} {'model':<22} {'succeeded':>10} {'turns':>6} {'seconds':>8} {'tokens in':>10}  other")
    for (task, model), eps in rows.items():
        # Only episodes the model played to a verdict count; errors and stops are listed beside them.
        done = [e for e in eps if e["status"] in ("succeeded", "failed", "ended")]
        ok = sum(e["status"] == "succeeded" for e in done)
        med = lambda k: median([e[k] for e in done if e.get(k) is not None] or [0])  # noqa: E731
        tok = [(e.get("usage") or {}).get("input_tokens") or 0 for e in done]
        other = ", ".join(f"{sum(e['status'] == s for e in eps)} {s}" for s in ("error", "stopped", "pending")
                          if any(e["status"] == s for e in eps))
        print(f"{task:<20} {model:<22} {f'{ok}/{len(done)}':>10} {med('turns'):>6} {med('seconds'):>8} "
              f"{(round(sum(tok) / len(tok)) if tok else 0):>10}  {other}")


if __name__ == "__main__":
    sys.exit(main())

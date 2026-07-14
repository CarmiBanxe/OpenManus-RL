#!/usr/bin/env python3
"""CLI прогона eval-харнесса (S16). Детерминированные наборы + опц. --live."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openmanus_rl.eval import EvalHarness, to_json, to_markdown  # noqa: E402
from openmanus_rl.eval.suites import build_default_suites, build_live_llm_suite  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenManus component eval harness")
    ap.add_argument("--live", action="store_true", help="add live-LLM suite (hits gateway)")
    ap.add_argument("--model", default="smart")
    ap.add_argument("--format", choices=["md", "json"], default="md")
    ap.add_argument("--out", default=None)
    ap.add_argument("--metrics", action="store_true", help="record latency to observability")
    args = ap.parse_args()

    suites = build_default_suites()
    if args.live:
        suites.append(build_live_llm_suite(args.model))

    report = EvalHarness(record_metrics=args.metrics).run(suites)
    text = to_json(report) if args.format == "json" else to_markdown(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"written -> {args.out}")
    else:
        print(text)

    # Гейт: все ДЕТЕРМИНИРОВАННЫЕ кейсы должны пройти (live не влияет на exit-код).
    det = [r for r in report.results if r.component != "live_llm"]
    ok = all(r.passed for r in det)
    print(f"\n{'✅ EVAL PASSED' if ok else '❌ EVAL FAILED'} "
          f"({sum(r.passed for r in det)}/{len(det)} deterministic)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

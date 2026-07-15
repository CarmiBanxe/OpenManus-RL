#!/usr/bin/env python3
"""CLI прогона eval-харнесса (S16). Детерминированные наборы + опц. --live."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openmanus_rl.eval import EvalHarness, RegressionThresholds, to_json, to_markdown  # noqa: E402
from openmanus_rl.eval.harness import EvalReport  # noqa: E402
from openmanus_rl.eval.suites import LIVE_COMPONENTS, build_default_suites, build_live_suites  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenManus component eval harness")
    ap.add_argument("--live", action="store_true", help="add live suites (gateway/Ollama)")
    ap.add_argument("--model", default="smart")
    ap.add_argument("--format", choices=["md", "json"], default="md")
    ap.add_argument("--out", default=None)
    ap.add_argument("--metrics", action="store_true", help="record latency to observability")
    ap.add_argument("--min-success", type=float, default=1.0, help="regression gate: min success rate")
    ap.add_argument("--max-p95", type=float, default=None, help="regression gate: max p95 latency (s)")
    ap.add_argument("--max-mean", type=float, default=None, help="regression gate: max mean latency (s)")
    args = ap.parse_args()

    suites = build_default_suites()
    if args.live:
        suites.extend(build_live_suites(args.model))

    report = EvalHarness(record_metrics=args.metrics).run(suites)
    text = to_json(report) if args.format == "json" else to_markdown(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"written -> {args.out}")
    else:
        print(text)

    # Гейт по ДЕТЕРМИНИРОВАННЫМ кейсам (live не влияет на exit-код) + пороги регрессии.
    det = EvalReport([r for r in report.results if r.component not in LIVE_COMPONENTS])
    th = RegressionThresholds(min_success_rate=args.min_success,
                              max_p95_latency_s=args.max_p95, max_mean_latency_s=args.max_mean)
    th_ok, violations = th.check(det)
    ok = all(r.passed for r in det.results) and th_ok
    for vi in violations:
        print(f"  ⚠ threshold: {vi}")
    print(f"\n{'✅ EVAL PASSED' if ok else '❌ EVAL FAILED'} "
          f"({det.summary['passed']}/{det.summary['count']} deterministic, "
          f"success {det.summary['success_rate'] * 100:.0f}%)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

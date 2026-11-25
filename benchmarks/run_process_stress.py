"""Repeat lifecycle tests without retries; any failure makes the run fail."""

from __future__ import annotations

import argparse
import faulthandler
import json
import platform
import time
import unittest
from pathlib import Path


CASES = (
    "test_cancel_episode_kills_active_and_suppresses_old_result",
    "test_scheduler_expiry_kills_active_call_and_preserves_new_request",
    "test_restart_budget_stops_repeated_faults_and_explicit_reset_rearms",
    "test_close_terminates_hung_process_and_reaps_it",
    "test_ipc_thread_owns_connection_close",
    "test_slow_ipc_exit_does_not_lose_request_after_episode_reset",
    "test_stuck_transport_reports_queued_failure_and_cannot_be_reset",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/process_stress"))
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from tests.test_process_backend import ProcessBackendTests

    rows = []
    started = time.perf_counter()
    with (args.output_dir / "tests.log").open("w", encoding="utf-8") as log:
        runner = unittest.TextTestRunner(stream=log, verbosity=2, failfast=True)
        for iteration in range(1, args.iterations + 1):
            result = runner.run(unittest.TestSuite(ProcessBackendTests(name) for name in CASES))
            row = {"iteration": iteration, "tests": result.testsRun, "passed": result.wasSuccessful(),
                   "failures": len(result.failures), "errors": len(result.errors)}
            rows.append(row)
            log.flush()
            print(json.dumps(row), flush=True)
            report = {"python": platform.python_version(), "platform": platform.platform(),
                      "requested_iterations": args.iterations, "rows": rows,
                      "passed": all(r["passed"] for r in rows) and len(rows) == args.iterations,
                      "elapsed_s": time.perf_counter() - started}
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            if not result.wasSuccessful():
                faulthandler.dump_traceback(file=log)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

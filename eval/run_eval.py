"""
Run evaluation against all test cases and save a report.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --guardrail-only
    python -m eval.run_eval --orchestrator-only
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the fashion agent")
    parser.add_argument("--guardrail-only", action="store_true", help="Only evaluate the guardrail")
    parser.add_argument("--orchestrator-only", action="store_true", help="Only evaluate the orchestrator")
    args = parser.parse_args()

    run_guardrail = not args.orchestrator_only
    run_orchestrator = not args.guardrail_only

    from eval.evaluator import run_evaluation, save_report
    from eval.test_cases import GUARDRAIL_TEST_CASES, ORCHESTRATOR_TEST_CASES

    report = run_evaluation(
        orchestrator_cases=ORCHESTRATOR_TEST_CASES if run_orchestrator else [],
        guardrail_cases=GUARDRAIL_TEST_CASES if run_guardrail else [],
    )
    save_report(report)


if __name__ == "__main__":
    main()

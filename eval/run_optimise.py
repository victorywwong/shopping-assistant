"""
Run GEPA prompt optimisation and save results to eval/optimised/.

Usage:
    python -m eval.run_optimise
    python -m eval.run_optimise --guardrail-only
    python -m eval.run_optimise --orchestrator-only
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
    parser = argparse.ArgumentParser(description="Optimise fashion agent prompts with GEPA")
    parser.add_argument("--guardrail-only", action="store_true", help="Only optimise the guardrail prompt")
    parser.add_argument("--orchestrator-only", action="store_true", help="Only optimise the orchestrator prompt")
    args = parser.parse_args()

    run_guardrail = not args.orchestrator_only
    run_orchestrator = not args.guardrail_only

    from eval.optimizer import optimise_guardrail, optimise_orchestrator

    if run_guardrail:
        print("\n=== Optimising Guardrail Prompt (GEPA) ===")
        guardrail_prompt = optimise_guardrail()
        print("\nOptimised guardrail instruction:\n")
        print(guardrail_prompt)

    if run_orchestrator:
        print("\n=== Optimising Orchestrator Prompt (GEPA) ===")
        orchestrator_prompt = optimise_orchestrator()
        print("\nOptimised orchestrator instruction:\n")
        print(orchestrator_prompt)

    print("\nOptimised prompts saved to eval/optimised/")
    print("Review and copy into agents/guardrails.py and agents/orchestrator.py as appropriate.")


if __name__ == "__main__":
    main()

"""
Prompt optimisation using GEPA (via DSPy).

Two independent optimisation targets:
  1. guardrail   — optimises GuardrailSignature instruction in agents/guardrails.py
  2. orchestrator — optimises FashionAssistantSignature instruction in agents/orchestrator.py

GEPA is chosen over MIPROv2 because:
  - Our eval has diverse, conflicting objectives (clarification vs refusal vs context)
    which benefit from Pareto-frontier exploration
  - The metric returns natural language feedback that GEPA's reflection_lm uses
    to generate targeted prompt mutations
"""

import os
import random
from pathlib import Path

import dspy
from dotenv import load_dotenv

from eval.dspy_modules import (
    FashionAssistantModule,
    FashionAssistantSignature,
    GuardrailModule,
    GuardrailSignature,
    configure_dspy,
    extract_instruction,
    save_optimised_prompt,
)
from eval.test_cases import (
    GUARDRAIL_TEST_CASES,
    ORCHESTRATOR_TEST_CASES,
    Criterion,
)

load_dotenv()

OPTIMISED_DIR = Path(__file__).parent / "optimised"
OPTIMISED_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Guardrail metric (score + natural language feedback for GEPA)
# ---------------------------------------------------------------------------

def guardrail_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):  # noqa: ARG001
    expected = example.expected
    predicted = getattr(prediction, "classification", "").strip().lower()

    if predicted == expected:
        return dspy.Prediction(
            score=1.0,
            feedback=f"Correct: classified '{example.user_message[:50]}' as '{predicted}'.",
        )
    return dspy.Prediction(
        score=0.0,
        feedback=(
            f"Incorrect: expected '{expected}' but got '{predicted}' "
            f"for message: \"{example.user_message}\". "
            f"Review the boundary between {expected} and {predicted} cases."
        ),
    )


# ---------------------------------------------------------------------------
# Orchestrator metric (score + feedback for GEPA)
# ---------------------------------------------------------------------------

def _check_criteria(conversation: str, response: str, criteria: list[Criterion]) -> tuple[float, str]:
    """Rule-based checks where possible; LLM judge for behavioural criteria."""
    import re
    from data.catalog import all_articles
    from openai import AzureOpenAI

    valid_ids = set(all_articles()["article_id"].astype(str).tolist())
    cited_ids = re.findall(r"\b0\d{9}\b", response)
    invalid_ids = [aid for aid in cited_ids if aid not in valid_ids]

    failures: list[str] = []
    passing: int = 0

    for criterion in criteria:
        if criterion in (Criterion.VALID_ARTICLE_IDS, Criterion.NO_HALLUCINATED_IDS):
            if invalid_ids:
                failures.append(f"{criterion.value}: hallucinated IDs {invalid_ids}")
            else:
                passing += 1
            continue

        # LLM judge for behavioural criteria
        client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        from eval.evaluator import CRITERION_DESCRIPTIONS, JUDGE_PROMPT
        import json

        prompt = JUDGE_PROMPT.format(
            conversation=conversation,
            response=response,
            criterion=criterion.value,
            description=CRITERION_DESCRIPTIONS[criterion],
        )
        raw = client.chat.completions.create(
            model=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5.4"),
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=80,
            temperature=0,
        )
        try:
            result = json.loads(raw.choices[0].message.content.strip())
            if result.get("pass"):
                passing += 1
            else:
                failures.append(f"{criterion.value}: {result.get('reason', '')}")
        except Exception:
            failures.append(f"{criterion.value}: judge parse error")

    score = passing / len(criteria) if criteria else 1.0
    feedback = "; ".join(failures) if failures else "All criteria passed."
    return score, feedback


def orchestrator_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):  # noqa: ARG001
    response = getattr(prediction, "response", "") or ""
    criteria = example.criteria
    conversation = example.conversation_history

    score, feedback = _check_criteria(conversation, response, criteria)

    return dspy.Prediction(
        score=score,
        feedback=feedback if score < 1.0 else f"All criteria passed for: '{example.conversation_history[:60]}'",
    )


# ---------------------------------------------------------------------------
# Build DSPy Examples
# ---------------------------------------------------------------------------

def _build_guardrail_examples() -> list[dspy.Example]:
    examples = []
    for tc in GUARDRAIL_TEST_CASES:
        examples.append(
            dspy.Example(
                user_message=tc.user_message,
                expected=tc.expected,
            ).with_inputs("user_message")
        )
    return examples


def _build_orchestrator_examples() -> list[dspy.Example]:
    """
    For each test case, build one Example per turn using cumulative conversation history.
    Only turns with criteria are included as optimisation targets.
    """
    examples = []
    for tc in ORCHESTRATOR_TEST_CASES:
        conversation_lines: list[str] = []
        for turn in tc.turns:
            if turn.criteria:
                examples.append(
                    dspy.Example(
                        conversation_history="\n".join(conversation_lines)
                        + f"\nUser: {turn.user_message}",
                        criteria=turn.criteria,
                    ).with_inputs("conversation_history")
                )
            conversation_lines.append(f"User: {turn.user_message}")
            conversation_lines.append(f"Assistant: [response]")
    return examples


# ---------------------------------------------------------------------------
# Optimise guardrail
# ---------------------------------------------------------------------------

def optimise_guardrail(
    max_metric_calls: int = 150,
    num_threads: int = 4,
) -> str:
    configure_dspy(os.getenv("AZURE_CHAT_MINI_DEPLOYMENT", "gpt-5.4-mini"))

    examples = _build_guardrail_examples()
    random.shuffle(examples)
    train, val = examples[:12], examples[12:]

    reflection_lm = dspy.LM(
        model=f"azure/{os.getenv('AZURE_CHAT_DEPLOYMENT', 'gpt-5.4')}",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=1.0,
    )

    optimizer = dspy.GEPA(
        metric=guardrail_metric,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        use_merge=True,
        max_merge_invocations=5,
        num_threads=num_threads,
        use_wandb=True,
        wandb_api_key=os.getenv("WANDB_API_KEY"),
        wandb_init_kwargs={
            "project": "digitalgenius-fashion-assistant",
            "name": "guardrail-gepa",
            "tags": ["guardrail", "gepa"],
        },
    )

    module = GuardrailModule()
    optimised = optimizer.compile(module, trainset=train, valset=val)
    instruction = extract_instruction(optimised, GuardrailSignature)

    out_path = str(OPTIMISED_DIR / "guardrail_prompt.txt")
    save_optimised_prompt(instruction, out_path)
    return instruction


# ---------------------------------------------------------------------------
# Optimise orchestrator
# ---------------------------------------------------------------------------

def optimise_orchestrator(
    max_metric_calls: int = 200,
    num_threads: int = 4,
) -> str:
    configure_dspy(os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5.4"))

    examples = _build_orchestrator_examples()
    random.shuffle(examples)
    split = int(len(examples) * 0.75)
    train, val = examples[:split], examples[split:]

    reflection_lm = dspy.LM(
        model=f"azure/{os.getenv('AZURE_CHAT_DEPLOYMENT', 'gpt-5.4')}",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=1.0,
    )

    optimizer = dspy.GEPA(
        metric=orchestrator_metric,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        use_merge=True,
        max_merge_invocations=5,
        num_threads=num_threads,
        use_wandb=True,
        wandb_api_key=os.getenv("WANDB_API_KEY"),
        wandb_init_kwargs={
            "project": "digitalgenius-fashion-assistant",
            "name": "orchestrator-gepa",
            "tags": ["orchestrator", "gepa"],
        },
    )

    module = FashionAssistantModule()
    optimised = optimizer.compile(module, trainset=train, valset=val)
    instruction = extract_instruction(optimised, FashionAssistantSignature)

    out_path = str(OPTIMISED_DIR / "orchestrator_prompt.txt")
    save_optimised_prompt(instruction, out_path)
    return instruction

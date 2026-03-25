# Evaluation & Prompt Optimisation Plan

---

## Overview

Two complementary systems:

1. **Automated Evaluation** — runs predefined conversation scenarios against the live agent, scores each response with an LLM-as-judge, and produces a structured pass/fail report
2. **Automated Prompt Optimisation** — uses evaluation failures to iteratively improve the orchestrator's system prompt, re-evaluates, and tracks improvement across iterations

---

## Directory Structure

```
eval/
├── __init__.py
├── test_cases.py       # Scenario definitions
├── evaluator.py        # Runs scenarios, scores with LLM-as-judge
├── optimizer.py        # Iterative prompt improvement loop
└── run_eval.py         # CLI entry point: python -m eval.run_eval
```

---

## Part 1 — Automated Evaluation

### Test Case Structure (`test_cases.py`)

Each test case defines:
- A multi-turn conversation (list of user messages)
- The expected behaviours to check at each turn

```python
@dataclass
class Turn:
    user_message: str
    expect_clarification: bool = False   # agent should ask a question
    expect_product_ids: bool = False     # agent should cite article_ids
    expect_refusal: bool = False         # agent should decline to answer

@dataclass
class TestCase:
    name: str
    turns: list[Turn]
    description: str
```

**Scenarios covered:**

| Scenario | Turns | What is tested |
|---|---|---|
| Specific product query | 1 | "Do you have Howie shorts in blue?" — returns real IDs |
| Vague query | 2 | "I want new clothes" → clarification → recommendation |
| Multi-turn outfit building | 4 | Trousers + top, context maintained across turns |
| Out-of-scope request | 1 | "Write my CV" → refusal |
| No-results scenario | 1 | Very specific non-existent item → honest response |
| Image-based query | 1 | Image upload → product recommendations |
| Context retention | 3 | Preferences given early are used in later turns |

---

### Evaluator (`evaluator.py`)

Runs each scenario against the live agent and scores responses using **gpt-5.4 as judge**.

**Per-turn scoring criteria (each 0 or 1):**

| Criterion | Check |
|---|---|
| `clarification_asked` | Response contains a question when `expect_clarification=True` |
| `valid_article_ids` | All cited article IDs exist in the catalog |
| `refusal_correct` | Response refuses without providing products when `expect_refusal=True` |
| `no_hallucinated_ids` | No article IDs cited that don't exist in the catalog |
| `context_maintained` | LLM judge checks that earlier-stated preferences are reflected |

**LLM-as-judge prompt pattern:**
```
You are evaluating a fashion assistant's response.

Conversation so far: {conversation}
Assistant response: {response}
Criterion: {criterion_description}

Answer with JSON: {"pass": true/false, "reason": "..."}
```

**Output per run:**
```
=== Evaluation Report ===
Timestamp: 2026-03-24 21:00:00

Scenario: vague_query                   PASS  (3/3 criteria)
Scenario: specific_product_query        PASS  (2/2 criteria)
Scenario: out_of_scope                  PASS  (1/1 criteria)
Scenario: multi_turn_outfit             FAIL  (5/6 criteria)
  ✗ context_maintained @ turn 3: agent re-asked for colour already given

Overall: 11/12 (91.7%)
```

Results are saved to `eval/results/YYYY-MM-DD_HH-MM.json` for trend tracking.

---

## Part 2 — Automated Prompt Optimisation

### Approach: Iterative LLM-Driven Refinement (`optimizer.py`)

Not DSPy (too heavy a dependency for this scope). Instead: a focused loop that feeds evaluation failures back to gpt-5.4 and asks it to improve the system prompt.

**Loop:**

```
1. Run full evaluation suite → collect failures
2. If all pass OR max_iterations reached → stop
3. Feed failures to gpt-5.4:
     "Here is the current system prompt.
      Here are the test cases that failed, with the agent's actual responses.
      Rewrite the system prompt to address these failures.
      Return only the new system prompt."
4. Patch orchestrator with new prompt
5. Re-run evaluation → compare scores
6. Keep new prompt only if score improves
7. Go to 1
```

**Safeguards:**
- Max 5 iterations (prevents runaway cost)
- Prompt diff is logged each iteration so changes are auditable
- If a new prompt regresses overall score, it is discarded and the loop stops

**Output:**
```
=== Prompt Optimisation Run ===

Iteration 1: 11/12 (91.7%) — 1 failure
  → Generating improved prompt...
  → Diff: added "do not re-ask for colour if already mentioned in conversation"

Iteration 2: 12/12 (100%) — 0 failures
  → All scenarios pass. Optimisation complete.

Final prompt written to: eval/optimised_prompt.txt
```

---

## CLI Entry Point (`run_eval.py`)

```bash
# Run evaluation only
python -m eval.run_eval

# Run evaluation + prompt optimisation
python -m eval.run_eval --optimise

# Run with a specific prompt file
python -m eval.run_eval --prompt eval/optimised_prompt.txt
```

---

## Phased Delivery

| Phase | Deliverable |
|---|---|
| 1 | `test_cases.py` — all 7 scenarios defined |
| 2 | `evaluator.py` — LLM-as-judge scoring, JSON report output |
| 3 | `optimizer.py` — iterative refinement loop |
| 4 | `run_eval.py` — CLI wiring + result persistence |

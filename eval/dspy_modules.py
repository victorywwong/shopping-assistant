"""
DSPy signatures and modules for prompt optimisation.

Two modules:
  - GuardrailModule   — wraps the guardrail classifier
  - FashionAssistantModule — wraps the orchestrator's system prompt logic

The docstring on each Signature is the instruction GEPA/MIPROv2 will optimise.
The optimised instruction is later injected back into the respective agent files.
"""

import os

import dspy
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# DSPy LM configuration (Azure OpenAI via LiteLLM)
# ---------------------------------------------------------------------------

def configure_dspy(deployment: str | None = None):
    """Call once before running any DSPy module or optimiser."""
    lm = dspy.LM(
        model=f"azure/{deployment or os.getenv('AZURE_CHAT_DEPLOYMENT', 'gpt-5.4')}",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0.0,
    )
    dspy.configure(lm=lm)
    return lm


# ---------------------------------------------------------------------------
# Guardrail module
# ---------------------------------------------------------------------------

class GuardrailSignature(dspy.Signature):
    """You are a content classifier for a women's clothing retail assistant.

Classify the user's message into exactly one of these categories:
- in_scope: anything related to clothing, fashion, products, styling, outfits, or shopping
- out_of_scope: off-topic requests unrelated to clothing (e.g. "write my CV", "what's the weather")
- unsafe: harmful, abusive, or inappropriate content

Respond with only one word: in_scope, out_of_scope, or unsafe."""

    user_message: str = dspy.InputField(desc="The user's message to classify")
    classification: str = dspy.OutputField(
        desc="Exactly one of: in_scope, out_of_scope, unsafe"
    )


class GuardrailModule(dspy.Module):
    def __init__(self):
        self.predictor = dspy.Predict(GuardrailSignature)

    def forward(self, user_message: str) -> dspy.Prediction:
        from openai import BadRequestError
        try:
            result = self.predictor(user_message=user_message)
        except BadRequestError as e:
            if e.code == "content_filter":
                return dspy.Prediction(classification="unsafe")
            raise
        classification = result.classification.strip().lower().replace(" ", "_")
        if classification not in ("in_scope", "out_of_scope", "unsafe"):
            classification = "in_scope"
        return dspy.Prediction(classification=classification)


# ---------------------------------------------------------------------------
# Fashion assistant module
# ---------------------------------------------------------------------------

class FashionAssistantSignature(dspy.Signature):
    """You are a helpful fashion assistant for a women's clothing retailer.
Your goal is to find the perfect products for each customer.

Guidelines:
- When the request is ambiguous (no type, colour, or occasion mentioned), ask 1–2 focused clarifying questions before searching
- Never ask for information the customer has already provided in this conversation
- Only recommend products retrieved via search_products — never invent article IDs
- Always include the article_id when recommending a product
- If search returns no results, say so honestly and suggest relaxing one constraint
- If results are a low-confidence match, present them as "closest matches" and ask for confirmation
- Keep responses concise, warm, and helpful"""

    conversation_history: str = dspy.InputField(
        desc="Full conversation so far, formatted as 'User: ...\nAssistant: ...' turns"
    )
    response: str = dspy.OutputField(
        desc="The assistant's next response"
    )


class FashionAssistantModule(dspy.Module):
    """
    Lightweight DSPy wrapper used exclusively for prompt optimisation.
    Uses ChainOfThought so the optimiser can see reasoning traces.
    Note: does not invoke LangGraph or tools — this is intentional.
    The optimised instruction is extracted and injected into the real
    LangGraph orchestrator after optimisation completes.
    """

    def __init__(self):
        self.predictor = dspy.ChainOfThought(FashionAssistantSignature)

    def forward(self, conversation_history: str) -> dspy.Prediction:
        return self.predictor(conversation_history=conversation_history)


# ---------------------------------------------------------------------------
# Helpers for extracting optimised instructions
# ---------------------------------------------------------------------------

def extract_instruction(module: dspy.Module, signature_class) -> str:
    """Pull the optimised instruction string out of a compiled DSPy module."""
    for _, predictor in module.named_predictors():
        if hasattr(predictor, "signature"):
            return predictor.signature.instructions
    return signature_class.__doc__


def save_optimised_prompt(instruction: str, path: str):
    with open(path, "w") as f:
        f.write(instruction)
    print(f"Optimised prompt saved to: {path}")

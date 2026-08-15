
"""
LLM boundary.

The core evaluation does not require a network call. `DeterministicLLM` makes
generation reproducible while preserving the same constrained interface that an
external instruction-tuned model would receive.

An external provider can be attached later without changing detection/policy.
"""
import re

class DeterministicLLM:
    def generate(self, scaffold, event):
        # The scaffold is educator-authored; this mock simply instantiates it.
        return scaffold["text"]

def build_prompt(scaffold, event):
    return {
        "role": "You are a pedagogical coach, not a co-creator.",
        "phase": event["phase"],
        "function": scaffold["function"],
        "permitted": ["invite contribution", "ask reflection", "surface perspectives"],
        "forbidden": ["choose an idea", "rank ideas", "generate an idea"],
        "learner_agency_rule": (
            "Do not select, rank, decide, resolve, or generate on behalf of the group."
        ),
        "task": "Produce one concise coaching intervention."
    }

def validate_message(message, scaffold):
    errors = []
    lower = message.lower()
    forbidden_phrases = {
        "generate an idea","here is the best idea","the solution is",
        "you should choose","i recommend choosing","the correct answer is",
        "i would choose","the best option is"
    }
    for phrase in forbidden_phrases:
        if phrase in lower:
            errors.append(f"forbidden_phrase:{phrase}")
    if not message.strip():
        errors.append("empty")
    if len(message.split()) > 80:
        errors.append("too_long")
    if not message.rstrip().endswith((".", "?", "!")):
        errors.append("not_sentence")
    return not errors, tuple(errors)

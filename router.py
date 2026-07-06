"""Task routing: maps explicit task types to handlers, and classifies free-form
prompts into a TaskType using simple keyword rules. No LLM calls happen here.
"""
import re
from enum import Enum

from handlers import (
    code_handler,
    llm_handler,
    logic_handler,
    math_handler,
    ner_handler,
    sentiment_handler,
    summary_handler,
)

_ROUTES = {
    "math": math_handler.handle,
    "llm": llm_handler.handle,
    "sentiment": sentiment_handler.handle,
    "summary": summary_handler.handle,
    "ner": ner_handler.handle,
    "code": code_handler.handle,
    "logic": logic_handler.handle,
}


def route(task_type: str):
    try:
        return _ROUTES[task_type]
    except KeyError:
        raise ValueError(f"Unknown task type: {task_type}")


class TaskType(str, Enum):
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    NER = "ner"
    CODE_DEBUGGING = "code_debugging"
    LOGIC = "logic"
    CODE_GENERATION = "code_generation"


_SUMMARIZATION_KEYWORDS = ("summarize", "summarise", "summary", "condense", "one sentence", "bullet points")
_SENTIMENT_KEYWORDS = ("sentiment", "classify as positive", "negative", "neutral", "opinion", "review")
_NER_KEYWORDS = ("named entities", "extract entities", "person", "organization", "location", "date", "ner")
_CODE_DEBUGGING_KEYWORDS = ("debug", "bug", "fix this code", "error", "traceback", "exception", "broken code")
_CODE_GENERATION_KEYWORDS = ("write a function", "implement", "generate code", "create a function", "return code")
_LOGIC_KEYWORDS = ("logic puzzle", "deduce", "constraints", "who is", "which person", "exactly one", "all conditions")
_MATH_KEYWORDS = ("percent", "percentage", "calculate", "total", "cost", "increase", "decrease", "ratio")

_MATH_OPERATOR_PATTERN = re.compile(r"\d+\s*[-+*/^%]\s*\d+")


def _contains_any_keyword(text: str, keywords: tuple) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in keywords)


def looks_like_math(prompt: str) -> bool:
    if _MATH_OPERATOR_PATTERN.search(prompt):
        return True
    return _contains_any_keyword(prompt, _MATH_KEYWORDS)


def classify_task(prompt: str) -> TaskType:
    if _contains_any_keyword(prompt, _SUMMARIZATION_KEYWORDS):
        return TaskType.SUMMARIZATION
    if _contains_any_keyword(prompt, _SENTIMENT_KEYWORDS):
        return TaskType.SENTIMENT
    if _contains_any_keyword(prompt, _NER_KEYWORDS):
        return TaskType.NER
    if _contains_any_keyword(prompt, _CODE_DEBUGGING_KEYWORDS):
        return TaskType.CODE_DEBUGGING
    if _contains_any_keyword(prompt, _CODE_GENERATION_KEYWORDS):
        return TaskType.CODE_GENERATION
    if _contains_any_keyword(prompt, _LOGIC_KEYWORDS):
        return TaskType.LOGIC
    if looks_like_math(prompt):
        return TaskType.MATH
    return TaskType.FACTUAL

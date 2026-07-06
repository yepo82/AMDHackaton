"""Local math resolution: a safe AST evaluator for raw expressions, plus
try_solve_math() for simple natural-language arithmetic questions — solved
locally to save LLM tokens. No eval() and no LLM calls here.
"""
import ast
import math
import operator
import re
from typing import Optional

from .llm_handler import ask_llm

MATH_FALLBACK_INSTRUCTION = "Solve the math problem. Return the final answer concisely."
MATH_FALLBACK_MAX_TOKENS = 128

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def handle(task: dict, client=None) -> dict:
    expression = task.get("input", "")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
        return {"status": "ok", "result": result}
    except Exception as exc:
        return {"status": "error", "result": None, "error": str(exc)}


_NUM = r"(\d+(?:\.\d+)?)"
_PCT = r"(?:%|percent\b|percentage\b)"

_PRICE_CHANGE_PATTERN = re.compile(
    rf"{_NUM}.*?\b(increases|increase|decreases|decrease)\b.*?by\s*{_NUM}\s*{_PCT}",
    re.IGNORECASE | re.DOTALL,
)
_PERCENT_OF_PATTERN = re.compile(rf"{_NUM}\s*{_PCT}\s*of\s*{_NUM}", re.IGNORECASE)
_DIVIDED_BY_PATTERN = re.compile(rf"{_NUM}\s*divided\s*by\s*{_NUM}", re.IGNORECASE)
_TIMES_PATTERN = re.compile(rf"{_NUM}\s*(?:times|multiplied\s*by)\s*{_NUM}", re.IGNORECASE)
_PLUS_PATTERN = re.compile(rf"{_NUM}\s*(?:\+|plus|added\s*to)\s*{_NUM}", re.IGNORECASE)
_MINUS_PATTERN = re.compile(rf"{_NUM}\s*(?:-|minus|subtracted\s*from)\s*{_NUM}", re.IGNORECASE)
_MULTIPLY_SYMBOL_PATTERN = re.compile(rf"{_NUM}\s*\*\s*{_NUM}")
_DIVIDE_SYMBOL_PATTERN = re.compile(rf"{_NUM}\s*/\s*{_NUM}")
_AVERAGE_PATTERN = re.compile(rf"average\s+of\s+{_NUM}\s+and\s+{_NUM}", re.IGNORECASE)
_POWER_PATTERN = re.compile(rf"{_NUM}\s*(?:\^|to\s*the\s*power\s*of)\s*{_NUM}", re.IGNORECASE)
_SQUARE_ROOT_PATTERN = re.compile(rf"square\s*root\s*of\s*{_NUM}", re.IGNORECASE)
_SQUARED_PATTERN = re.compile(rf"{_NUM}\s*squared\b", re.IGNORECASE)


def _format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    text = f"{round(value, 6):.6f}".rstrip("0").rstrip(".")
    return text


def _try_price_change(prompt: str) -> Optional[str]:
    match = _PRICE_CHANGE_PATTERN.search(prompt)
    if not match:
        return None
    base_str, direction, percent_str = match.groups()
    base = float(base_str)
    delta = base * float(percent_str) / 100
    new_price = base + delta if direction.lower().startswith("increase") else base - delta
    return _format_number(new_price)


_SINGLE_ARG_OPS = (
    (_SQUARE_ROOT_PATTERN, lambda x: None if x < 0 else math.sqrt(x)),
    (_SQUARED_PATTERN, lambda x: x**2),
)

_TWO_ARG_OPS = (
    (_PERCENT_OF_PATTERN, lambda pct, base: pct / 100 * base),
    (_AVERAGE_PATTERN, lambda a, b: (a + b) / 2),
    (_POWER_PATTERN, lambda a, b: a**b),
    (_DIVIDED_BY_PATTERN, lambda a, b: None if b == 0 else a / b),
    (_TIMES_PATTERN, lambda a, b: a * b),
    (_PLUS_PATTERN, lambda a, b: a + b),
    (_MINUS_PATTERN, lambda a, b: a - b),
    (_MULTIPLY_SYMBOL_PATTERN, lambda a, b: a * b),
    (_DIVIDE_SYMBOL_PATTERN, lambda a, b: None if b == 0 else a / b),
)


def try_solve_math(prompt: str) -> Optional[str]:
    """Resolve a simple natural-language math question locally, or return None
    if it can't be solved with confidence."""
    price_change = _try_price_change(prompt)
    if price_change is not None:
        return price_change

    for pattern, op in _SINGLE_ARG_OPS:
        match = pattern.search(prompt)
        if match:
            result = op(float(match.group(1)))
            return None if result is None else _format_number(result)

    for pattern, op in _TWO_ARG_OPS:
        match = pattern.search(prompt)
        if match:
            a, b = (float(g) for g in match.groups())
            result = op(a, b)
            return None if result is None else _format_number(result)

    return None


def handle_math_fallback(prompt: str, client) -> str:
    return ask_llm(client, MATH_FALLBACK_INSTRUCTION, prompt, max_tokens=MATH_FALLBACK_MAX_TOKENS)

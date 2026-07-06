import pytest

from handlers import math_handler
from router import TaskType, classify_task, looks_like_math, route


def test_route_returns_known_handler():
    assert route("math") is math_handler.handle


def test_route_raises_for_unknown_type():
    with pytest.raises(ValueError):
        route("unknown_type")


def test_classify_summarization():
    prompt = "Please summarize this article in one sentence."
    assert classify_task(prompt) == TaskType.SUMMARIZATION


def test_classify_sentiment():
    prompt = "Classify the sentiment of this review as positive or negative."
    assert classify_task(prompt) == TaskType.SENTIMENT


def test_classify_ner():
    prompt = "Extract named entities such as person, organization and location from this text."
    assert classify_task(prompt) == TaskType.NER


def test_classify_code_debugging():
    prompt = "There is a bug in this code, please debug it. I keep getting a traceback."
    assert classify_task(prompt) == TaskType.CODE_DEBUGGING


def test_classify_code_generation():
    prompt = "Write a function that reverses a string."
    assert classify_task(prompt) == TaskType.CODE_GENERATION


def test_classify_logic():
    prompt = "Solve this logic puzzle: given these constraints, deduce who is telling the truth. Exactly one statement is false."
    assert classify_task(prompt) == TaskType.LOGIC


def test_classify_math():
    prompt = "If a shirt costs $50 and there's a 20 percent discount, calculate the total cost."
    assert classify_task(prompt) == TaskType.MATH


def test_classify_math_from_numeric_expression():
    prompt = "What is 45 + 78?"
    assert classify_task(prompt) == TaskType.MATH


def test_classify_math_from_bare_percent_sign():
    prompt = "What is 25% of 80?"
    assert classify_task(prompt) == TaskType.MATH


def test_classify_math_from_times_phrasing_not_in_keyword_list():
    prompt = "What is 15 times 6?"
    assert classify_task(prompt) == TaskType.MATH


def test_classify_math_from_divided_by_phrasing_not_in_keyword_list():
    prompt = "What is 120 divided by 5?"
    assert classify_task(prompt) == TaskType.MATH


def test_classify_math_from_average_phrasing():
    prompt = "What is the average of 10 and 20?"
    assert classify_task(prompt) == TaskType.MATH


def test_classify_math_from_square_root_phrasing():
    prompt = "What is the square root of 16?"
    assert classify_task(prompt) == TaskType.MATH


def test_classify_factual_fallback():
    prompt = "What is the capital of France?"
    assert classify_task(prompt) == TaskType.FACTUAL


def test_looks_like_math_detects_numeric_operator():
    assert looks_like_math("Compute 12 * 30 please.") is True


def test_looks_like_math_detects_keyword():
    assert looks_like_math("Calculate the total cost of the order.") is True


def test_looks_like_math_returns_false_for_non_math_text():
    assert looks_like_math("What is the capital of France?") is False

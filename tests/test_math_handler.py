from handlers.math_handler import MATH_FALLBACK_MAX_TOKENS, handle, try_solve_math


def test_simple_addition():
    result = handle({"input": "2 + 2"})
    assert result["status"] == "ok"
    assert result["result"] == 4


def test_operator_precedence():
    result = handle({"input": "3 * 4 - 5"})
    assert result["status"] == "ok"
    assert result["result"] == 7


def test_invalid_expression_returns_error():
    result = handle({"input": "import os"})
    assert result["status"] == "error"


def test_try_solve_math_percentage_of():
    assert try_solve_math("What is 25% of 80?") == "20"


def test_try_solve_math_addition():
    assert try_solve_math("Calculate 24 + 18") == "42"


def test_try_solve_math_division():
    assert try_solve_math("What is 120 divided by 5?") == "24"


def test_try_solve_math_multiplication():
    assert try_solve_math("What is 15 times 6?") == "90"


def test_try_solve_math_price_increase():
    prompt = "If a price is 80 and increases by 25%, what is the new price?"
    assert try_solve_math(prompt) == "100"


def test_try_solve_math_price_decrease():
    prompt = "If a price is 100 and decreases by 10%, what is the new price?"
    assert try_solve_math(prompt) == "90"


def test_try_solve_math_returns_none_for_non_math_prompt():
    assert try_solve_math("What is the capital of France?") is None


def test_try_solve_math_returns_none_for_ambiguous_prompt():
    assert try_solve_math("Compare 5 and 10.") is None


def test_try_solve_math_decimal_result_has_no_trailing_zeros():
    assert try_solve_math("What is 100 divided by 8?") == "12.5"


def test_try_solve_math_percentage_of_with_word_percent():
    assert try_solve_math("What is 25 percent of 80?") == "20"


def test_try_solve_math_price_increase_with_word_percent():
    prompt = "If a price is 80 and increases by 25 percent, what is the new price?"
    assert try_solve_math(prompt) == "100"


def test_try_solve_math_average_of_two_numbers():
    assert try_solve_math("What is the average of 10 and 20?") == "15"


def test_try_solve_math_square_root():
    assert try_solve_math("What is the square root of 16?") == "4"


def test_try_solve_math_squared():
    assert try_solve_math("What is 5 squared?") == "25"


def test_try_solve_math_power_word_form():
    assert try_solve_math("What is 2 to the power of 10?") == "1024"


def test_try_solve_math_power_symbol_form():
    assert try_solve_math("What is 2^10?") == "1024"


def test_try_solve_math_ignores_unrelated_percentile_mention():
    assert try_solve_math("Tell me about the 90th percentile of scores.") is None


def test_math_fallback_max_tokens_is_tight():
    assert MATH_FALLBACK_MAX_TOKENS <= 128

from handlers.math_handler import handle


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

import pytest

from handlers import math_handler
from router import route


def test_route_returns_known_handler():
    assert route("math") is math_handler.handle


def test_route_raises_for_unknown_type():
    with pytest.raises(ValueError):
        route("unknown_type")

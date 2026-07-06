"""Maps task types to their handler functions."""
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

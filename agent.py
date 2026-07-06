"""General-purpose agent: classifies a prompt and routes it to the right resolver.

The Fireworks config and client are created lazily -- only when a task
actually needs the LLM -- and reused across calls on the same agent
instance. Errors are not caught here; callers (e.g. main.py) decide how to
handle failures.
"""
from typing import Optional

from config import load_config
from fireworks_client import FireworksClient
from handlers.code_handler import handle_code_debugging, handle_code_generation
from handlers.llm_handler import handle_factual
from handlers.logic_handler import handle_logic
from handlers.math_handler import handle_math_fallback, try_solve_math
from handlers.ner_handler import handle_ner
from handlers.sentiment_handler import handle_sentiment
from handlers.summary_handler import handle_summary
from router import TaskType, classify_task

_LLM_HANDLERS = {
    TaskType.FACTUAL: handle_factual,
    TaskType.SENTIMENT: handle_sentiment,
    TaskType.SUMMARIZATION: handle_summary,
    TaskType.NER: handle_ner,
    TaskType.CODE_DEBUGGING: handle_code_debugging,
    TaskType.CODE_GENERATION: handle_code_generation,
    TaskType.LOGIC: handle_logic,
}


class GeneralPurposeAgent:
    def __init__(self):
        self._client: Optional[FireworksClient] = None

    @property
    def client(self) -> FireworksClient:
        if self._client is None:
            self._client = FireworksClient(load_config())
        return self._client

    def solve(self, prompt: str) -> str:
        task_type = classify_task(prompt)

        if task_type == TaskType.MATH:
            local_answer = try_solve_math(prompt)
            if local_answer is not None:
                return local_answer
            return handle_math_fallback(prompt, self.client)

        handler = _LLM_HANDLERS[task_type]
        return handler(prompt, self.client)

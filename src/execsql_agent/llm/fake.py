"""Deterministic queued language-model responses for tests and demos."""

from collections import deque
from collections.abc import Sequence

from execsql_agent.llm.base import LLMClient, LLMClientError
from execsql_agent.models import LLMError, LLMRequest, LLMResponse


class FakeLLMClient(LLMClient):
    """Return configured responses in order without network access."""

    def __init__(self, responses: Sequence[LLMResponse | LLMError]) -> None:
        self._responses = deque(responses)
        self.requests: list[LLMRequest] = []

    @property
    def remaining_responses(self) -> int:
        """Return the number of queued model turns."""

        return len(self._responses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Record the request and return the next deterministic response."""

        self.requests.append(request)
        if not self._responses:
            raise LLMClientError(
                LLMError(
                    code="fake_queue_exhausted",
                    message="FakeLLMClient response queue is exhausted.",
                    retryable=False,
                    attempts=1,
                )
            )
        response = self._responses.popleft()
        if isinstance(response, LLMError):
            raise LLMClientError(response)
        return response

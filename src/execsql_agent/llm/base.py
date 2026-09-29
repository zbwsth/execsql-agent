"""Abstract language-model boundary used by business modules."""

from abc import ABC, abstractmethod

from execsql_agent.models import LLMError, LLMRequest, LLMResponse


class LLMClientError(RuntimeError):
    """Domain exception carrying structured provider-independent details."""

    def __init__(self, error: LLMError) -> None:
        super().__init__(error.message)
        self.error = error


class LLMClient(ABC):
    """Provider-independent synchronous completion interface."""

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a normalized response or raise ``LLMClientError``."""

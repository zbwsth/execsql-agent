"""Application and domain configuration loading for ExecSQL-Agent."""

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictConfigModel(BaseModel):
    """Base configuration model that rejects misspelled fields."""

    model_config = ConfigDict(extra="forbid")


class PipelineConfig(StrictConfigModel):
    """Fixed Pipeline Agent settings."""

    max_steps: int = Field(default=3, ge=1)


class FunctionCallingConfig(StrictConfigModel):
    """Function Calling Agent settings reserved for the next phase."""

    max_steps: int = Field(default=5, ge=1)


class AgentConfig(StrictConfigModel):
    """Settings shared by agent modes."""

    empty_result_repair: bool = False


class ExecutorConfig(StrictConfigModel):
    """Limits used by the read-only SQLite executor."""

    max_rows: int = Field(default=100, ge=1)
    progress_handler_ops: int = Field(default=10_000, ge=1)
    progress_handler_max_callbacks: int = Field(default=1_000, ge=1)


class LLMConfig(StrictConfigModel):
    """Provider-independent model call limits."""

    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    retry_backoff_seconds: float = Field(default=0.0, ge=0)


class EvaluationConfig(StrictConfigModel):
    """Stable result-comparison defaults."""

    numeric_tolerance: float = Field(default=1e-6, ge=0)
    strict_columns: bool = False


class AppConfig(StrictConfigModel):
    """Top-level application configuration."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    function_calling: FunctionCallingConfig = Field(default_factory=FunctionCallingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


class DomainConfig(StrictConfigModel):
    """Database-specific context kept outside the domain-neutral Agent loop."""

    dataset_name: str
    database_path: str
    domain_description: str
    queryable_scope: list[str] = Field(default_factory=list)
    table_relationships: list[str] = Field(default_factory=list)
    value_notes: list[str] = Field(default_factory=list)
    query_guidelines: list[str] = Field(default_factory=list)
    unsupported_questions: list[str] = Field(default_factory=list)
    default_result_limit: int = Field(default=100, ge=1)

    def to_prompt(self) -> str:
        """Render concise model context without embedding database records."""

        sections = [
            f"Dataset: {self.dataset_name}",
            f"Domain: {self.domain_description}",
        ]
        groups = (
            ("Queryable scope", self.queryable_scope),
            ("Table relationships", self.table_relationships),
            ("Value notes", self.value_notes),
            ("Query guidelines", self.query_guidelines),
            ("Unsupported questions", self.unsupported_questions),
        )
        for title, values in groups:
            if values:
                sections.append(f"{title}:\n- " + "\n- ".join(values))
        sections.append(f"Maximum returned rows: {self.default_result_limit}")
        return "\n\n".join(sections)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate application configuration from a YAML file."""

    config_path = Path(path)
    raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")
    return AppConfig.model_validate(raw)


def load_domain_config(path: str | Path) -> DomainConfig:
    """Load a strict JSON domain description used only as model context."""

    config_path = Path(path)
    raw: object = json.loads(config_path.read_text(encoding="utf-8"))
    return DomainConfig.model_validate(raw)

"""VERL v0.9.0 BaseTool adapter over the frozen Harness-v2 ToolRegistry."""

from __future__ import annotations

import asyncio
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from execsql_agent.models import ToolCallRequest
from execsql_agent.tools.registry import ToolRegistry
from training.bird_phase_c import sha256_file
from training.build_bird_sft_dataset import _extract_database

TOOL_NAMES = {
    "list_tables",
    "inspect_schema",
    "validate_sql",
    "execute_sql",
}


class _HarnessParameters(BaseModel):
    """Lossless Harness JSON Schema parameters accepted by VERL's duck type."""

    model_config = ConfigDict(extra="allow")
    type: str
    properties: dict[str, Any]
    required: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)


class _HarnessFunctionSchema(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: str
    parameters: _HarnessParameters


class _HarnessToolSchema(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    function: _HarnessFunctionSchema


class BirdToolRegistryTool(BaseTool):
    """Route one VERL tool call through the existing trusted ToolRegistry."""

    def __init__(
        self,
        config: dict[str, Any],
        tool_schema: OpenAIFunctionToolSchema | _HarnessToolSchema | None,
    ) -> None:
        self._instances: dict[str, tuple[tempfile.TemporaryDirectory[str], ToolRegistry, str]] = {}
        super().__init__(config=config, tool_schema=tool_schema)

    def get_openai_tool_schema(self) -> _HarnessToolSchema:
        name = self.config.get("name")
        if name not in TOOL_NAMES:
            raise ValueError(f"Unsupported Harness-v2 tool name: {name!r}")
        registry = ToolRegistry(Path("/__execsql_schema_only__.sqlite"))
        definition = next(item for item in registry.definitions if item.name == name)
        return _HarnessToolSchema.model_validate(
            {
                "type": "function",
                "function": definition.model_dump(mode="json"),
            }
        )

    def _create_sync(
        self,
        create_kwargs: dict[str, Any],
    ) -> tuple[tempfile.TemporaryDirectory[str], ToolRegistry, str]:
        required = {
            "database_id",
            "database_path",
            "database_sha256",
            "database_archive_member",
        }
        if set(create_kwargs) != required:
            raise ValueError(f"create_kwargs must contain exactly {sorted(required)}")
        database_id = create_kwargs["database_id"]
        database_path = create_kwargs["database_path"]
        database_sha256 = create_kwargs["database_sha256"]
        archive_member = create_kwargs["database_archive_member"]
        if not isinstance(database_id, str) or Path(database_id).name != database_id:
            raise ValueError("invalid database_id")
        expected_member = f"train_databases/{database_id}/{database_id}.sqlite"
        expected_path = f"data/bird/train/runtime_databases/{database_id}/{database_id}.sqlite"
        if archive_member != expected_member:
            raise ValueError("database_archive_member does not match database_id")
        if database_path != expected_path:
            raise ValueError("database_path does not match trusted runtime layout")
        if not isinstance(database_sha256, str) or len(database_sha256) != 64:
            raise ValueError("database_sha256 must be a 64-character string")

        archive_path = Path(str(self.config["database_archive"])).resolve()
        temporary_root = Path(str(self.config["temporary_root"])).resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(f"Trusted BIRD archive not found: {archive_path}")
        temporary_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            holder, extracted = _extract_database(
                archive,
                db_id=database_id,
                temporary_root=temporary_root,
            )
        try:
            if sha256_file(extracted) != database_sha256.casefold():
                raise ValueError("trusted database SHA-256 does not match metadata")
            return holder, ToolRegistry(extracted), database_id
        except BaseException:
            holder.cleanup()
            raise

    async def create(
        self,
        instance_id: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, ToolResponse]:
        create_kwargs = kwargs.get("create_kwargs", {})
        if not isinstance(create_kwargs, dict):
            raise ValueError("create_kwargs must be an object")
        holder, registry, database_id = await asyncio.to_thread(
            self._create_sync,
            create_kwargs,
        )
        actual_id = instance_id or str(uuid4())
        self._instances[actual_id] = (holder, registry, database_id)
        return actual_id, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict[str, object]]:
        state = self._instances.get(instance_id)
        if state is None:
            raise KeyError(f"Unknown tool instance: {instance_id}")
        _, registry, database_id = state
        call = ToolCallRequest(
            id=f"verl_{uuid4().hex}",
            name=self.name,
            arguments=parameters,
        )
        validation, result = await asyncio.to_thread(registry.dispatch, call)
        agent_data = kwargs.get("agent_data")
        if self.name == "execute_sql" and agent_data is not None:
            agent_data.extra_fields["execsql_database_id"] = database_id
            agent_data.extra_fields["execsql_last_execute_sql"] = parameters.get("sql")
            agent_data.extra_fields["execsql_last_execute_result"] = result.model_dump(mode="json")
        metrics: dict[str, object] = {
            "database_id": database_id,
            "tool_name": self.name,
            "validation_valid": validation.valid,
            "success": result.success,
            "executed": result.executed,
        }
        return ToolResponse(text=result.model_dump_json()), 0.0, metrics

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        del kwargs
        state = self._instances.pop(instance_id, None)
        if state is not None:
            holder, _, _ = state
            await asyncio.to_thread(holder.cleanup)

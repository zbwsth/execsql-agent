"""Stable fixed-pipeline Text-to-SQL baseline."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from execsql_agent.diagnosis.error_diagnoser import DiagnosisError, ErrorDiagnoser
from execsql_agent.generation.sql_generator import SQLGenerationError, SQLGenerator
from execsql_agent.llm.base import LLMClientError
from execsql_agent.models import (
    DatabaseSchema,
    ErrorDiagnosis,
    ErrorType,
    ExecutionError,
    ExecutionResult,
    GenerationMode,
    LLMError,
    PipelineAgentResult,
    PipelineStep,
    PipelineTerminationReason,
)
from execsql_agent.tools.schema_loader import SchemaLoader
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator


class PipelineAgent:
    """Generate, execute, diagnose, and repair SQL in a fixed safe loop."""

    def __init__(
        self,
        database_path: str | Path,
        generator: SQLGenerator,
        *,
        diagnoser: ErrorDiagnoser | None = None,
        validator: SQLValidator | None = None,
        executor: SQLExecutor | None = None,
        max_steps: int = 3,
        empty_result_repair: bool = False,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.database_path = Path(database_path)
        self.generator = generator
        self.diagnoser = diagnoser or ErrorDiagnoser(generator.client)
        self.validator = validator or SQLValidator()
        self.executor = executor or SQLExecutor(self.database_path, validator=self.validator)
        self.max_steps = max_steps
        self.empty_result_repair = empty_result_repair

    def run(self, question: str) -> PipelineAgentResult:
        """Run the fixed pipeline until success or an explicit terminal condition."""

        try:
            schema = SchemaLoader(self.database_path).load()
        except (OSError, sqlite3.Error):
            return PipelineAgentResult(
                question=question,
                database_id=self.database_path.stem,
                schema_summary="",
                steps=[],
                protocol_completed=False,
                execution_success=False,
                answer_grounded=False,
                termination_reason=PipelineTerminationReason.UNRECOVERABLE_ERROR,
                total_steps=0,
            )

        steps: list[PipelineStep] = []
        history_sql: list[str] = []
        executed_normalized_sql: set[str] = set()
        previous_error: ExecutionError | str | None = None
        previous_diagnosis: ErrorDiagnosis | None = None
        unsafe_seen = False
        final_sql: str | None = None
        last_execution: ExecutionResult | None = None

        for step_index in range(1, self.max_steps + 1):
            mode = (
                GenerationMode.INITIAL_GENERATION
                if step_index == 1
                else GenerationMode.REPAIR
            )
            try:
                generation = self.generator.generate(
                    question=question,
                    schema=schema,
                    mode=mode,
                    history_sql=history_sql,
                    previous_error=previous_error,
                    diagnosis=previous_diagnosis,
                )
            except LLMClientError as error:
                steps.append(
                    PipelineStep(
                        step_index=step_index,
                        mode=mode,
                        llm_error=error.error,
                        observation=error.error.message,
                    )
                )
                return self._result(
                    question,
                    schema,
                    steps,
                    final_sql,
                    last_execution,
                    PipelineTerminationReason.MODEL_ERROR,
                )
            except SQLGenerationError as error:
                steps.append(
                    PipelineStep(
                        step_index=step_index,
                        mode=mode,
                        llm_error=LLMError(
                            code="invalid_generation",
                            message=str(error),
                            retryable=False,
                        ),
                        observation=str(error),
                    )
                )
                return self._result(
                    question,
                    schema,
                    steps,
                    final_sql,
                    last_execution,
                    PipelineTerminationReason.MODEL_ERROR,
                )

            final_sql = generation.sql
            history_sql.append(generation.sql)
            safety = self.validator.validate(generation.sql)

            if not safety.safe:
                diagnosis = self.diagnoser.diagnose(
                    sql=generation.sql,
                    schema=schema,
                    error_message=safety.reason or "Unsafe SQL was blocked.",
                    forced_error_type=ErrorType.UNSAFE_SQL,
                )
                steps.append(
                    PipelineStep(
                        step_index=step_index,
                        mode=mode,
                        generation=generation,
                        safety=safety,
                        diagnosis=diagnosis,
                        observation=safety.reason or "Unsafe SQL was blocked.",
                    )
                )
                if unsafe_seen:
                    return self._result(
                        question,
                        schema,
                        steps,
                        final_sql,
                        last_execution,
                        PipelineTerminationReason.UNSAFE_SQL,
                    )
                unsafe_seen = True
                previous_error = safety.reason or "Unsafe SQL was blocked."
                previous_diagnosis = diagnosis
                continue

            if safety.normalized_sql in executed_normalized_sql:
                diagnosis = self.diagnoser.diagnose(
                    sql=generation.sql,
                    schema=schema,
                    error_message="The normalized SQL has already been executed.",
                    forced_error_type=ErrorType.REPEATED_SQL,
                )
                steps.append(
                    PipelineStep(
                        step_index=step_index,
                        mode=mode,
                        generation=generation,
                        safety=safety,
                        diagnosis=diagnosis,
                        observation="Repeated normalized SQL was not executed.",
                    )
                )
                return self._result(
                    question,
                    schema,
                    steps,
                    final_sql,
                    last_execution,
                    PipelineTerminationReason.REPEATED_SQL,
                )

            executed_normalized_sql.add(safety.normalized_sql)
            execution = self.executor.execute(generation.sql)
            last_execution = execution
            if execution.execution_success:
                if self.empty_result_repair and execution.returned_row_count == 0:
                    diagnosis = self.diagnoser.diagnose(
                        sql=generation.sql,
                        schema=schema,
                        error_message="The query returned no rows.",
                        forced_error_type=ErrorType.EMPTY_RESULT,
                    )
                    steps.append(
                        PipelineStep(
                            step_index=step_index,
                            mode=mode,
                            generation=generation,
                            safety=safety,
                            execution=execution,
                            diagnosis=diagnosis,
                            observation="The query returned no rows; repair is enabled.",
                        )
                    )
                    previous_error = "The query returned no rows."
                    previous_diagnosis = diagnosis
                    continue

                steps.append(
                    PipelineStep(
                        step_index=step_index,
                        mode=mode,
                        generation=generation,
                        safety=safety,
                        execution=execution,
                        observation=(
                            f"Query succeeded and returned "
                            f"{execution.returned_row_count} row(s)."
                        ),
                    )
                )
                return self._result(
                    question,
                    schema,
                    steps,
                    final_sql,
                    execution,
                    PipelineTerminationReason.COMPLETED,
                )

            error_message = (
                execution.error.message
                if execution.error is not None
                else execution.blocked_reason or "Unknown execution failure."
            )
            try:
                diagnosis = self.diagnoser.diagnose(
                    sql=generation.sql,
                    schema=schema,
                    error_message=error_message,
                )
            except (LLMClientError, DiagnosisError) as error:
                llm_error = (
                    error.error
                    if isinstance(error, LLMClientError)
                    else LLMError(
                        code="invalid_diagnosis", message=str(error), retryable=False
                    )
                )
                steps.append(
                    PipelineStep(
                        step_index=step_index,
                        mode=mode,
                        generation=generation,
                        safety=safety,
                        execution=execution,
                        llm_error=llm_error,
                        observation=llm_error.message,
                    )
                )
                return self._result(
                    question,
                    schema,
                    steps,
                    final_sql,
                    execution,
                    PipelineTerminationReason.MODEL_ERROR,
                )

            steps.append(
                PipelineStep(
                    step_index=step_index,
                    mode=mode,
                    generation=generation,
                    safety=safety,
                    execution=execution,
                    diagnosis=diagnosis,
                    observation=error_message,
                )
            )
            previous_error = execution.error or error_message
            previous_diagnosis = diagnosis

        return self._result(
            question,
            schema,
            steps,
            final_sql,
            last_execution,
            PipelineTerminationReason.MAX_STEPS_REACHED,
        )

    @staticmethod
    def _result(
        question: str,
        schema: DatabaseSchema,
        steps: list[PipelineStep],
        final_sql: str | None,
        execution_result: ExecutionResult | None,
        termination_reason: PipelineTerminationReason,
    ) -> PipelineAgentResult:
        execution_success = bool(
            execution_result is not None and execution_result.execution_success
        )
        completed = termination_reason is PipelineTerminationReason.COMPLETED
        return PipelineAgentResult(
            question=question,
            database_id=schema.database_id,
            schema_summary=schema.summary_text,
            steps=steps,
            final_sql=final_sql,
            execution_result=execution_result,
            protocol_completed=completed,
            execution_success=execution_success,
            answer_grounded=completed and execution_success,
            result_correct=None,
            termination_reason=termination_reason,
            total_steps=len(steps),
        )

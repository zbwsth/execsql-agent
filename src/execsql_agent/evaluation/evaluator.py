"""Batch evaluation for Pipeline and Function Calling on real SQLite results."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Literal
from uuid import uuid4

from execsql_agent.agents.function_calling import FunctionCallingAgent
from execsql_agent.agents.pipeline import PipelineAgent
from execsql_agent.evaluation.comparator import (
    compare_bird_execution_result,
    compare_execution_result,
)
from execsql_agent.evaluation.metrics import calculate_mode_metrics, compare_modes
from execsql_agent.generation.sql_generator import SQLGenerator
from execsql_agent.llm.base import LLMClient
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.llm.openai_compatible import OpenAICompatibleLLMClient
from execsql_agent.models import (
    BehaviorScenario,
    BirdPredictionTimeoutPolicy,
    CaseEvaluation,
    EvaluationAgentMode,
    EvaluationCase,
    EvaluationCheckpointRecord,
    EvaluationReport,
    ExecutionResult,
    FailureKind,
    FunctionCallingAgentResult,
    LLMMessage,
    ModeMetrics,
    PipelineAgentResult,
    Trajectory,
)
from execsql_agent.tools.registry import ToolRegistry
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.trajectory.logger import TrajectoryLogger

ConcreteMode = Literal["pipeline", "function-calling"]
ClientFactory = Callable[[EvaluationCase, ConcreteMode], LLMClient]
DatabaseResolver = Callable[[EvaluationCase], Path]


class Evaluator:
    """Run isolated cases, compare real results, and aggregate auditable metrics."""

    def __init__(
        self,
        database_path: str | Path | None,
        client_factory: ClientFactory,
        *,
        trajectory_logger: TrajectoryLogger | None = None,
        domain_context: str | None = None,
        llm_backend: str = "FakeLLMClient",
        run_mode: str = "deterministic/mock",
        database_resolver: DatabaseResolver | None = None,
        use_evidence: bool = False,
        prediction_timeout_seconds: float = 30.0,
        prediction_timeout_policy: BirdPredictionTimeoutPolicy | None = None,
        max_agent_steps: int | None = None,
    ) -> None:
        if database_path is None and database_resolver is None:
            raise ValueError("database_path or database_resolver is required")
        if prediction_timeout_seconds <= 0:
            raise ValueError("prediction_timeout_seconds must be positive")
        if max_agent_steps is not None and max_agent_steps < 1:
            raise ValueError("max_agent_steps must be at least 1")
        self.database_path = Path(database_path) if database_path is not None else None
        self.client_factory = client_factory
        self.trajectory_logger = trajectory_logger
        self.domain_context = domain_context
        self.llm_backend = llm_backend
        self.run_mode = run_mode
        self.database_resolver = database_resolver
        self.use_evidence = use_evidence
        self.prediction_timeout_seconds = prediction_timeout_seconds
        self.prediction_timeout_policy = prediction_timeout_policy
        self.max_agent_steps = max_agent_steps

    def evaluate(
        self,
        cases: Sequence[EvaluationCase],
        *,
        dataset_name: str,
        agent_mode: EvaluationAgentMode | str,
        seed: int = 0,
        tags: Sequence[str] = (),
        limit: int | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        config_fingerprint: str | None = None,
    ) -> EvaluationReport:
        """Evaluate a filtered case set; one case failure never stops later cases."""

        mode = EvaluationAgentMode(agent_mode)
        selected = [
            case
            for case in cases
            if not tags or all(tag in case.tags for tag in tags)
        ]
        Random(seed).shuffle(selected)
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be at least 1")
            selected = selected[:limit]

        if mode is EvaluationAgentMode.BOTH:
            concrete_modes: list[ConcreteMode] = ["pipeline", "function-calling"]
        elif mode is EvaluationAgentMode.PIPELINE:
            concrete_modes = ["pipeline"]
        else:
            concrete_modes = ["function-calling"]
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
        completed = self._load_checkpoint(
            checkpoint,
            resume=resume,
            config_fingerprint=config_fingerprint,
        )
        evaluations: list[CaseEvaluation] = []
        for concrete_mode in concrete_modes:
            for case in selected:
                checkpoint_key = (case.id, concrete_mode)
                if checkpoint_key in completed:
                    evaluations.append(completed[checkpoint_key])
                    continue
                try:
                    evaluation = self._evaluate_case(
                        case, dataset_name, concrete_mode
                    )
                except Exception as error:
                    # The batch boundary deliberately records an isolated case crash and continues.
                    evaluation = self._exception_case(
                        case, dataset_name, concrete_mode, str(error)
                    )
                evaluations.append(evaluation)
                if checkpoint is not None:
                    if config_fingerprint is None:
                        raise ValueError(
                            "config_fingerprint is required with checkpoint_path"
                        )
                    self._append_checkpoint(
                        checkpoint,
                        EvaluationCheckpointRecord(
                            config_fingerprint=config_fingerprint,
                            case_id=case.id,
                            agent_mode=concrete_mode,
                            evaluation=evaluation,
                        ),
                    )

        metrics: dict[str, ModeMetrics] = {}
        for concrete_mode in concrete_modes:
            mode_cases = [
                case for case in evaluations if case.agent_mode == concrete_mode
            ]
            metrics[concrete_mode] = calculate_mode_metrics(
                mode_cases, include_tool_metrics=concrete_mode == "function-calling"
            )
        comparison = None
        if mode is EvaluationAgentMode.BOTH:
            comparison = compare_modes(
                metrics["pipeline"], metrics["function-calling"]
            )
        database_ids = sorted({case.database_id for case in selected})
        report_database_id = (
            self.database_path.stem
            if self.database_path is not None
            else database_ids[0]
            if len(database_ids) == 1
            else "multiple"
        )
        return EvaluationReport(
            dataset_name=dataset_name,
            database_id=report_database_id,
            agent_mode=mode,
            llm_backend=self.llm_backend,
            run_mode=self.run_mode,
            seed=seed,
            database_ids=database_ids,
            oracle_evidence=self.use_evidence,
            experiment_config_fingerprint=config_fingerprint,
            selected_tags=list(tags),
            cases=evaluations,
            mode_metrics=metrics,
            comparison=comparison,
        )

    def _evaluate_case(
        self,
        case: EvaluationCase,
        dataset_name: str,
        agent_mode: ConcreteMode,
    ) -> CaseEvaluation:
        database_path = self._database_path_for(case)
        client = self.client_factory(case, agent_mode)
        max_steps = self._max_steps(case, agent_mode)
        prompt_question = self._prompt_question(case)
        started_at = perf_counter()
        try:
            if agent_mode == "pipeline":
                pipeline_agent = PipelineAgent(
                    database_path,
                    SQLGenerator(client),
                    max_steps=max_steps,
                )
                pipeline_result = pipeline_agent.run(prompt_question)
                duration_ms = (perf_counter() - started_at) * 1000
                evaluation, trajectory = self._pipeline_evaluation(
                    case,
                    dataset_name,
                    database_path,
                    pipeline_result,
                    client,
                    duration_ms,
                )
            else:
                function_agent = FunctionCallingAgent(
                    database_path,
                    client,
                    max_steps=max_steps,
                    llm_backend=type(client).__name__,
                    run_mode=self.run_mode,
                    domain_context=self.domain_context,
                )
                function_result = function_agent.run(prompt_question)
                duration_ms = (perf_counter() - started_at) * 1000
                evaluation, trajectory = self._function_evaluation(
                    case,
                    dataset_name,
                    database_path,
                    function_result,
                    client,
                    duration_ms,
                )
        finally:
            if isinstance(client, OpenAICompatibleLLMClient):
                client.close()
        if self.trajectory_logger is not None:
            self.trajectory_logger.log(trajectory)
        return evaluation

    def _database_path_for(self, case: EvaluationCase) -> Path:
        if self.database_resolver is not None:
            return Path(self.database_resolver(case))
        if self.database_path is None:
            raise ValueError("No database path is configured.")
        return self.database_path

    def _prompt_question(self, case: EvaluationCase) -> str:
        if self.use_evidence and case.evidence:
            return f"{case.question}\n\nExternal knowledge evidence:\n{case.evidence}"
        return case.question

    def _max_steps(self, case: EvaluationCase, agent_mode: ConcreteMode) -> int:
        if self.max_agent_steps is not None:
            return self.max_agent_steps
        defaults = {"pipeline": 3, "function-calling": 5}
        if isinstance(case, BehaviorScenario):
            return case.max_steps.get(agent_mode, defaults[agent_mode])
        return defaults[agent_mode]

    @staticmethod
    def _load_checkpoint(
        path: Path | None,
        *,
        resume: bool,
        config_fingerprint: str | None,
    ) -> dict[tuple[str, ConcreteMode], CaseEvaluation]:
        if path is None:
            if resume:
                raise ValueError("resume requires checkpoint_path")
            return {}
        if config_fingerprint is None:
            raise ValueError("config_fingerprint is required with checkpoint_path")
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8")
        if text.strip() and not resume:
            raise ValueError(
                f"Evaluation checkpoint already exists; use --resume: {path}"
            )
        records: dict[tuple[str, ConcreteMode], CaseEvaluation] = {}
        lines = text.splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = EvaluationCheckpointRecord.model_validate_json(line)
            except ValueError as error:
                if line_number == len(lines) and not text.endswith("\n"):
                    break
                raise ValueError(
                    f"Invalid evaluation checkpoint at line {line_number}: {error}"
                ) from error
            if record.config_fingerprint != config_fingerprint:
                raise ValueError(
                    "Evaluation checkpoint config fingerprint does not match "
                    "the current experiment."
                )
            if (
                record.case_id != record.evaluation.case_id
                or record.agent_mode != record.evaluation.agent_mode
            ):
                raise ValueError(
                    f"Inconsistent evaluation checkpoint record: {record.case_id}"
                )
            key = (record.case_id, record.agent_mode)
            records[key] = record.evaluation
        return records

    @staticmethod
    def _append_checkpoint(
        path: Path, record: EvaluationCheckpointRecord
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(record.model_dump_json())
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _pipeline_evaluation(
        self,
        case: EvaluationCase,
        dataset_name: str,
        database_path: Path,
        result: PipelineAgentResult,
        client: LLMClient,
        duration_ms: float,
    ) -> tuple[CaseEvaluation, Trajectory]:
        executions = [
            step.execution
            for step in result.steps
            if step.execution is not None and step.execution.executed
        ]
        first_success = executions[0].execution_success if executions else None
        result_correct, scored_execution = self._score_result(
            case, database_path, result.final_sql, result.execution_result
        )
        actual_result = scored_execution or result.execution_result
        bird_facts = self._bird_score_facts(
            case, result_correct, scored_execution
        )
        execution_success = (
            bool(scored_execution and scored_execution.execution_success)
            if case.comparison_mode == "bird_set"
            else result.execution_success
        )
        message_history = self._client_message_history(client)
        trajectory = TrajectoryLogger.from_pipeline_result(
            result, llm_backend=type(client).__name__, run_mode=self.run_mode
        )
        unsafe_count = sum(
            step.safety is not None and not step.safety.safe for step in result.steps
        )
        failure = self._failure_kind(
            result_correct=result_correct,
            protocol_completed=result.protocol_completed,
            execution_success=execution_success,
            termination_reason=result.termination_reason.value,
            first_execution_success=first_success,
            invalid_tool_calls=0,
            comparison_mode=case.comparison_mode,
        )
        evaluation = CaseEvaluation(
            case_id=case.id,
            dataset_name=dataset_name,
            tags=case.tags,
            question=case.question,
            database_id=case.database_id,
            agent_mode="pipeline",
            protocol_completed=result.protocol_completed,
            execution_success=execution_success,
            answer_grounded=result.answer_grounded,
            result_correct=result_correct,
            difficulty=case.difficulty,
            comparison_mode=case.comparison_mode,
            expected_refusal=case.expected_refusal,
            refused=False,
            expected_unsafe_sql=case.expected_unsafe_sql,
            failure_kind=failure,
            termination_reason=result.termination_reason.value,
            final_sql=result.final_sql,
            expected_result=case.expected_result,
            gold_sql=case.gold_sql,
            gold_runtime_ms=case.gold_runtime_ms,
            prediction_timeout_budget_seconds=bird_facts[0],
            prediction_runtime_ms=bird_facts[1],
            prediction_timeout=bird_facts[2],
            stable_bird_ex=bird_facts[3],
            ex_at_30s=bird_facts[4],
            required_tools=case.required_tools,
            expected_tool_sequences=case.expected_tool_sequences,
            actual_result=actual_result,
            first_execution_success=first_success,
            repair_succeeded=(
                result.execution_success if first_success is False else None
            ),
            trajectory_id=trajectory.trajectory_id,
            total_duration_ms=duration_ms,
            total_llm_turns=len(result.steps),
            total_tool_calls=0,
            total_sql_executions=len(executions),
            invalid_tool_calls=0,
            repeated_tool_calls=0,
            unsafe_sql_count=unsafe_count,
            message_history=message_history,
            error_message=self._execution_error_message(actual_result),
        )
        trajectory = trajectory.model_copy(
            update={
                "dataset_name": dataset_name,
                "tags": case.tags,
                "result_correct": result_correct,
                "message_history": message_history,
                "unsafe_sql_count": unsafe_count,
            }
        )
        return evaluation, trajectory

    def _function_evaluation(
        self,
        case: EvaluationCase,
        dataset_name: str,
        database_path: Path,
        result: FunctionCallingAgentResult,
        client: LLMClient,
        duration_ms: float,
    ) -> tuple[CaseEvaluation, Trajectory]:
        calls = [call for step in result.steps for call in step.tool_calls]
        validations = [
            validation for step in result.steps for validation in step.validations
        ]
        tool_results = [
            tool_result for step in result.steps for tool_result in step.tool_results
        ]
        observations = [
            observation for step in result.steps for observation in step.tool_messages
        ]
        execute_results: list[ExecutionResult] = []
        for tool_result in tool_results:
            if tool_result.tool_name == "execute_sql":
                execution = ToolRegistry.execution_result(tool_result)
                if execution is not None:
                    execute_results.append(execution)
        executions = [execution for execution in execute_results if execution.executed]
        first_success = executions[0].execution_success if executions else None
        final_execution = execute_results[-1] if execute_results else None
        final_execution_success = bool(
            final_execution is not None and final_execution.execution_success
        )
        result_correct, scored_execution = self._score_result(
            case, database_path, result.final_sql, final_execution
        )
        actual_result = scored_execution or final_execution
        bird_facts = self._bird_score_facts(
            case, result_correct, scored_execution
        )
        evaluation_execution_success = (
            bool(scored_execution and scored_execution.execution_success)
            if case.comparison_mode == "bird_set"
            else final_execution_success
        )
        refused = bool(
            result.protocol_completed
            and not execute_results
            and result.final_answer
        )
        invalid_codes = {"unknown_tool", "invalid_json_arguments", "invalid_arguments"}
        invalid_count = sum(
            validation.error_code in invalid_codes for validation in validations
        )
        repeated_count = sum(validation.duplicate for validation in validations)
        unsafe_count = sum(
            tool_result.error_code == "unsafe_sql" for tool_result in tool_results
        )
        failure = self._failure_kind(
            result_correct=result_correct,
            protocol_completed=result.protocol_completed,
            execution_success=evaluation_execution_success,
            termination_reason=result.termination_reason.value,
            first_execution_success=first_success,
            invalid_tool_calls=invalid_count,
            comparison_mode=case.comparison_mode,
        )
        trajectory = TrajectoryLogger.from_function_result(
            result, llm_backend=type(client).__name__, run_mode=self.run_mode
        )
        message_history = trajectory.message_history
        evaluation = CaseEvaluation(
            case_id=case.id,
            dataset_name=dataset_name,
            tags=case.tags,
            question=case.question,
            database_id=case.database_id,
            agent_mode="function-calling",
            protocol_completed=result.protocol_completed,
            execution_success=evaluation_execution_success,
            answer_grounded=result.answer_grounded,
            result_correct=result_correct,
            difficulty=case.difficulty,
            comparison_mode=case.comparison_mode,
            expected_refusal=case.expected_refusal,
            refused=refused,
            expected_unsafe_sql=case.expected_unsafe_sql,
            failure_kind=failure,
            termination_reason=result.termination_reason.value,
            final_sql=result.final_sql,
            final_answer=result.final_answer,
            expected_result=case.expected_result,
            gold_sql=case.gold_sql,
            gold_runtime_ms=case.gold_runtime_ms,
            prediction_timeout_budget_seconds=bird_facts[0],
            prediction_runtime_ms=bird_facts[1],
            prediction_timeout=bird_facts[2],
            stable_bird_ex=bird_facts[3],
            ex_at_30s=bird_facts[4],
            required_tools=case.required_tools,
            expected_tool_sequences=case.expected_tool_sequences,
            actual_result=actual_result,
            first_execution_success=first_success,
            repair_succeeded=(
                any(execution.execution_success for execution in executions[1:])
                if first_success is False
                else None
            ),
            trajectory_id=result.trajectory_id,
            total_duration_ms=duration_ms,
            total_llm_turns=result.total_llm_turns,
            total_tool_calls=len(calls),
            total_sql_executions=len(executions),
            invalid_tool_calls=invalid_count,
            repeated_tool_calls=repeated_count,
            unsafe_sql_count=unsafe_count,
            message_history=message_history,
            assistant_tool_calls=calls,
            tool_observations=observations,
            actual_tool_sequence=[call.name for call in calls],
            tool_argument_validity=[
                validation.known_tool and validation.arguments_valid
                for validation in validations
            ],
            tool_execution_successes=[
                tool_result.success
                for tool_result in tool_results
                if tool_result.executed
            ],
            error_message=self._execution_error_message(actual_result),
        )
        trajectory = trajectory.model_copy(
            update={
                "dataset_name": dataset_name,
                "tags": case.tags,
                "result_correct": result_correct,
                "execution_result": actual_result,
                "execution_success": evaluation_execution_success,
                "message_history": message_history,
                "assistant_tool_calls": calls,
                "tool_observations": observations,
                "invalid_tool_calls": invalid_count,
                "repeated_tool_calls": repeated_count,
                "unsafe_sql_count": unsafe_count,
            }
        )
        return evaluation, trajectory

    def _score_result(
        self,
        case: EvaluationCase,
        database_path: Path,
        final_sql: str | None,
        agent_execution: ExecutionResult | None,
    ) -> tuple[bool | None, ExecutionResult | None]:
        if case.comparison_mode != "bird_set":
            return compare_execution_result(agent_execution, case.expected_result), None
        if final_sql is None:
            return False, None
        timeout_seconds = self._prediction_timeout_budget(case)
        scored_execution = SQLExecutor(database_path).execute_full(
            final_sql,
            timeout_seconds=timeout_seconds,
        )
        return (
            compare_bird_execution_result(scored_execution, case.expected_result),
            scored_execution,
        )

    def _prediction_timeout_budget(self, case: EvaluationCase) -> float:
        if self.prediction_timeout_policy is None or case.gold_runtime_ms is None:
            return self.prediction_timeout_seconds
        return self.prediction_timeout_policy.timeout_seconds(case.gold_runtime_ms)

    def _bird_score_facts(
        self,
        case: EvaluationCase,
        result_correct: bool | None,
        scored_execution: ExecutionResult | None,
    ) -> tuple[float | None, float | None, bool, bool | None, bool | None]:
        if case.comparison_mode != "bird_set":
            return None, None, False, None, None
        timeout_budget = self._prediction_timeout_budget(case)
        runtime_ms = (
            scored_execution.duration_ms if scored_execution is not None else None
        )
        timed_out = bool(
            scored_execution is not None
            and scored_execution.error is not None
            and scored_execution.error.message.startswith("query timed out after ")
        )
        stable = result_correct is True
        ex_at_seconds = (
            self.prediction_timeout_policy.ex_at_seconds
            if self.prediction_timeout_policy is not None
            else 30.0
        )
        ex_at_30s = bool(
            stable
            and runtime_ms is not None
            and runtime_ms <= ex_at_seconds * 1000
        )
        return timeout_budget, runtime_ms, timed_out, stable, ex_at_30s

    @staticmethod
    def _client_message_history(client: LLMClient) -> list[LLMMessage]:
        if isinstance(client, FakeLLMClient) and client.requests:
            return list(client.requests[-1].messages)
        return []

    @staticmethod
    def _execution_error_message(result: ExecutionResult | None) -> str | None:
        if result is None:
            return None
        if result.error is not None:
            return result.error.message
        return result.blocked_reason

    @staticmethod
    def _failure_kind(
        *,
        result_correct: bool | None,
        protocol_completed: bool,
        execution_success: bool,
        termination_reason: str,
        first_execution_success: bool | None,
        invalid_tool_calls: int,
        comparison_mode: str,
    ) -> FailureKind | None:
        if comparison_mode == "bird_set":
            termination_map = {
                "unsafe_sql": FailureKind.UNSAFE_SQL,
                "repeated_sql": FailureKind.REPEATED_SQL,
                "repeated_tool_call": FailureKind.REPEATED_TOOL_CALL,
                "max_steps_reached": FailureKind.MAX_STEPS,
                "unresolved_completion": FailureKind.UNRESOLVED_COMPLETION,
                "model_error": FailureKind.MODEL_ERROR,
                "unrecoverable_error": FailureKind.UNRECOVERABLE_ERROR,
            }
            if termination_reason in termination_map:
                return termination_map[termination_reason]
            if first_execution_success is not None and not execution_success:
                return FailureKind.EXECUTION_ERROR
            if invalid_tool_calls and not protocol_completed:
                return FailureKind.INVALID_TOOL_CALL
            if protocol_completed and not execution_success:
                return FailureKind.UNGROUNDED_ANSWER
        if result_correct is False:
            return FailureKind.SEMANTIC_MISMATCH
        termination_map = {
            "unsafe_sql": FailureKind.UNSAFE_SQL,
            "repeated_sql": FailureKind.REPEATED_SQL,
            "repeated_tool_call": FailureKind.REPEATED_TOOL_CALL,
            "max_steps_reached": FailureKind.MAX_STEPS,
            "unresolved_completion": FailureKind.UNRESOLVED_COMPLETION,
            "model_error": FailureKind.MODEL_ERROR,
            "unrecoverable_error": FailureKind.UNRECOVERABLE_ERROR,
        }
        if termination_reason in termination_map:
            return termination_map[termination_reason]
        if protocol_completed and not execution_success:
            return FailureKind.UNGROUNDED_ANSWER
        if first_execution_success is False and not execution_success:
            return FailureKind.EXECUTION_ERROR
        if invalid_tool_calls and not protocol_completed:
            return FailureKind.INVALID_TOOL_CALL
        return None

    def _exception_case(
        self,
        case: EvaluationCase,
        dataset_name: str,
        agent_mode: ConcreteMode,
        error_message: str,
    ) -> CaseEvaluation:
        return CaseEvaluation(
            case_id=case.id,
            dataset_name=dataset_name,
            tags=case.tags,
            question=case.question,
            database_id=case.database_id,
            agent_mode=agent_mode,
            protocol_completed=False,
            execution_success=False,
            answer_grounded=False,
            result_correct=(False if case.comparison_mode == "bird_set" else None),
            expected_refusal=case.expected_refusal,
            refused=False,
            expected_unsafe_sql=case.expected_unsafe_sql,
            failure_kind=FailureKind.UNRECOVERABLE_ERROR,
            termination_reason="unrecoverable_error",
            difficulty=case.difficulty,
            comparison_mode=case.comparison_mode,
            expected_result=case.expected_result,
            gold_sql=case.gold_sql,
            gold_runtime_ms=case.gold_runtime_ms,
            prediction_timeout_budget_seconds=(
                self._prediction_timeout_budget(case)
                if case.comparison_mode == "bird_set"
                else None
            ),
            stable_bird_ex=(False if case.comparison_mode == "bird_set" else None),
            ex_at_30s=(False if case.comparison_mode == "bird_set" else None),
            required_tools=case.required_tools,
            expected_tool_sequences=case.expected_tool_sequences,
            trajectory_id=str(uuid4()),
            total_duration_ms=0,
            total_llm_turns=0,
            total_tool_calls=0,
            total_sql_executions=0,
            invalid_tool_calls=0,
            repeated_tool_calls=0,
            unsafe_sql_count=0,
            error_message=error_message,
        )

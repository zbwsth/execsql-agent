"""Chinese CLI for single runs and deterministic Synthetic evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

from execsql_agent.agents.function_calling import FunctionCallingAgent
from execsql_agent.agents.pipeline import PipelineAgent
from execsql_agent.config import DomainConfig, load_domain_config
from execsql_agent.evaluation.bird import (
    load_bird_dataset,
    make_bird_database_resolver,
)
from execsql_agent.evaluation.evaluator import Evaluator
from execsql_agent.evaluation.reports import write_reports
from execsql_agent.evaluation.synthetic import (
    build_scripted_client,
    load_evaluation_dataset,
)
from execsql_agent.generation.sql_generator import SQLGenerator
from execsql_agent.llm.base import LLMClient
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.llm.openai_compatible import (
    OpenAICompatibleLLMClient,
    OpenAICompatibleSettings,
)
from execsql_agent.models import (
    BirdExperimentProtocol,
    EvaluationAgentMode,
    EvaluationCase,
    FunctionCallingAgentResult,
    LLMResponse,
    ResponseMode,
    ToolCallRequest,
)
from execsql_agent.tools.registry import ToolRegistry
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.trajectory.logger import TrajectoryLogger


def _generation_response(sql: str, reason: str) -> LLMResponse:
    return LLMResponse(
        final_answer=json.dumps(
            {
                "sql": sql,
                "reason": reason,
                "referenced_tables": [],
                "referenced_columns": [],
            },
            ensure_ascii=False,
        ),
        response_mode=ResponseMode.PLAIN_FINAL,
    )


def _builtin_fake_responses(agent_mode: str) -> list[LLMResponse]:
    repaired_sql = (
        "SELECT c.customer_name, ROUND(SUM(o.total_amount), 2) AS total_spent "
        "FROM customers AS c JOIN orders AS o ON o.customer_id = c.customer_id "
        "WHERE o.status = 'completed' GROUP BY c.customer_id, c.customer_name "
        "ORDER BY total_spent DESC LIMIT 5"
    )
    if agent_mode == "pipeline":
        return [
            _generation_response(
                "SELECT customer_name FROM customer ORDER BY customer_name LIMIT 5",
                "演示第一次使用错误表名",
            ),
            _generation_response(repaired_sql, "根据 missing_table 修复表名和聚合查询"),
        ]
    return [
        LLMResponse(
            tool_calls=[
                ToolCallRequest(id="call_schema_1", name="inspect_schema", arguments={})
            ],
            response_mode=ResponseMode.NATIVE_TOOL_CALLS,
        ),
        LLMResponse(
            tool_calls=[
                ToolCallRequest(
                    id="call_bad_sql",
                    name="execute_sql",
                    arguments={"sql": "SELECT customer_name FROM customer LIMIT 5"},
                )
            ],
            response_mode=ResponseMode.NATIVE_TOOL_CALLS,
        ),
        LLMResponse(
            tool_calls=[
                ToolCallRequest(id="call_schema_2", name="inspect_schema", arguments={})
            ],
            response_mode=ResponseMode.NATIVE_TOOL_CALLS,
        ),
        LLMResponse(
            tool_calls=[
                ToolCallRequest(
                    id="call_fixed_sql",
                    name="execute_sql",
                    arguments={"sql": repaired_sql},
                )
            ],
            response_mode=ResponseMode.NATIVE_TOOL_CALLS,
        ),
        LLMResponse(
            final_answer="已根据真实 SQLite 查询结果返回消费金额最高的五名客户。",
            response_mode=ResponseMode.PLAIN_FINAL,
        ),
    ]


def _load_fake_responses(path: Path) -> list[LLMResponse]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("FakeLLM 响应文件根节点必须是 JSON 数组。")
    return [LLMResponse.model_validate(item) for item in raw]


def _build_client(agent_mode: str, llm_mode: str, response_file: Path | None) -> LLMClient:
    if llm_mode == "openai":
        return OpenAICompatibleLLMClient()
    responses = (
        _load_fake_responses(response_file)
        if response_file is not None
        else _builtin_fake_responses(agent_mode)
    )
    return FakeLLMClient(responses)


def _print_result(
    *,
    termination_reason: str,
    final_sql: str | None,
    execution_result: object,
    final_answer: str | None,
    trajectory_path: Path | str,
) -> None:
    print(f"终止原因：{termination_reason}")
    print(f"最终 SQL：{final_sql or '无'}")
    if hasattr(execution_result, "model_dump_json"):
        print(f"执行结果：{execution_result.model_dump_json()}")
    else:
        print("执行结果：无")
    if final_answer:
        print(f"最终回答：{final_answer}")
    print(f"轨迹文件：{trajectory_path}")


def _run_command(args: argparse.Namespace) -> int:
    database = Path(args.database)
    domain_config: DomainConfig | None = (
        load_domain_config(args.domain_config) if args.domain_config else None
    )
    trajectory_logger = TrajectoryLogger(Path(args.trajectory_file))
    client = _build_client(args.agent_mode, args.llm, args.fake_responses)
    run_mode = "deterministic/mock" if isinstance(client, FakeLLMClient) else "live"
    backend = type(client).__name__
    try:
        if args.agent_mode == "pipeline":
            agent = PipelineAgent(
                database,
                SQLGenerator(client),
                max_steps=args.max_steps or 3,
            )
            result = agent.run(args.question)
            trajectory = trajectory_logger.from_pipeline_result(
                result, llm_backend=backend, run_mode=run_mode
            )
            trajectory_path = trajectory_logger.log(trajectory)
            _print_result(
                termination_reason=result.termination_reason.value,
                final_sql=result.final_sql,
                execution_result=result.execution_result,
                final_answer=None,
                trajectory_path=trajectory_path,
            )
            return 0 if result.protocol_completed else 1

        registry = None
        if domain_config is not None:
            registry = ToolRegistry(
                database,
                executor=SQLExecutor(
                    database, max_rows=domain_config.default_result_limit
                ),
            )
        function_agent = FunctionCallingAgent(
            database,
            client,
            registry=registry,
            trajectory_logger=trajectory_logger,
            max_steps=args.max_steps or 5,
            llm_backend=backend,
            run_mode=run_mode,
            domain_context=(domain_config.to_prompt() if domain_config else None),
        )
        function_result = function_agent.run(args.question, session_id=args.session_id)
        _print_function_turns(function_result)
        _print_result(
            termination_reason=function_result.termination_reason.value,
            final_sql=function_result.final_sql,
            execution_result=function_result.execution_result,
            final_answer=function_result.final_answer,
            trajectory_path=function_result.trajectory_path or trajectory_logger.output_path,
        )
        print("Memory 摘要：")
        print(function_agent.memory_store.render_prompt(args.session_id))
        return 0 if function_result.protocol_completed else 1
    finally:
        if isinstance(client, OpenAICompatibleLLMClient):
            client.close()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _protocol_source_fingerprint() -> str:
    digest = sha256()
    project_root = Path(__file__).resolve().parents[2]
    for relative_path in (
        Path("src/execsql_agent/agents/function_calling.py"),
        Path("src/execsql_agent/llm/openai_compatible.py"),
        Path("src/execsql_agent/models.py"),
        Path("src/execsql_agent/tools/registry.py"),
        Path("src/execsql_agent/tools/sql_executor.py"),
        Path("src/execsql_agent/tools/sql_validator.py"),
        Path("src/execsql_agent/evaluation/evaluator.py"),
        Path("src/execsql_agent/evaluation/comparator.py"),
        Path("src/execsql_agent/evaluation/metrics.py"),
    ):
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update((project_root / relative_path).read_bytes())
    return digest.hexdigest()


def _load_bird_protocol(path: Path) -> BirdExperimentProtocol:
    return BirdExperimentProtocol.model_validate_json(path.read_text(encoding="utf-8"))


def _write_experiment_config(
    output_dir: Path,
    stable_config: dict[str, object],
    *,
    resume: bool,
) -> str:
    canonical = json.dumps(
        stable_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = sha256(canonical.encode("utf-8")).hexdigest()
    path = output_dir / "experiment_config.json"
    if path.exists():
        existing_raw: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing_raw, dict):
            raise ValueError(f"Invalid experiment config: {path}")
        existing_fingerprint = existing_raw.get("config_fingerprint")
        if not resume:
            raise ValueError(
                f"Experiment config already exists; use --resume: {path}"
            )
        if existing_fingerprint != fingerprint:
            raise ValueError(
                "Existing experiment config fingerprint does not match the "
                "current model/protocol/dataset/cache configuration."
            )
        return fingerprint
    if resume and (output_dir / "case_checkpoint.jsonl").exists():
        raise ValueError("Resume checkpoint exists without experiment_config.json")
    stale_outputs = [
        output_dir / name
        for name in (
            "case_checkpoint.jsonl",
            "trajectories.jsonl",
            "evaluation_report.json",
            "evaluation_report.md",
            "case_results.csv",
        )
        if (output_dir / name).exists()
    ]
    if stale_outputs:
        raise ValueError(
            "Experiment output already exists without a matching config: "
            + ", ".join(str(path) for path in stale_outputs)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **stable_config,
        "config_fingerprint": fingerprint,
        "run_timestamp": datetime.now(UTC).isoformat(),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
    return fingerprint


def _evaluate_command(args: argparse.Namespace) -> int:
    database_resolver = None
    database_path = args.database
    protocol: BirdExperimentProtocol | None = None
    config_fingerprint: str | None = None
    checkpoint_path: Path | None = None
    model_name = args.model or os.environ.get("OPENAI_MODEL", "")
    output_dir = Path(args.output_dir)
    if args.dataset_format == "bird":
        if args.database_root is None:
            raise ValueError("BIRD evaluation requires --database-root.")
        if not args.real_model:
            raise ValueError("BIRD evaluation requires --real-model.")
        protocol = _load_bird_protocol(args.bird_protocol_config)
        if args.agent_mode != protocol.agent_mode:
            raise ValueError(
                f"BIRD protocol requires --agent-mode {protocol.agent_mode}."
            )
        if not model_name:
            raise ValueError("BIRD evaluation requires --model or OPENAI_MODEL.")
        dataset = load_bird_dataset(
            args.dataset,
            args.database_root,
            gold_cache_path=args.bird_gold_cache,
            gold_cache_read_only=True,
            gold_timeout_seconds=args.gold_timeout_seconds,
        )
        if args.case_id:
            requested_ids = set(args.case_id)
            available_ids = {case.id for case in dataset.cases}
            missing_ids = sorted(requested_ids - available_ids)
            if missing_ids:
                raise ValueError(
                    "Unknown BIRD case ids: " + ", ".join(missing_ids)
                )
            dataset = dataset.model_copy(
                update={
                    "cases": [
                        case for case in dataset.cases if case.id in requested_ids
                    ]
                }
            )
        database_resolver = make_bird_database_resolver(args.database_root)
        database_path = None
        preflight: object = json.loads(
            args.bird_preflight_report.read_text(encoding="utf-8")
        )
        if not isinstance(preflight, dict):
            raise ValueError("BIRD preflight report must be a JSON object.")
        if preflight.get("dataset_revision") != protocol.dataset_revision:
            raise ValueError(
                "BIRD protocol dataset revision does not match preflight report."
            )
        cache = preflight.get("cache")
        provenance = preflight.get("database_package_provenance")
        if not isinstance(cache, dict) or not isinstance(provenance, dict):
            raise ValueError("BIRD preflight report is missing cache provenance.")
        stable_config: dict[str, object] = {
            "model": model_name,
            "adapter": args.adapter,
            "dataset": {
                "name": dataset.dataset_name,
                "revision": protocol.dataset_revision,
                "path": str(Path(args.dataset).resolve()),
                "sha256": _sha256_file(Path(args.dataset)),
                "selected_case_ids": [case.id for case in dataset.cases],
                "limit": args.limit,
            },
            "database_provenance": provenance,
            "gold_cache": {
                "version": cache.get("version"),
                "path": str(Path(args.bird_gold_cache).resolve()),
                "sha256": _sha256_file(Path(args.bird_gold_cache)),
            },
            "decoding_config": {
                "temperature": protocol.temperature,
                "top_p": protocol.top_p,
                "max_tokens": protocol.max_tokens,
                "seed": protocol.seed,
                "enable_thinking": protocol.enable_thinking,
            },
            "client_config": {
                "base_url": os.environ.get("OPENAI_BASE_URL", ""),
                "request_timeout_seconds": protocol.request_timeout_seconds,
                "max_retries": protocol.max_retries,
            },
            "timeout_policy": protocol.timeout_policy.model_dump(mode="json"),
            "max_steps": protocol.max_agent_steps,
            "agent_mode": protocol.agent_mode,
            "oracle_evidence": protocol.use_evidence,
            "domain_config": (
                {
                    "path": str(args.domain_config.resolve()),
                    "sha256": _sha256_file(args.domain_config),
                }
                if args.domain_config is not None
                else None
            ),
            "vllm_configuration": protocol.vllm_common,
            "git_commit_sha": _git_commit_sha(),
            "protocol_source_sha256": _protocol_source_fingerprint(),
        }
        config_fingerprint = _write_experiment_config(
            output_dir,
            stable_config,
            resume=args.resume,
        )
        checkpoint_path = output_dir / "case_checkpoint.jsonl"
    else:
        if database_path is None:
            raise ValueError("Native evaluation requires --database.")
        if args.resume:
            raise ValueError("--resume is currently reserved for BIRD evaluation.")
        dataset = load_evaluation_dataset(args.dataset)
    domain_config = (
        load_domain_config(args.domain_config) if args.domain_config else None
    )

    def real_client_factory(
        _case: EvaluationCase,
        _mode: Literal["pipeline", "function-calling"],
    ) -> LLMClient:
        if protocol is None:
            return OpenAICompatibleLLMClient()
        settings = OpenAICompatibleSettings(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", ""),
            model=model_name,
            enable_thinking=protocol.enable_thinking,
        )
        return OpenAICompatibleLLMClient(
            settings,
            timeout_seconds=protocol.request_timeout_seconds,
            max_retries=protocol.max_retries,
            temperature=protocol.temperature,
            top_p=protocol.top_p,
            max_tokens=protocol.max_tokens,
            seed=protocol.seed,
        )

    client_factory = real_client_factory if args.real_model else build_scripted_client
    evaluator = Evaluator(
        database_path,
        client_factory,
        trajectory_logger=TrajectoryLogger(output_dir / "trajectories.jsonl"),
        domain_context=(domain_config.to_prompt() if domain_config else None),
        llm_backend=("OpenAICompatibleLLMClient" if args.real_model else "FakeLLMClient"),
        run_mode=("live" if args.real_model else "deterministic/mock"),
        database_resolver=database_resolver,
        use_evidence=(protocol.use_evidence if protocol is not None else args.use_evidence),
        prediction_timeout_seconds=args.prediction_timeout_seconds,
        prediction_timeout_policy=(
            protocol.timeout_policy if protocol is not None else None
        ),
        max_agent_steps=(
            protocol.max_agent_steps if protocol is not None else None
        ),
    )
    report = evaluator.evaluate(
        dataset.cases,
        dataset_name=dataset.dataset_name,
        agent_mode=EvaluationAgentMode(args.agent_mode),
        seed=(protocol.seed if protocol is not None else args.seed),
        tags=args.tag or [],
        limit=args.limit,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
        config_fingerprint=config_fingerprint,
    )
    paths = write_reports(report, output_dir)
    if args.real_model:
        print("真实模型评测完成；指标来自本次实际模型调用和 SQLite 执行。")
    else:
        print("评测完成：deterministic/mock，仅表示流程可复现，不代表真实模型能力。")
    for mode, metrics in report.mode_metrics.items():
        print(
            f"{mode}：案例 {metrics.total_cases.numerator:g}，"
            f"执行成功率 {_display_metric(metrics.final_execution_success_rate.value)}，"
            f"结果正确率 {_display_metric(metrics.result_accuracy.value)}"
        )
    if report.comparison is not None:
        print(
            "结果正确率差值（Function Calling - Pipeline）："
            f"{_display_metric(report.comparison.result_accuracy_difference)}"
        )
    print(f"JSON 报告：{paths['json']}")
    print(f"CSV 报告：{paths['csv']}")
    print(f"Markdown 报告：{paths['markdown']}")
    return 0


def _display_metric(value: float | None) -> str:
    return "null" if value is None else f"{value:.4f}"


def _print_function_turns(result: FunctionCallingAgentResult) -> None:
    """Print bounded per-turn Function Calling facts without exposing credentials."""

    for step in result.steps:
        print(f"第 {step.turn_index} 轮：")
        for call in step.tool_calls:
            arguments = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
            print(f"  工具调用：{call.name} {arguments}")
            if call.name in {"validate_sql", "execute_sql"} and call.arguments:
                sql = call.arguments.get("sql")
                if isinstance(sql, str):
                    print(f"  SQL：{sql}")
        for tool_result in step.tool_results:
            summary = (
                f"success={tool_result.success}, executed={tool_result.executed}, "
                f"error={tool_result.error_code or 'none'}"
            )
            execution = ToolRegistry.execution_result(tool_result)
            if execution is not None:
                summary += (
                    f", execution_success={execution.execution_success}, "
                    f"rows={execution.returned_row_count}, "
                    f"sample={execution.rows[:3]}"
                )
            print(f"  执行结果：{summary}")


def build_parser() -> argparse.ArgumentParser:
    """Build single-run and Synthetic evaluation commands."""

    parser = argparse.ArgumentParser(description="ExecSQL-Agent 最小运行工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="运行单个 Text-to-SQL 问题")
    run_parser.add_argument(
        "--agent-mode",
        choices=["pipeline", "function-calling"],
        required=True,
        help="Agent 编排模式",
    )
    run_parser.add_argument("--database", required=True, help="SQLite 数据库路径")
    run_parser.add_argument("--question", required=True, help="中文自然语言问题")
    run_parser.add_argument(
        "--domain-config",
        type=Path,
        help="可选的严格 JSON 领域配置，仅作为模型上下文和结果行限制",
    )
    run_parser.add_argument(
        "--session-id",
        default="default",
        help="Function Calling 会话 ID；同一轨迹文件中按此隔离最近 5 轮上下文",
    )
    run_parser.add_argument(
        "--llm",
        choices=["fake", "openai"],
        default="fake",
        help="模型后端，默认使用 deterministic/mock FakeLLM",
    )
    run_parser.add_argument(
        "--fake-responses",
        type=Path,
        help="可选的 FakeLLM LLMResponse JSON 数组文件",
    )
    run_parser.add_argument(
        "--trajectory-file",
        type=Path,
        default=Path("data/trajectories/trajectories.jsonl"),
        help="JSONL 轨迹文件路径",
    )
    run_parser.add_argument("--max-steps", type=int, help="覆盖默认最大步骤数")
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="运行 deterministic/mock Synthetic 批量评测"
    )
    evaluate_parser.add_argument(
        "--agent-mode",
        choices=[mode.value for mode in EvaluationAgentMode],
        required=True,
        help="评测 Pipeline、Function Calling 或两者",
    )
    database_group = evaluate_parser.add_mutually_exclusive_group(required=True)
    database_group.add_argument("--database", help="单数据库评测的 SQLite 路径")
    database_group.add_argument(
        "--database-root",
        type=Path,
        help="BIRD 数据库根目录，包含 <db_id>/<db_id>.sqlite",
    )
    evaluate_parser.add_argument("--dataset", required=True, type=Path, help="评测集路径")
    evaluate_parser.add_argument(
        "--dataset-format",
        choices=["native", "bird"],
        default="native",
        help="输入数据格式，默认使用项目原生格式",
    )
    evaluate_parser.add_argument(
        "--output-dir", required=True, type=Path, help="JSON、CSV、Markdown 输出目录"
    )
    evaluate_parser.add_argument("--limit", type=int, help="最多运行的案例数")
    evaluate_parser.add_argument(
        "--case-id",
        action="append",
        help="只运行指定 case id；可重复，用于固定 smoke subset",
    )
    evaluate_parser.add_argument(
        "--tag", action="append", help="按标签筛选，可重复指定且需全部匹配"
    )
    evaluate_parser.add_argument("--seed", type=int, default=0, help="固定随机种子")
    evaluate_parser.add_argument(
        "--domain-config",
        type=Path,
        help="可选的严格 JSON 领域配置",
    )
    evaluate_parser.add_argument(
        "--real-model",
        action="store_true",
        help="使用 OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL 调用真实模型",
    )
    evaluate_parser.add_argument(
        "--use-evidence",
        action="store_true",
        help="显式向模型提供数据集 evidence（默认关闭）",
    )
    evaluate_parser.add_argument(
        "--bird-gold-cache",
        type=Path,
        default=Path("data/bird/cache/mini_dev_gold_results.jsonl"),
        help="BIRD scorer-side gold result cache",
    )
    evaluate_parser.add_argument(
        "--bird-preflight-report",
        type=Path,
        default=Path("data/bird/preflight_report.json"),
        help="冻结的 BIRD 数据与数据库 provenance 报告",
    )
    evaluate_parser.add_argument(
        "--bird-protocol-config",
        type=Path,
        default=Path("config/bird_eval_protocol.json"),
        help="Base/SFT 共用的冻结评测协议",
    )
    evaluate_parser.add_argument(
        "--gold-timeout-seconds",
        type=float,
        default=900.0,
        help="仅用于 BIRD gold cache 初始化的执行超时",
    )
    evaluate_parser.add_argument(
        "--prediction-timeout-seconds",
        type=float,
        default=30.0,
        help="无 cached gold runtime 时的 prediction scorer fallback timeout",
    )
    evaluate_parser.add_argument(
        "--model",
        help="真实评测使用的 OpenAI-compatible served model name",
    )
    evaluate_parser.add_argument(
        "--adapter",
        help="实验元数据中的 adapter 名称或路径；Base 留空",
    )
    evaluate_parser.add_argument(
        "--resume",
        action="store_true",
        help="按 config fingerprint 从逐 case checkpoint 安全续跑",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected agent and render Chinese errors."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            return _evaluate_command(args)
        return _run_command(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"运行失败：{error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""BIRD Train structural analysis, split freezing, and pool manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

VALIDATION_CANDIDATES: dict[str, tuple[str, ...]] = {
    "A": (
        "public_review_platform",
        "hockey",
        "regional_sales",
        "restaurant",
        "computer_student",
    ),
    "B": (
        "legislator",
        "movies_4",
        "address",
        "food_inspection_2",
        "cs_semester",
        "sales",
        "world",
    ),
    "C": (
        "video_games",
        "books",
        "retail_complains",
        "image_and_language",
        "beer_factory",
        "law_episode",
    ),
}
DISTANCE_TIE_THRESHOLD = 0.01
DEFAULT_SEED = "bird-phase-c-v1"
DEFAULT_MONITORING_SIZE = 128
DEFAULT_MAX_DATABASE_SHARE = 0.06


def _normalized_question(question: str) -> str:
    return "".join(question.split()).casefold()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _case_id(index: int) -> str:
    return f"bird_train_{index:05d}"


def _schema_scale(schema: dict[str, Any]) -> tuple[int, int]:
    tables = schema["table_names_original"]
    columns = schema["column_names_original"]
    return len(tables), sum(1 for table_index, _ in columns if table_index >= 0)


def _table_bucket(count: int) -> str:
    if count <= 5:
        return "1-5"
    if count <= 10:
        return "6-10"
    return "11+"


def _column_bucket(count: int) -> str:
    if count <= 50:
        return "1-50"
    if count <= 100:
        return "51-100"
    return "101+"


def analyze_sql(sql: str) -> dict[str, object]:
    """Return SQLite structural features and physical tables."""

    tree = sqlglot.parse_one(sql, read="sqlite")
    cte_names = {
        cte.alias_or_name.casefold() for cte in tree.find_all(exp.CTE)
    }
    physical_tables = sorted(
        {
            table.name
            for table in tree.find_all(exp.Table)
            if table.name.casefold() not in cte_names
        },
        key=str.casefold,
    )
    join_count = sum(1 for _ in tree.find_all(exp.Join))
    select_count = sum(1 for _ in tree.find_all(exp.Select))
    has_group_by = tree.find(exp.Group) is not None
    has_having = tree.find(exp.Having) is not None
    has_cte = any(True for _ in tree.find_all(exp.CTE))
    has_set_operation = any(
        True
        for _ in tree.find_all((exp.Union, exp.Intersect, exp.Except))
    )
    has_aggregation = any(True for _ in tree.find_all(exp.AggFunc))
    return {
        "physical_tables": physical_tables,
        "physical_table_count": len(physical_tables),
        "table_mode": "single" if len(physical_tables) <= 1 else "multi",
        "join_count": join_count,
        "join_bucket": (
            "0"
            if join_count == 0
            else "1"
            if join_count == 1
            else "2"
            if join_count == 2
            else "3+"
        ),
        "aggregation": has_aggregation,
        "group_by": has_group_by,
        "having": has_having,
        "nested_select": select_count > 1,
        "cte": has_cte,
        "set_operation": has_set_operation,
        "complex": (
            len(physical_tables) > 1
            or join_count > 0
            or has_aggregation
            or has_group_by
            or has_having
            or select_count > 1
            or has_cte
            or has_set_operation
        ),
    }


def build_records(
    annotations: list[dict[str, Any]],
    schemas: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(annotations):
        db_id = str(row["db_id"])
        features = analyze_sql(str(row["SQL"]))
        schema_tables, schema_columns = _schema_scale(schemas[db_id])
        physical = {
            str(table).casefold()
            for table in features["physical_tables"]
        }
        known = {
            str(table).casefold()
            for table in schemas[db_id]["table_names_original"]
        }
        unknown_tables = sorted(physical - known)
        features.update(
            {
                "schema_tables": schema_tables,
                "schema_columns": schema_columns,
                "schema_table_bucket": _table_bucket(schema_tables),
                "schema_column_bucket": _column_bucket(schema_columns),
            }
        )
        records.append(
            {
                "case_id": _case_id(index),
                "annotation_index": index,
                "database_id": db_id,
                "question": str(row["question"]),
                "gold_sql": str(row["SQL"]),
                "evidence": str(row.get("evidence", "")),
                "features": features,
                "static_schema_match": not unknown_tables,
                "unknown_physical_tables": unknown_tables,
            }
        )
    return records


def _proportions(
    records: list[dict[str, Any]], key: str
) -> dict[str, float]:
    counts = Counter(str(record["features"][key]) for record in records)
    total = len(records)
    return {
        value: count / total for value, count in sorted(counts.items())
    }


def structure_report(records: list[dict[str, Any]]) -> dict[str, object]:
    binary_keys = (
        "aggregation",
        "group_by",
        "having",
        "nested_select",
        "cte",
        "set_operation",
    )
    numeric_keys = (
        "physical_table_count",
        "join_count",
        "schema_tables",
        "schema_columns",
    )
    report: dict[str, object] = {
        "cases": len(records),
        "categorical": {
            key: _proportions(records, key)
            for key in (
                "table_mode",
                "join_bucket",
                "schema_table_bucket",
                "schema_column_bucket",
            )
        },
        "binary": {
            key: sum(bool(record["features"][key]) for record in records)
            / len(records)
            for key in binary_keys
        },
        "numeric": {},
    }
    numeric: dict[str, object] = {}
    for key in numeric_keys:
        values = sorted(int(record["features"][key]) for record in records)
        numeric[key] = {
            "mean": sum(values) / len(values),
            "p50": values[len(values) // 2],
            "p90": values[min(len(values) - 1, math.ceil(len(values) * 0.9) - 1)],
            "max": values[-1],
        }
    report["numeric"] = numeric
    return report


def _total_variation(
    left: dict[str, float], right: dict[str, float]
) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys
    )


def structure_distance(
    full: dict[str, Any], candidate: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    for key in full["categorical"]:
        components[key] = _total_variation(
            full["categorical"][key],
            candidate["categorical"][key],
        )
    for key in full["binary"]:
        components[key] = abs(
            full["binary"][key] - candidate["binary"][key]
        )
    for key in ("schema_tables", "schema_columns"):
        full_mean = float(full["numeric"][key]["mean"])
        candidate_mean = float(candidate["numeric"][key]["mean"])
        components[f"{key}_mean_normalized"] = (
            abs(full_mean - candidate_mean) / max(full_mean, 1.0)
        )
    return sum(components.values()) / len(components), components


def _stratum(record: dict[str, Any]) -> str:
    features = record["features"]
    return "|".join(
        (
            str(features["table_mode"]),
            f"join={features['join_bucket']}",
            f"agg={int(bool(features['aggregation']))}",
            f"nested={int(bool(features['nested_select']))}",
            f"schema={features['schema_table_bucket']}",
        )
    )


def _stable_rank(seed: str, case_id: str) -> str:
    return hashlib.sha256(f"{seed}|{case_id}".encode()).hexdigest()


def stratified_sample(
    records: list[dict[str, Any]],
    *,
    size: int,
    seed: str,
    max_database_share: float,
) -> list[dict[str, Any]]:
    if size < 1 or size > len(records):
        raise ValueError("Invalid stratified sample size")
    cap = max(1, math.ceil(size * max_database_share))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_stratum(record)].append(record)
    exact = {
        key: size * len(group) / len(records)
        for key, group in groups.items()
    }
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remainder = size - sum(quotas.values())
    for key in sorted(
        groups,
        key=lambda item: (
            -(exact[item] - quotas[item]),
            item,
        ),
    )[:remainder]:
        quotas[key] += 1

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    database_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    for key in sorted(groups):
        ordered = sorted(
            groups[key],
            key=lambda record: _stable_rank(seed, str(record["case_id"])),
        )
        for record in ordered:
            if stratum_counts[key] >= quotas[key]:
                break
            db_id = str(record["database_id"])
            if database_counts[db_id] >= cap:
                continue
            selected.append(record)
            selected_ids.add(str(record["case_id"]))
            database_counts[db_id] += 1
            stratum_counts[key] += 1

    if len(selected) < size:
        remaining = sorted(
            (
                record
                for record in records
                if str(record["case_id"]) not in selected_ids
            ),
            key=lambda record: _stable_rank(
                f"{seed}|fill", str(record["case_id"])
            ),
        )
        for record in remaining:
            db_id = str(record["database_id"])
            if database_counts[db_id] >= cap:
                continue
            selected.append(record)
            selected_ids.add(str(record["case_id"]))
            database_counts[db_id] += 1
            if len(selected) == size:
                break
    if len(selected) != size:
        raise ValueError(
            f"Database cap prevented selecting {size} cases; got {len(selected)}"
        )
    return sorted(selected, key=lambda record: int(record["annotation_index"]))


def _manifest_entry(record: dict[str, Any]) -> dict[str, object]:
    return {
        "case_id": record["case_id"],
        "annotation_index": record["annotation_index"],
        "database_id": record["database_id"],
        "question_sha256": hashlib.sha256(
            str(record["question"]).encode()
        ).hexdigest(),
        "gold_sql_sha256": hashlib.sha256(
            str(record["gold_sql"]).encode()
        ).hexdigest(),
        "evidence_present": bool(str(record["evidence"]).strip()),
        "features": record["features"],
        "static_schema_match": record["static_schema_match"],
        "unknown_physical_tables": record["unknown_physical_tables"],
    }


def _write_checksums(output_dir: Path) -> None:
    checksum_path = output_dir / "SHA256SUMS"
    lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path != checksum_path:
            lines.append(f"{sha256_file(path)}  {path.name}")
    temporary = checksum_path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(checksum_path)


def prepare(
    *,
    annotations_path: Path,
    tables_path: Path,
    output_dir: Path,
    eligibility_path: Path | None,
    pool_exclusions_path: Path | None,
    sft_size: int,
    grpo_size: int,
    monitoring_size: int,
    seed: str,
    max_database_share: float,
) -> None:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    tables = json.loads(tables_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, list) or not isinstance(tables, list):
        raise ValueError("BIRD annotations and tables must be arrays")
    schemas = {str(schema["db_id"]): schema for schema in tables}
    records = build_records(annotations, schemas)
    full_report = structure_report(records)

    candidate_reports: dict[str, object] = {}
    scores: dict[str, float] = {}
    for name, databases in VALIDATION_CANDIDATES.items():
        candidate_records = [
            record
            for record in records
            if record["database_id"] in databases
        ]
        report = structure_report(candidate_records)
        score, components = structure_distance(full_report, report)
        scores[name] = score
        candidate_reports[name] = {
            "databases": list(databases),
            "structure": report,
            "distance_score": score,
            "distance_components": components,
        }

    closest = min(scores, key=scores.get)
    a_b_delta = abs(scores["A"] - scores["B"])
    frozen = (
        "B"
        if closest in {"A", "B"} and a_b_delta < DISTANCE_TIE_THRESHOLD
        else closest
    )
    validation_databases = set(VALIDATION_CANDIDATES[frozen])
    train_databases = sorted(
        {str(record["database_id"]) for record in records}
        - validation_databases
    )
    validation_records = [
        record
        for record in records
        if record["database_id"] in validation_databases
    ]
    train_records = [
        record
        for record in records
        if record["database_id"] not in validation_databases
    ]
    static_invalid = [
        record for record in train_records if not record["static_schema_match"]
    ]
    screen_records = [
        record for record in train_records if record["static_schema_match"]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "version": 1,
        "dataset": "bird_full_train",
        "annotations": {
            "path": str(annotations_path),
            "sha256": sha256_file(annotations_path),
            "cases": len(records),
        },
        "tables": {
            "path": str(tables_path),
            "sha256": sha256_file(tables_path),
            "databases": len(schemas),
        },
        "frozen_candidate": frozen,
        "decision": {
            "closest_candidate": closest,
            "a_b_distance_delta": a_b_delta,
            "tie_threshold": DISTANCE_TIE_THRESHOLD,
            "reason": (
                "A and B are structurally tied within threshold; "
                "freeze B as specified."
                if frozen == "B" and closest == "A"
                else "Freeze the lowest structural-distance candidate."
            ),
        },
        "validation_databases": sorted(validation_databases),
        "training_databases": train_databases,
        "validation_cases": len(validation_records),
        "non_validation_cases": len(train_records),
    }
    _atomic_json(output_dir / "bird_db_split_v1.json", split_manifest)
    _atomic_json(
        output_dir / "bird_validation_structure_audit_v1.json",
        {
            "version": 1,
            "feature_source": (
                "Full BIRD Train annotations, gold SQL, and train_tables.json"
            ),
            "mini_dev_used": False,
            "full_train": full_report,
            "candidates": candidate_reports,
            "frozen_candidate": frozen,
        },
    )
    _atomic_json(
        output_dir / "bird_val_manifest_v1.json",
        {
            "version": 1,
            "split": "validation",
            "candidate": frozen,
            "databases": sorted(validation_databases),
            "cases": [
                _manifest_entry(record) for record in validation_records
            ],
        },
    )
    monitoring = stratified_sample(
        validation_records,
        size=min(monitoring_size, len(validation_records)),
        seed=f"{seed}|monitoring",
        max_database_share=0.25,
    )
    _atomic_json(
        output_dir / "bird_val_monitoring_v1.json",
        {
            "version": 1,
            "split": "validation_monitoring",
            "candidate": frozen,
            "selection": "deterministic structural strata",
            "cases": [
                {
                    **_manifest_entry(record),
                    "question": record["question"],
                    "gold_sql": record["gold_sql"],
                    "evidence": record["evidence"],
                }
                for record in monitoring
            ],
        },
    )
    _atomic_json(
        output_dir / "bird_screen_manifest_v1.json",
        {
            "version": 1,
            "split": "non_validation_screen",
            "cases": [_manifest_entry(record) for record in screen_records],
            "static_invalid_cases": [
                _manifest_entry(record) for record in static_invalid
            ],
        },
    )

    if eligibility_path is not None:
        eligibility_raw = json.loads(
            eligibility_path.read_text(encoding="utf-8")
        )
        outcomes = eligibility_raw.get("cases")
        if not isinstance(outcomes, dict):
            raise ValueError("Eligibility report has no cases mapping")
        eligible = [
            record
            for record in screen_records
            if outcomes.get(str(record["case_id"]), {}).get("eligible")
            is True
        ]
        pool_exclusion_ids: set[str] = set()
        pool_exclusion_metadata: dict[str, object] | None = None
        if pool_exclusions_path is not None:
            pool_exclusions = json.loads(
                pool_exclusions_path.read_text(encoding="utf-8")
            )
            raw_exclusion_ids = pool_exclusions.get("case_ids")
            if not isinstance(raw_exclusion_ids, list) or not all(
                isinstance(case_id, str) for case_id in raw_exclusion_ids
            ):
                raise ValueError("Pool exclusions require a case_ids array")
            pool_exclusion_ids = set(raw_exclusion_ids)
            eligible_ids = {str(record["case_id"]) for record in eligible}
            unknown_exclusions = pool_exclusion_ids - eligible_ids
            if unknown_exclusions:
                raise ValueError(
                    "Pool exclusions are not execution-eligible cases: "
                    + ", ".join(sorted(unknown_exclusions)[:5])
                )
            eligible = [
                record
                for record in eligible
                if str(record["case_id"]) not in pool_exclusion_ids
            ]
            pool_exclusion_metadata = {
                "path": str(pool_exclusions_path),
                "sha256": sha256_file(pool_exclusions_path),
                "cases": len(pool_exclusion_ids),
            }
        unique_eligible: list[dict[str, Any]] = []
        duplicate_eligible: list[dict[str, Any]] = []
        seen_questions: set[str] = set()
        for record in eligible:
            normalized_question = _normalized_question(
                str(record["question"])
            )
            if normalized_question in seen_questions:
                duplicate_eligible.append(record)
                continue
            seen_questions.add(normalized_question)
            unique_eligible.append(record)
        if len(unique_eligible) < sft_size + grpo_size:
            raise ValueError(
                "Not enough unique execution-eligible questions for "
                "SFT and GRPO pools"
            )
        sft = stratified_sample(
            unique_eligible,
            size=sft_size,
            seed=f"{seed}|sft",
            max_database_share=max_database_share,
        )
        sft_ids = {str(record["case_id"]) for record in sft}
        grpo_candidates = [
            record
            for record in unique_eligible
            if str(record["case_id"]) not in sft_ids
        ]
        grpo = stratified_sample(
            grpo_candidates,
            size=grpo_size,
            seed=f"{seed}|grpo",
            max_database_share=max_database_share,
        )
        grpo_ids = {str(record["case_id"]) for record in grpo}
        reserve = [
            record
            for record in train_records
            if str(record["case_id"]) not in sft_ids | grpo_ids
        ]
        for name, selected in (
            ("sft", sft),
            ("grpo", grpo),
            ("reserve", reserve),
        ):
            _atomic_json(
                output_dir / f"bird_{name}_pool_v1.json",
                {
                    "version": 1,
                    "pool": name,
                    "seed": seed,
                    "max_database_share": max_database_share,
                    "cases": [
                        _manifest_entry(record) for record in selected
                    ],
                },
            )
        if sft_ids & grpo_ids:
            raise ValueError("SFT and GRPO pools overlap")
        _atomic_json(
            output_dir / "bird_pool_audit_v1.json",
            {
                "version": 1,
                "execution_eligibility": {
                    "path": str(eligibility_path),
                    "sha256": sha256_file(eligibility_path),
                    "eligible_cases": len(eligible),
                    "unique_eligible_questions": len(unique_eligible),
                    "duplicate_eligible_questions_excluded": len(
                        duplicate_eligible
                    ),
                    "ineligible_cases": len(train_records) - len(eligible),
                },
                "pool_exclusions": pool_exclusion_metadata,
                "sft_cases": len(sft),
                "grpo_cases": len(grpo),
                "reserve_cases": len(reserve),
                "sft_grpo_overlap": 0,
                "static_schema_mismatch_cases": len(static_invalid),
                "sft_structure": structure_report(sft),
                "grpo_structure": structure_report(grpo),
                "reserve_structure": structure_report(reserve),
                "sft_database_counts": dict(
                    sorted(Counter(str(r["database_id"]) for r in sft).items())
                ),
                "grpo_database_counts": dict(
                    sorted(Counter(str(r["database_id"]) for r in grpo).items())
                ),
            },
        )
    _write_checksums(output_dir)
    print(
        json.dumps(
            {
                "frozen_candidate": frozen,
                "split_sha256": sha256_file(
                    output_dir / "bird_db_split_v1.json"
                ),
                "validation_cases": len(validation_records),
                "screen_cases": len(screen_records),
                "static_invalid": len(static_invalid),
                "pools_written": eligibility_path is not None,
            },
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare frozen BIRD Train split and pool manifests."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eligibility", type=Path)
    parser.add_argument("--pool-exclusions", type=Path)
    parser.add_argument("--sft-size", type=int, default=2500)
    parser.add_argument("--grpo-size", type=int, default=2500)
    parser.add_argument(
        "--monitoring-size", type=int, default=DEFAULT_MONITORING_SIZE
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument(
        "--max-database-share",
        type=float,
        default=DEFAULT_MAX_DATABASE_SHARE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 < args.max_database_share <= 1:
        raise ValueError("max-database-share must be in (0, 1]")
    prepare(
        annotations_path=args.annotations,
        tables_path=args.tables,
        output_dir=args.output_dir,
        eligibility_path=args.eligibility,
        pool_exclusions_path=args.pool_exclusions,
        sft_size=args.sft_size,
        grpo_size=args.grpo_size,
        monitoring_size=args.monitoring_size,
        seed=args.seed,
        max_database_share=args.max_database_share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from google.cloud import firestore

from config import logger
from firestore_client import get_firestore_client, get_firestore_console_link

_CAVEAT_TEXT = (
    "This analysis covers only the recorded FAILURES for this run. It cannot compute "
    "failure RATES per parameter value because Firestore does not store a denominator "
    "(total attempts, including passes, per param combination). All counts below are "
    "absolute counts among failures only."
)

_BACKTICK_RE = re.compile(r'`([^`]+)`')

_PARAM_SET_EXCLUDED_KEYS = ("symbols", "cases")


def _get_run_doc_ref(db: firestore.Client, run_id: str):
    return db.collection("test-results").document(run_id)


def fetch_run(db: firestore.Client, run_id: str) -> Dict[str, Any]:
    snapshot = _get_run_doc_ref(db, run_id).get()
    if not snapshot.exists:
        logger.error(f"No test-results run found with ID: {run_id}")
        sys.exit(1)
    return {**(snapshot.to_dict() or {}), "run_id": run_id}


def fetch_latest_run(db: firestore.Client, eval_function_name: Optional[str] = None) -> Dict[str, Any]:
    """Fetch the most recently created test-results run, optionally scoped to one eval_function_name."""
    query = db.collection("test-results")
    if eval_function_name:
        query = query.where("eval_function_name", "==", eval_function_name)
    query = query.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1)

    docs = list(query.stream())
    if not docs:
        scope = f" for eval_function_name={eval_function_name!r}" if eval_function_name else ""
        logger.error(f"No test-results runs found{scope}.")
        sys.exit(1)

    snapshot = docs[0]
    return {**(snapshot.to_dict() or {}), "run_id": snapshot.id}


def fetch_subcollection(doc_ref, name: str) -> List[Dict[str, Any]]:
    return [d.to_dict() or {} for d in doc_ref.collection(name).stream()]


def safe_get(item: Dict[str, Any], *path: str, default: Any = None) -> Any:
    current: Any = item
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def safe_get_params(item: Dict[str, Any]) -> Dict[str, Any]:
    params = safe_get(item, "request_payload", "params", default={})
    return params if isinstance(params, dict) else {}


def build_param_set_key(params: Dict[str, Any]) -> str:
    """Groups records by the full set of grading params present, whatever they are,
    since not every eval function shares a common param like `comparison`. `symbols`
    and `cases` are excluded — they're per-submission structural data, not grading
    configuration, and including them would make almost every record's key unique."""
    filtered = {k: v for k, v in params.items() if k not in _PARAM_SET_EXCLUDED_KEYS}
    try:
        return json.dumps(filtered, sort_keys=True, default=str)
    except TypeError:
        return str(sorted(filtered.items(), key=lambda kv: str(kv[0])))


def extract_backtick_value(detail: str) -> Optional[str]:
    m = _BACKTICK_RE.search(detail or "")
    return m.group(1) if m else None


def classify_exception_side(detail: str, response: Any, answer: Any) -> str:
    """Determines whether the response or the answer string is the one that failed
    to parse, by matching the backtick-quoted value in the exception detail message."""
    try:
        extracted = extract_backtick_value(detail)
        if extracted is None:
            return "unknown"

        resp_str = str(response) if response is not None else ""
        ans_str = str(answer) if answer is not None else ""
        resp_match = bool(resp_str) and (extracted == resp_str or extracted in resp_str)
        ans_match = bool(ans_str) and (extracted == ans_str or extracted in ans_str)

        if resp_match and not ans_match:
            return "response"
        if ans_match and not resp_match:
            return "answer"
        return "unknown"
    except Exception:
        return "unknown"


def categorize_grade_mismatch(items: List[Dict[str, Any]], top_n: int) -> Dict[str, Any]:
    mismatches = [i for i in items if i.get("error_type") == "**Grade Mismatch**"]

    by_direction: Counter = Counter()
    by_param_set_and_direction: Counter = Counter()

    for item in mismatches:
        param_set_key = build_param_set_key(safe_get_params(item))

        original_grade = item.get("original_grade")
        if original_grade is True:
            direction = "true_became_false"
        elif original_grade is False:
            direction = "false_became_true"
        else:
            direction = "unknown_direction"

        by_direction[direction] += 1
        by_param_set_and_direction[f"{param_set_key}|{direction}"] += 1

    return {
        "total": len(mismatches),
        "by_direction": dict(by_direction),
        "by_param_set_and_direction": dict(by_param_set_and_direction),
        "examples": mismatches[:top_n],
    }


def categorize_grader_exception(items: List[Dict[str, Any]], top_n: int) -> Dict[str, Any]:
    exceptions = [i for i in items if i.get("error_type") == "Grader Exception"]

    by_failing_side: Counter = Counter()
    for item in exceptions:
        response = safe_get(item, "request_payload", "response")
        answer = safe_get(item, "request_payload", "answer")
        side = classify_exception_side(item.get("detail", ""), response, answer)
        by_failing_side[side] += 1

    return {
        "total": len(exceptions),
        "by_failing_side": dict(by_failing_side),
        "examples": exceptions[:top_n],
    }


def categorize_missing_api_field(items: List[Dict[str, Any]], top_n: int) -> Dict[str, Any]:
    missing = [i for i in items if i.get("error_type") == "Missing API Field"]
    return {"total": len(missing), "examples": missing[:top_n]}


def categorize_errors(items: List[Dict[str, Any]], top_n: int) -> Dict[str, Any]:
    by_error_type = Counter(item.get("error_type", "(unknown)") for item in items)
    by_param_set = Counter(build_param_set_key(safe_get_params(item)) for item in items)
    return {
        "total": len(items),
        "by_error_type": dict(by_error_type),
        "by_param_set": dict(by_param_set),
        "grade_mismatch": categorize_grade_mismatch(items, top_n),
        "grader_exception": categorize_grader_exception(items, top_n),
        "missing_api_field": categorize_missing_api_field(items, top_n),
    }


def summarize_light_subcollection(items: List[Dict[str, Any]], type_key: str, top_n: int) -> Dict[str, Any]:
    counter = Counter(item.get(type_key, "(unknown)") for item in items)
    return {
        "total": len(items),
        f"by_{type_key}": dict(counter),
        "examples": items[:top_n],
    }


def summarize_feedback_warnings(items: List[Dict[str, Any]], top_n: int) -> Dict[str, Any]:
    by_warning_type = Counter(item.get("warning_type", "(unknown)") for item in items)
    return {
        "total": len(items),
        "by_warning_type": dict(by_warning_type),
        "non_empty_db_feedback_count": sum(1 for i in items if i.get("db_feedback")),
        "non_empty_api_feedback_count": sum(1 for i in items if i.get("api_feedback")),
        "examples": items[:top_n],
    }


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def build_report(
        run_doc: Dict[str, Any],
        errors_cat: Dict[str, Any],
        network_errors_cat: Dict[str, Any],
        feedback_warnings_cat: Dict[str, Any],
        parsing_warnings_cat: Dict[str, Any],
        run_id: str,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_params": {
            "eval_function_name": run_doc.get("eval_function_name"),
            "source_eval_function_name": run_doc.get("source_eval_function_name"),
            "sql_limit": run_doc.get("sql_limit"),
            "request_delay": run_doc.get("request_delay"),
            "max_concurrency": run_doc.get("max_concurrency"),
            "grade_params_json": run_doc.get("grade_params_json"),
            "seed": run_doc.get("seed"),
            "status": run_doc.get("status"),
            "timestamp": run_doc.get("timestamp"),
            "created_at": run_doc.get("created_at"),
        },
        "run_totals": {
            "pass_count": run_doc.get("pass_count"),
            "total_count": run_doc.get("total_count"),
            "number_of_errors": run_doc.get("number_of_errors"),
            "number_of_feedback_warnings": run_doc.get("number_of_feedback_warnings"),
            "number_of_parsing_warnings": run_doc.get("number_of_parsing_warnings"),
            "pass_rate": run_doc.get("pass_rate"),
        },
        "caveat": _CAVEAT_TEXT,
        "errors_analysis": errors_cat,
        "network_errors_analysis": network_errors_cat,
        "feedback_warnings_analysis": feedback_warnings_cat,
        "parsing_warnings_analysis": parsing_warnings_cat,
    }


def counter_to_bullets(counter: Dict[str, int], indent: str = "  ") -> List[str]:
    items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return [f"{indent}{key}: {count}" for key, count in items]


def print_console_summary(report: Dict[str, Any], project_id: str, top_n: int) -> None:
    run_id = report["run_id"]
    print("-" * 60)
    print(f"Run ID: {run_id}")
    if project_id:
        print(f"Console: {get_firestore_console_link(project_id, 'test-results', run_id)}")
    print("-" * 60)

    rp = report["run_params"]
    print(f"Function: {rp['eval_function_name']}")
    if rp["source_eval_function_name"] and rp["source_eval_function_name"] != rp["eval_function_name"]:
        print(f"Source data: {rp['source_eval_function_name']}")
    print(f"SQL Limit: {rp['sql_limit']}  Seed: {rp['seed']}")
    if rp["grade_params_json"]:
        print(f"Grade Params JSON: {rp['grade_params_json']}")

    rt = report["run_totals"]
    print(f"\nPass: {rt['pass_count']}/{rt['total_count']}  "
          f"Errors: {rt['number_of_errors']}  "
          f"Feedback Warnings: {rt['number_of_feedback_warnings']}  "
          f"Parsing Warnings: {rt['number_of_parsing_warnings']}  "
          f"Pass Rate: {rt['pass_rate']}")

    print("\nNOTE: " + report["caveat"])

    ea = report["errors_analysis"]
    print(f"\n=== Errors (total: {ea['total']}) ===")
    print("By error type:")
    for line in counter_to_bullets(ea["by_error_type"]):
        print(line)
    print("By param set:")
    for line in counter_to_bullets(ea["by_param_set"]):
        print(line)

    gm = ea["grade_mismatch"]
    print(f"\nGrade Mismatch (total: {gm['total']}):")
    print("  By direction:")
    for line in counter_to_bullets(gm["by_direction"], indent="    "):
        print(line)
    print("  By param set + direction:")
    for line in counter_to_bullets(gm["by_param_set_and_direction"], indent="    "):
        print(line)

    ge = ea["grader_exception"]
    print(f"\nGrader Exception (total: {ge['total']}):")
    print("  By failing side:")
    for line in counter_to_bullets(ge["by_failing_side"], indent="    "):
        print(line)

    maf = ea["missing_api_field"]
    print(f"\nMissing API Field (total: {maf['total']})")

    ne = report["network_errors_analysis"]
    print(f"\n=== Network Errors (total: {ne['total']}) ===")
    for line in counter_to_bullets(ne.get("by_error_type", {})):
        print(line)

    fw = report["feedback_warnings_analysis"]
    print(f"\n=== Feedback Warnings (total: {fw['total']}) ===")
    for line in counter_to_bullets(fw["by_warning_type"]):
        print(line)
    print(f"  non-empty db_feedback: {fw['non_empty_db_feedback_count']}  "
          f"non-empty api_feedback: {fw['non_empty_api_feedback_count']}")

    pw = report["parsing_warnings_analysis"]
    print(f"\n=== Parsing Warnings (total: {pw['total']}) ===")
    for line in counter_to_bullets(pw.get("by_warning_type", {})):
        print(line)


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Analyze a stored test-results run's failures from Firestore.")
    parser.add_argument("--run_id", default=None,
                        help="Firestore test-results document ID. Defaults to the most recent run "
                             "(optionally narrowed by --eval_function_name) when omitted.")
    parser.add_argument("--eval_function_name", default=None,
                        help="When --run_id is omitted, only consider the latest run for this eval_function_name.")
    parser.add_argument("--output", default=None, help="Path to write the JSON report (default: analysis_<run_id>.json)")
    parser.add_argument("--top_n", type=int, default=5, help="Max example records kept per category bucket (default: 5)")

    args = parser.parse_args()

    db, project_id = get_firestore_client()
    if args.run_id:
        run_doc = fetch_run(db, args.run_id)
    else:
        run_doc = fetch_latest_run(db, args.eval_function_name)
    run_id = run_doc["run_id"]

    try:
        doc_ref = _get_run_doc_ref(db, run_id)
        errors = fetch_subcollection(doc_ref, "errors")
        network_errors = fetch_subcollection(doc_ref, "network_errors")
        feedback_warnings = fetch_subcollection(doc_ref, "feedback_warnings")
        parsing_warnings = fetch_subcollection(doc_ref, "parsing_warnings")

        errors_cat = categorize_errors(errors, args.top_n)
        network_errors_cat = summarize_light_subcollection(network_errors, "error_type", args.top_n)
        feedback_warnings_cat = summarize_feedback_warnings(feedback_warnings, args.top_n)
        parsing_warnings_cat = summarize_light_subcollection(parsing_warnings, "warning_type", args.top_n)

        report = build_report(
            run_doc, errors_cat, network_errors_cat, feedback_warnings_cat,
            parsing_warnings_cat, run_id,
        )

        output_path = args.output or f"analysis_{run_id}.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=_json_default)

        print_console_summary(report, project_id, args.top_n)
        print(f"\nFull JSON report written to: {output_path}")

    except Exception as e:
        logger.error(f"Fatal error during analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

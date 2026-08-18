import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import firestore

from config import logger
from firestore_client import get_firestore_client, get_firestore_console_link

# Unit-prefix/scale mismatch heuristic (best-effort, not authoritative).
_NUM_UNIT_RE = re.compile(r'^\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(.*)$')
_SI_PREFIX_CHARS = set('mkMGnµc')
_POWERS_OF_TEN = [10 ** e for e in range(-3, 7)]
_RATIO_TOLERANCE = 0.01

# Grader Exception value classification.
_BACKTICK_RE = re.compile(r'`([^`]+)`')
_COMMA_RE = re.compile(r',')
_PLACEHOLDER_RE = re.compile(r'^[\W_]{1,3}$')
_UNIT_SUFFIX_RE = re.compile(r'\d\s+[a-zA-Z/*^()\-]+$')


def _get_run_doc_ref(db: firestore.Client, run_id: str):
    return db.collection("test-results").document(run_id)


def fetch_run(db: firestore.Client, run_id: str) -> Dict[str, Any]:
    snapshot = _get_run_doc_ref(db, run_id).get()
    if not snapshot.exists:
        logger.error(f"No test-results run found with ID: {run_id}")
        sys.exit(1)
    return {**(snapshot.to_dict() or {}), "run_id": run_id}


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


def extract_magnitude_and_unit(value: Any) -> Tuple[Optional[float], str]:
    try:
        s = str(value).strip()
        m = _NUM_UNIT_RE.match(s)
        if not m:
            return None, ""
        return float(m.group(1)), m.group(2).strip()
    except (ValueError, TypeError):
        return None, ""


def strip_si_prefix(unit: str) -> str:
    if len(unit) > 1 and unit[0] in _SI_PREFIX_CHARS:
        return unit[1:]
    return unit


def detect_unit_prefix_mismatch(response: Any, answer: Any) -> bool:
    """Best-effort heuristic: flags likely SI-prefix/scale differences between
    a response and answer string (e.g. '4000 m' vs '4 km'). Not authoritative —
    false positives/negatives are expected."""
    try:
        if response is None or answer is None:
            return False
        mag1, unit1 = extract_magnitude_and_unit(response)
        mag2, unit2 = extract_magnitude_and_unit(answer)
        if mag1 is None or mag2 is None or not unit1 or not unit2:
            return False
        if mag1 == 0 or mag2 == 0:
            return False
        base1, base2 = strip_si_prefix(unit1), strip_si_prefix(unit2)
        if not base1 or not base2:
            return False
        if not (base1 in base2 or base2 in base1):
            return False
        ratio = max(abs(mag1), abs(mag2)) / min(abs(mag1), abs(mag2))
        return any(abs(ratio - p) / p <= _RATIO_TOLERANCE for p in _POWERS_OF_TEN)
    except (ValueError, ZeroDivisionError, TypeError):
        return False


def extract_backtick_value(detail: str) -> Optional[str]:
    m = _BACKTICK_RE.search(detail or "")
    return m.group(1) if m else None


def classify_structural_pattern(value: str) -> str:
    if _COMMA_RE.search(value):
        return "comma_separated"
    if _PLACEHOLDER_RE.match(value):
        return "single_char_or_placeholder"
    if _UNIT_SUFFIX_RE.search(value):
        return "has_unit_like_suffix"
    return "other"


def classify_exception_value(detail: str, response: Any, answer: Any) -> Dict[str, Optional[str]]:
    try:
        extracted = extract_backtick_value(detail)
        if extracted is None:
            return {"side": "unknown", "structural_tag": "other", "extracted_value": None}

        resp_str = str(response) if response is not None else ""
        ans_str = str(answer) if answer is not None else ""
        resp_match = bool(resp_str) and (extracted == resp_str or extracted in resp_str)
        ans_match = bool(ans_str) and (extracted == ans_str or extracted in ans_str)

        if resp_match and not ans_match:
            side = "response"
        elif ans_match and not resp_match:
            side = "answer"
        else:
            side = "unknown"

        return {
            "side": side,
            "structural_tag": classify_structural_pattern(extracted),
            "extracted_value": extracted,
        }
    except Exception:
        return {"side": "unknown", "structural_tag": "other", "extracted_value": None}


def categorize_grade_mismatch(items: List[Dict[str, Any]], top_n: int) -> Dict[str, Any]:
    mismatches = [i for i in items if i.get("error_type") == "**Grade Mismatch**"]

    by_comparison_mode: Counter = Counter()
    by_comparison_mode_and_direction: Counter = Counter()
    flagged_examples: List[Dict[str, Any]] = []
    flagged_count = 0

    for item in mismatches:
        params = safe_get_params(item)
        comparison = params.get("comparison")
        if comparison is None:
            mode_label = "(default)"
        elif comparison == "":
            mode_label = "(empty)"
        else:
            mode_label = str(comparison)
        by_comparison_mode[mode_label] += 1

        original_grade = item.get("original_grade")
        if original_grade is True:
            direction = "true_became_false"
        elif original_grade is False:
            direction = "false_became_true"
        else:
            direction = "unknown_direction"
        by_comparison_mode_and_direction[f"{mode_label}|{direction}"] += 1

        response = safe_get(item, "request_payload", "response")
        answer = safe_get(item, "request_payload", "answer")
        if detect_unit_prefix_mismatch(response, answer):
            flagged_count += 1
            if len(flagged_examples) < top_n:
                flagged_examples.append({
                    "submission_id": item.get("submission_id"),
                    "response": response,
                    "answer": answer,
                    "comparison": mode_label,
                })

    return {
        "total": len(mismatches),
        "by_comparison_mode": dict(by_comparison_mode),
        "by_comparison_mode_and_direction": dict(by_comparison_mode_and_direction),
        "unit_prefix_mismatch_heuristic": {
            "flagged_count": flagged_count,
            "examples": flagged_examples,
            "note": "heuristic, not authoritative",
        },
        "examples": mismatches[:top_n],
    }


def categorize_grader_exception(items: List[Dict[str, Any]], top_n: int) -> Dict[str, Any]:
    exceptions = [i for i in items if i.get("error_type") == "Grader Exception"]

    by_failing_side: Counter = Counter()
    by_side_and_structural_tag: Counter = Counter()
    examples: List[Dict[str, Any]] = []

    for item in exceptions:
        response = safe_get(item, "request_payload", "response")
        answer = safe_get(item, "request_payload", "answer")
        classification = classify_exception_value(item.get("detail", ""), response, answer)
        by_failing_side[classification["side"]] += 1
        by_side_and_structural_tag[f"{classification['side']}|{classification['structural_tag']}"] += 1

        if len(examples) < top_n:
            examples.append({**item, "classification": classification})

    return {
        "total": len(exceptions),
        "by_failing_side": dict(by_failing_side),
        "by_side_and_structural_tag": dict(by_side_and_structural_tag),
        "examples": examples,
    }


def categorize_missing_api_field(items: List[Dict[str, Any]], top_n: int) -> Dict[str, Any]:
    missing = [i for i in items if i.get("error_type") == "Missing API Field"]
    return {"total": len(missing), "examples": missing[:top_n]}


def categorize_param_signals(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    symbols_present_count = 0
    cases_present_count = 0
    rtol_values: List[float] = []

    for item in items:
        params = safe_get_params(item)
        try:
            if bool(params.get("symbols")):
                symbols_present_count += 1
        except TypeError:
            pass
        try:
            if bool(params.get("cases")):
                cases_present_count += 1
        except TypeError:
            pass
        rtol = params.get("rtol")
        if isinstance(rtol, (int, float)) and not isinstance(rtol, bool):
            rtol_values.append(rtol)

    return {
        "symbols_present_count": symbols_present_count,
        "cases_present_count": cases_present_count,
        "rtol_stats": {
            "count_non_null": len(rtol_values),
            "min": min(rtol_values) if rtol_values else None,
            "max": max(rtol_values) if rtol_values else None,
        },
        "total_errors_considered": len(items),
    }


def categorize_errors(items: List[Dict[str, Any]], top_n: int) -> Dict[str, Any]:
    by_error_type = Counter(item.get("error_type", "(unknown)") for item in items)
    return {
        "total": len(items),
        "by_error_type": dict(by_error_type),
        "grade_mismatch": categorize_grade_mismatch(items, top_n),
        "grader_exception": categorize_grader_exception(items, top_n),
        "missing_api_field": categorize_missing_api_field(items, top_n),
        "param_signals": categorize_param_signals(items),
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


def build_signals(errors_cat: Dict[str, Any]) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    by_mode = errors_cat.get("grade_mismatch", {}).get("by_comparison_mode", {})
    for mode, count in by_mode.items():
        if count <= 0:
            continue
        signals.append({
            "topic": f"comparison mode '{mode}'",
            "failure_count": count,
            "interpretation_hint_unsupported_feature": (
                "high count here may indicate this comparison mode is unsupported "
                "or behaves differently in the tested function"
            ),
            "interpretation_hint_param_tuning": (
                "alternatively, this may simply be the most heavily-sampled mode "
                "in this run and/or need different grade_params tuning"
            ),
        })

    by_direction: Counter = Counter()
    for key, count in errors_cat.get("grade_mismatch", {}).get("by_comparison_mode_and_direction", {}).items():
        direction = key.split("|", 1)[-1]
        by_direction[direction] += count
    if by_direction:
        signals.append({
            "topic": "grade mismatch direction",
            "counts": dict(by_direction),
            "interpretation_hint_unsupported_feature": (
                "'true_became_false' suggests the new function is stricter or "
                "missing support for something the old function accepted"
            ),
            "interpretation_hint_param_tuning": (
                "'false_became_true' suggests the new function is more lenient "
                "or parses/compares differently — may just need param adjustment"
            ),
        })

    return signals


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
        signals: List[Dict[str, Any]],
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
        "errors_analysis": errors_cat,
        "network_errors_analysis": network_errors_cat,
        "feedback_warnings_analysis": feedback_warnings_cat,
        "parsing_warnings_analysis": parsing_warnings_cat,
        "signals": signals,
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

    gm = ea["grade_mismatch"]
    print(f"\nGrade Mismatch (total: {gm['total']}):")
    print("  By comparison mode:")
    for line in counter_to_bullets(gm["by_comparison_mode"], indent="    "):
        print(line)
    print("  By comparison mode + direction:")
    for line in counter_to_bullets(gm["by_comparison_mode_and_direction"], indent="    "):
        print(line)
    print(f"  Unit-prefix mismatch heuristic flagged: {gm['unit_prefix_mismatch_heuristic']['flagged_count']} "
          f"(best-effort, not authoritative)")

    ge = ea["grader_exception"]
    print(f"\nGrader Exception (total: {ge['total']}):")
    print("  By failing side:")
    for line in counter_to_bullets(ge["by_failing_side"], indent="    "):
        print(line)
    print("  By side + structural tag:")
    for line in counter_to_bullets(ge["by_side_and_structural_tag"], indent="    "):
        print(line)

    maf = ea["missing_api_field"]
    print(f"\nMissing API Field (total: {maf['total']})")

    ps = ea["param_signals"]
    print(f"\nParam signals (across all errors, total considered: {ps['total_errors_considered']}):")
    print(f"  symbols present: {ps['symbols_present_count']}")
    print(f"  cases present: {ps['cases_present_count']}")
    print(f"  rtol stats: {ps['rtol_stats']}")

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

    print("\n=== Signals ===")
    for signal in report["signals"]:
        print(f"- {signal['topic']}")
        if "failure_count" in signal:
            print(f"    failures: {signal['failure_count']}")
        if "counts" in signal:
            print(f"    counts: {signal['counts']}")
        print(f"    unsupported-feature hint: {signal['interpretation_hint_unsupported_feature']}")
        print(f"    param-tuning hint: {signal['interpretation_hint_param_tuning']}")


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Analyze a stored test-results run's failures from Firestore.")
    parser.add_argument("--run_id", required=True, help="Firestore test-results document ID")
    parser.add_argument("--output", default=None, help="Path to write the JSON report (default: analysis_<run_id>.json)")
    parser.add_argument("--top_n", type=int, default=5, help="Max example records kept per category bucket (default: 5)")

    args = parser.parse_args()

    db, project_id = get_firestore_client()
    run_doc = fetch_run(db, args.run_id)

    try:
        doc_ref = _get_run_doc_ref(db, args.run_id)
        errors = fetch_subcollection(doc_ref, "errors")
        network_errors = fetch_subcollection(doc_ref, "network_errors")
        feedback_warnings = fetch_subcollection(doc_ref, "feedback_warnings")
        parsing_warnings = fetch_subcollection(doc_ref, "parsing_warnings")

        errors_cat = categorize_errors(errors, args.top_n)
        network_errors_cat = summarize_light_subcollection(network_errors, "error_type", args.top_n)
        feedback_warnings_cat = summarize_feedback_warnings(feedback_warnings, args.top_n)
        parsing_warnings_cat = summarize_light_subcollection(parsing_warnings, "warning_type", args.top_n)
        signals = build_signals(errors_cat)

        report = build_report(
            run_doc, errors_cat, network_errors_cat, feedback_warnings_cat,
            parsing_warnings_cat, signals, args.run_id,
        )

        output_path = args.output or f"analysis_{args.run_id}.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=_json_default)

        print_console_summary(report, project_id, args.top_n)
        print(f"\nFull JSON report written to: {output_path}")

    except Exception as e:
        logger.error(f"Fatal error during analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

import csv
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import analyze_run as ar


# ---------------------------------------------------------------------------
# safe_get / safe_get_params
# ---------------------------------------------------------------------------

def test_safe_get_returns_nested_value():
    item = {"a": {"b": {"c": 42}}}
    assert ar.safe_get(item, "a", "b", "c") == 42


def test_safe_get_missing_key_returns_default():
    item = {"a": {"b": 1}}
    assert ar.safe_get(item, "a", "x", default="fallback") == "fallback"


def test_safe_get_non_dict_intermediate_returns_default():
    item = {"a": "not-a-dict"}
    assert ar.safe_get(item, "a", "b", default="fallback") == "fallback"


def test_safe_get_none_value_returns_default():
    item = {"a": None}
    assert ar.safe_get(item, "a", default="fallback") == "fallback"


def test_safe_get_params_returns_dict():
    item = {"request_payload": {"params": {"comparison": "exact"}}}
    assert ar.safe_get_params(item) == {"comparison": "exact"}


def test_safe_get_params_missing_returns_empty_dict():
    assert ar.safe_get_params({}) == {}


def test_safe_get_params_non_dict_returns_empty_dict():
    item = {"request_payload": {"params": "not-a-dict"}}
    assert ar.safe_get_params(item) == {}


# ---------------------------------------------------------------------------
# build_param_set_key
# ---------------------------------------------------------------------------

def test_build_param_set_key_excludes_symbols_and_cases():
    params = {"comparison": "exact", "symbols": ["x"], "cases": ["a", "b"]}
    key = ar.build_param_set_key(params)
    assert "symbols" not in key
    assert "cases" not in key
    assert "comparison" in key


def test_build_param_set_key_is_stable_regardless_of_dict_order():
    key1 = ar.build_param_set_key({"a": 1, "b": 2})
    key2 = ar.build_param_set_key({"b": 2, "a": 1})
    assert key1 == key2


def test_build_param_set_key_falls_back_on_type_error():
    class Unstringable:
        def __str__(self):
            raise TypeError("cannot stringify")

    bad = Unstringable()
    params = {"weird": bad}
    expected = str(sorted(params.items(), key=lambda kv: str(kv[0])))
    assert ar.build_param_set_key(params) == expected


# ---------------------------------------------------------------------------
# extract_backtick_value
# ---------------------------------------------------------------------------

def test_extract_backtick_value_finds_match():
    assert ar.extract_backtick_value("could not parse `3.14`") == "3.14"


def test_extract_backtick_value_no_backticks_returns_none():
    assert ar.extract_backtick_value("no backticks here") is None


def test_extract_backtick_value_none_input_returns_none():
    assert ar.extract_backtick_value(None) is None


# ---------------------------------------------------------------------------
# classify_exception_side
# ---------------------------------------------------------------------------

def test_classify_exception_side_response_match():
    result = ar.classify_exception_side("could not parse `3.14`", response="3.14", answer="2")
    assert result == "response"


def test_classify_exception_side_answer_match():
    result = ar.classify_exception_side("could not parse `2`", response="3.14", answer="2")
    assert result == "answer"


def test_classify_exception_side_no_backtick_is_unknown():
    result = ar.classify_exception_side("generic failure", response="3.14", answer="2")
    assert result == "unknown"


def test_classify_exception_side_both_match_is_unknown():
    result = ar.classify_exception_side("could not parse `x`", response="x", answer="x")
    assert result == "unknown"


def test_classify_exception_side_swallows_exceptions():
    class Explodes:
        def __str__(self):
            raise RuntimeError("boom")

    result = ar.classify_exception_side("could not parse `3`", response=Explodes(), answer=None)
    assert result == "unknown"


# ---------------------------------------------------------------------------
# categorize_grade_mismatch / categorize_grader_exception / categorize_missing_api_field
# ---------------------------------------------------------------------------

def test_categorize_grade_mismatch_buckets_by_direction():
    items = [
        {"error_type": "**Grade Mismatch**", "original_grade": True, "request_payload": {"params": {}}},
        {"error_type": "**Grade Mismatch**", "original_grade": False, "request_payload": {"params": {}}},
        {"error_type": "**Grade Mismatch**", "original_grade": None, "request_payload": {"params": {}}},
        {"error_type": "Other", "original_grade": True, "request_payload": {"params": {}}},
    ]
    result = ar.categorize_grade_mismatch(items, top_n=5)
    assert result["total"] == 3
    assert result["by_direction"] == {
        "true_became_false": 1,
        "false_became_true": 1,
        "unknown_direction": 1,
    }


def test_categorize_grade_mismatch_truncates_examples_to_top_n():
    items = [
        {"error_type": "**Grade Mismatch**", "original_grade": True, "request_payload": {"params": {}}}
        for _ in range(10)
    ]
    result = ar.categorize_grade_mismatch(items, top_n=3)
    assert result["total"] == 10
    assert len(result["examples"]) == 3


def test_categorize_grader_exception_buckets_by_failing_side():
    items = [
        {
            "error_type": "Grader Exception",
            "detail": "could not parse `3.14`",
            "request_payload": {"response": "3.14", "answer": "2"},
        },
        {
            "error_type": "Other",
            "detail": "could not parse `3.14`",
            "request_payload": {"response": "3.14", "answer": "2"},
        },
    ]
    result = ar.categorize_grader_exception(items, top_n=5)
    assert result["total"] == 1
    assert result["by_failing_side"] == {"response": 1}


def test_categorize_missing_api_field_filters_and_truncates():
    items = [
        {"error_type": "Missing API Field"},
        {"error_type": "Missing API Field"},
        {"error_type": "Other"},
    ]
    result = ar.categorize_missing_api_field(items, top_n=1)
    assert result["total"] == 2
    assert len(result["examples"]) == 1


# ---------------------------------------------------------------------------
# categorize_errors
# ---------------------------------------------------------------------------

def test_categorize_errors_aggregates_subcategories():
    items = [
        {"error_type": "**Grade Mismatch**", "original_grade": True, "request_payload": {"params": {"comparison": "exact"}}},
        {"error_type": "Grader Exception", "detail": "`x`", "request_payload": {"params": {}, "response": "x", "answer": "y"}},
        {"error_type": "Missing API Field", "request_payload": {"params": {}}},
    ]
    result = ar.categorize_errors(items, top_n=5)
    assert result["total"] == 3
    assert result["by_error_type"] == {
        "**Grade Mismatch**": 1,
        "Grader Exception": 1,
        "Missing API Field": 1,
    }
    assert result["grade_mismatch"]["total"] == 1
    assert result["grader_exception"]["total"] == 1
    assert result["missing_api_field"]["total"] == 1


# ---------------------------------------------------------------------------
# summarize_light_subcollection / summarize_feedback_warnings
# ---------------------------------------------------------------------------

def test_summarize_light_subcollection_counts_by_key():
    items = [{"error_type": "Timeout"}, {"error_type": "Timeout"}, {"error_type": "DNS"}]
    result = ar.summarize_light_subcollection(items, "error_type", top_n=2)
    assert result["total"] == 3
    assert result["by_error_type"] == {"Timeout": 2, "DNS": 1}
    assert len(result["examples"]) == 2


def test_summarize_light_subcollection_unknown_key_default():
    items = [{}]
    result = ar.summarize_light_subcollection(items, "warning_type", top_n=5)
    assert result["by_warning_type"] == {"(unknown)": 1}


def test_summarize_feedback_warnings_counts_non_empty_feedback():
    items = [
        {"warning_type": "Mismatch", "db_feedback": "well done", "api_feedback": ""},
        {"warning_type": "Mismatch", "db_feedback": "", "api_feedback": "try again"},
        {"warning_type": "Other", "db_feedback": "", "api_feedback": ""},
    ]
    result = ar.summarize_feedback_warnings(items, top_n=5)
    assert result["total"] == 3
    assert result["by_warning_type"] == {"Mismatch": 2, "Other": 1}
    assert result["non_empty_db_feedback_count"] == 1
    assert result["non_empty_api_feedback_count"] == 1


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

def _sample_run_doc():
    return {
        "eval_function_name": "my_func",
        "source_eval_function_name": "my_func",
        "sql_limit": 100,
        "request_delay": 0.0,
        "max_concurrency": 5,
        "grade_params_json": None,
        "seed": 1,
        "status": "completed",
        "timestamp": None,
        "created_at": "2024-01-01T00:00:00",
        "pass_count": 8,
        "total_count": 10,
        "number_of_errors": 2,
        "number_of_feedback_warnings": 0,
        "number_of_parsing_warnings": 0,
        "pass_rate": 80.0,
    }


def test_build_report_omits_feedback_warnings_section_when_none():
    report = ar.build_report(_sample_run_doc(), {"total": 0}, {"total": 0}, None, {"total": 0}, run_id="run1")
    assert "feedback_warnings_analysis" not in report
    assert report["run_id"] == "run1"
    assert report["errors_analysis"] == {"total": 0}


def test_build_report_includes_feedback_warnings_section_when_present():
    report = ar.build_report(
        _sample_run_doc(), {"total": 0}, {"total": 0}, {"total": 3}, {"total": 0}, run_id="run1"
    )
    assert report["feedback_warnings_analysis"] == {"total": 3}


# ---------------------------------------------------------------------------
# counter_to_bullets / format_record_line / print_examples
# ---------------------------------------------------------------------------

def test_counter_to_bullets_sorts_by_count_desc_then_key_asc():
    counter = {"b": 1, "a": 1, "c": 2}
    assert ar.counter_to_bullets(counter, indent="") == ["c: 2", "a: 1", "b: 1"]


def test_format_record_line_includes_submission_and_values():
    item = {"submission_id": "s1", "request_payload": {"answer": "2", "response": "3"}}
    line = ar.format_record_line(item, indent="")
    assert "[s1]" in line
    assert "answer='2'" in line
    assert "response='3'" in line


def test_print_examples_prints_header_and_lines(capsys):
    examples = [{"submission_id": "s1", "request_payload": {}}]
    ar.print_examples(examples, top_n=5)
    out = capsys.readouterr().out
    assert "Examples (showing up to 5):" in out
    assert "[s1]" in out


def test_print_examples_prints_nothing_when_empty(capsys):
    ar.print_examples([], top_n=5)
    out = capsys.readouterr().out
    assert out == ""


# ---------------------------------------------------------------------------
# flatten_records
# ---------------------------------------------------------------------------

def test_flatten_records_produces_one_row_per_record():
    errors = [{"error_type": "**Grade Mismatch**", "submission_id": "s1", "request_payload": {"params": {}}}]
    network_errors = [{"error_type": "Timeout", "submission_id": "s2", "request_payload": {"params": {}}}]
    feedback_warnings = [{"warning_type": "Mismatch", "submission_id": "s3", "request_payload": {"params": {}}}]
    parsing_warnings = [{"warning_type": "Malformed", "submission_id": "s4", "request_payload": {"params": {}}}]

    rows = ar.flatten_records(errors, network_errors, feedback_warnings, parsing_warnings)

    assert [r["category"] for r in rows] == ["errors", "network_errors", "feedback_warnings", "parsing_warnings"]
    assert rows[0]["type"] == "**Grade Mismatch**"
    assert rows[0]["submission_id"] == "s1"
    assert rows[2]["type"] == "Mismatch"


# ---------------------------------------------------------------------------
# _json_default
# ---------------------------------------------------------------------------

def test_json_default_uses_isoformat_when_available():
    dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert ar._json_default(dt) == dt.isoformat()


def test_json_default_falls_back_to_str():
    assert ar._json_default(object()).startswith("<object object at")


# ---------------------------------------------------------------------------
# Firestore-backed I/O functions (mocked)
# ---------------------------------------------------------------------------

def test_fetch_run_returns_merged_dict_when_found():
    snapshot = MagicMock()
    snapshot.exists = True
    snapshot.to_dict.return_value = {"eval_function_name": "my_func"}

    db = MagicMock()
    db.collection.return_value.document.return_value.get.return_value = snapshot

    result = ar.fetch_run(db, "run1")

    assert result == {"eval_function_name": "my_func", "run_id": "run1"}
    db.collection.assert_called_with("test-results")
    db.collection.return_value.document.assert_called_with("run1")


def test_fetch_run_exits_when_not_found():
    snapshot = MagicMock()
    snapshot.exists = False

    db = MagicMock()
    db.collection.return_value.document.return_value.get.return_value = snapshot

    with pytest.raises(SystemExit) as exc_info:
        ar.fetch_run(db, "missing-run")
    assert exc_info.value.code == 1


def test_fetch_latest_run_returns_most_recent_doc():
    doc = MagicMock()
    doc.id = "run123"
    doc.to_dict.return_value = {"eval_function_name": "my_func"}

    db = MagicMock()
    db.collection.return_value.order_by.return_value.limit.return_value.stream.return_value = [doc]

    result = ar.fetch_latest_run(db, eval_function_name=None)

    assert result == {"eval_function_name": "my_func", "run_id": "run123"}
    db.collection.return_value.where.assert_not_called()


def test_fetch_latest_run_scopes_by_eval_function_name():
    doc = MagicMock()
    doc.id = "run123"
    doc.to_dict.return_value = {}

    db = MagicMock()
    query = db.collection.return_value
    query.where.return_value.order_by.return_value.limit.return_value.stream.return_value = [doc]

    ar.fetch_latest_run(db, eval_function_name="my_func")

    query.where.assert_called_once_with("eval_function_name", "==", "my_func")


def test_fetch_latest_run_exits_when_no_runs_found():
    db = MagicMock()
    db.collection.return_value.order_by.return_value.limit.return_value.stream.return_value = []

    with pytest.raises(SystemExit) as exc_info:
        ar.fetch_latest_run(db, eval_function_name=None)
    assert exc_info.value.code == 1


def test_fetch_subcollection_returns_dicts():
    doc1 = MagicMock()
    doc1.to_dict.return_value = {"submission_id": "s1"}
    doc2 = MagicMock()
    doc2.to_dict.return_value = None

    doc_ref = MagicMock()
    doc_ref.collection.return_value.stream.return_value = [doc1, doc2]

    result = ar.fetch_subcollection(doc_ref, "errors")

    assert result == [{"submission_id": "s1"}, {}]
    doc_ref.collection.assert_called_with("errors")


# ---------------------------------------------------------------------------
# write_csv_report (real filesystem via tmp_path)
# ---------------------------------------------------------------------------

def test_write_csv_report_writes_header_and_rows(tmp_path):
    rows = [
        {
            "category": "errors",
            "type": "**Grade Mismatch**",
            "submission_id": "s1",
            "param_set": "{}",
            "original_grade": True,
            "answer": "2",
            "response": "3",
            "message": None,
        }
    ]
    output_path = tmp_path / "report.csv"

    ar.write_csv_report(rows, str(output_path))

    with open(output_path, newline="") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 1
    assert reader[0]["submission_id"] == "s1"
    assert reader[0]["category"] == "errors"


# ---------------------------------------------------------------------------
# print_console_summary
# ---------------------------------------------------------------------------

def test_print_console_summary_prints_key_sections(capsys):
    errors = [{"error_type": "**Grade Mismatch**", "original_grade": True, "request_payload": {"params": {}}}]
    network_errors = [{"error_type": "Timeout"}]
    parsing_warnings = [{"warning_type": "Malformed"}]

    errors_cat = ar.categorize_errors(errors, top_n=5)
    network_errors_cat = ar.summarize_light_subcollection(network_errors, "error_type", top_n=5)
    parsing_warnings_cat = ar.summarize_light_subcollection(parsing_warnings, "warning_type", top_n=5)

    report = ar.build_report(
        _sample_run_doc(), errors_cat, network_errors_cat, None, parsing_warnings_cat, run_id="run123"
    )

    ar.print_console_summary(report, project_id="my-project", top_n=5)

    out = capsys.readouterr().out
    assert "Run ID: run123" in out
    assert "Pass: 8/10" in out
    assert "=== Errors (total: 1) ===" in out
    assert "=== Network Errors (total: 1) ===" in out
    assert "=== Parsing Warnings (total: 1) ===" in out
    assert "feedback_warnings_analysis" not in out


# ---------------------------------------------------------------------------
# main() end-to-end smoke test
# ---------------------------------------------------------------------------

def test_main_writes_json_and_csv_reports(tmp_path, monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["analyze_run.py"])

    run_doc_snapshot = MagicMock()
    run_doc_snapshot.id = "run123"
    run_doc_snapshot.to_dict.return_value = _sample_run_doc()

    error_doc = MagicMock()
    error_doc.to_dict.return_value = {
        "error_type": "**Grade Mismatch**",
        "submission_id": "s1",
        "original_grade": True,
        "request_payload": {"params": {"comparison": "exact"}, "answer": "2", "response": "3"},
    }
    network_error_doc = MagicMock()
    network_error_doc.to_dict.return_value = {"error_type": "Timeout", "submission_id": "s2"}
    parsing_warning_doc = MagicMock()
    parsing_warning_doc.to_dict.return_value = {"warning_type": "Malformed", "submission_id": "s3"}

    subcollections = {
        "errors": [error_doc],
        "network_errors": [network_error_doc],
        "parsing_warnings": [parsing_warning_doc],
    }

    def collection_side_effect(name):
        mock = MagicMock()
        mock.stream.return_value = subcollections.get(name, [])
        return mock

    doc_ref = MagicMock()
    doc_ref.collection.side_effect = collection_side_effect

    db = MagicMock()
    db.collection.return_value.order_by.return_value.limit.return_value.stream.return_value = [run_doc_snapshot]
    db.collection.return_value.document.return_value = doc_ref

    monkeypatch.setattr(ar, "get_firestore_client", lambda: (db, "my-project"))

    ar.main()

    json_path = tmp_path / "analysis_run123.json"
    csv_path = tmp_path / "analysis_run123.csv"
    assert json_path.exists()
    assert csv_path.exists()

    report = json.loads(json_path.read_text())
    assert report["run_id"] == "run123"
    assert report["errors_analysis"]["total"] == 1
    assert report["network_errors_analysis"]["total"] == 1
    assert report["parsing_warnings_analysis"]["total"] == 1
    assert "feedback_warnings_analysis" not in report

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert {row["category"] for row in rows} == {"errors", "network_errors", "parsing_warnings"}

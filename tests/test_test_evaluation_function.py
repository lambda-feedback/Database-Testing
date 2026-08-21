import json
from unittest.mock import MagicMock

import pytest

import test_evaluation_function as tef


# ---------------------------------------------------------------------------
# _parse_exclude_grade_param_args
# ---------------------------------------------------------------------------

def test_parse_exclude_grade_param_args_groups_repeated_keys():
    result = tef._parse_exclude_grade_param_args(["comparison=exact", "comparison=approx", "units=si"])
    assert result == {"comparison": ["exact", "approx"], "units": ["si"]}


def test_parse_exclude_grade_param_args_rejects_malformed_pair():
    with pytest.raises(ValueError):
        tef._parse_exclude_grade_param_args(["no_equals_sign"])


# ---------------------------------------------------------------------------
# _parse_eval_function_param_args
# ---------------------------------------------------------------------------

def test_parse_eval_function_param_args_json_decodes_values():
    result = tef._parse_eval_function_param_args(["physical_quantity=true", "count=3", "name=\"bob\""])
    assert result == {"physical_quantity": True, "count": 3, "name": "bob"}


def test_parse_eval_function_param_args_falls_back_to_raw_string():
    result = tef._parse_eval_function_param_args(["comparison=exact"])
    assert result == {"comparison": "exact"}


def test_parse_eval_function_param_args_rejects_malformed_pair():
    with pytest.raises(ValueError):
        tef._parse_eval_function_param_args(["no_equals_sign"])


# ---------------------------------------------------------------------------
# _parse_max_error_threshold
# ---------------------------------------------------------------------------

def test_parse_max_error_threshold_integer_string():
    assert tef._parse_max_error_threshold("10") == 10
    assert isinstance(tef._parse_max_error_threshold("10"), int)


def test_parse_max_error_threshold_float_string():
    assert tef._parse_max_error_threshold("0.1") == 0.1
    assert isinstance(tef._parse_max_error_threshold("0.1"), float)


def test_parse_max_error_threshold_out_of_range_float_raises():
    with pytest.raises(ValueError):
        tef._parse_max_error_threshold("1.5")


# ---------------------------------------------------------------------------
# start_test
# ---------------------------------------------------------------------------

def _valid_event(**overrides):
    event = {"endpoint": "https://example.com/eval", "eval_function_name": "my_func"}
    event.update(overrides)
    return event


def _patch_common(monkeypatch, tmp_path, results=None, save_side_effect=None):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(tef, "get_firestore_client", lambda: (MagicMock(), "my-project"))
    monkeypatch.setattr(tef, "get_db_connection", lambda: MagicMock())
    monkeypatch.setattr(tef, "fetch_data", lambda *a, **kw: [{"submission_id": "s1"}])
    monkeypatch.setattr(tef, "fetch_excluded_submission_ids", lambda *a, **kw: [])

    results = results or {
        "pass_count": 8,
        "total_count": 10,
        "tested_count": 10,
        "number_of_errors": 2,
        "number_of_network_errors": 0,
        "list_of_errors": [],
        "list_of_network_errors": [],
        "list_of_feedback_warnings": [],
        "list_of_parsing_warnings": [],
    }

    async def fake_test_endpoint(*a, **kw):
        return results

    monkeypatch.setattr(tef, "test_endpoint", fake_test_endpoint)

    if save_side_effect is not None:
        save_mock = MagicMock(side_effect=save_side_effect)
    else:
        save_mock = MagicMock(return_value=("doc123", "https://console.example.com/doc123"))
    monkeypatch.setattr(tef, "save_test_results_to_firestore", save_mock)

    return results, save_mock


def test_start_test_happy_path_writes_report_and_returns_summary(monkeypatch, tmp_path):
    results, save_mock = _patch_common(monkeypatch, tmp_path)

    summary = tef.start_test(_valid_event(), None)

    assert summary["status"] == "success"
    assert summary["pass_count"] == 8
    assert summary["total_count"] == 10
    assert summary["firestore_doc_id"] == "doc123"
    assert summary["firestore_link"] == "https://console.example.com/doc123"

    report_path = tmp_path / "report_data.json"
    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert saved["firestore_doc_id"] == "doc123"


def test_start_test_falls_back_to_local_when_firestore_save_fails(monkeypatch, tmp_path):
    results, save_mock = _patch_common(monkeypatch, tmp_path, save_side_effect=RuntimeError("firestore down"))

    summary = tef.start_test(_valid_event(), None)

    assert summary["status"] == "completed_local_only"
    assert summary["firestore_error"] == "firestore down"
    assert summary["errors"] == results["list_of_errors"]
    assert summary["network_errors"] == results["list_of_network_errors"]

    report_path = tmp_path / "report_data.json"
    saved = json.loads(report_path.read_text())
    assert saved["status"] == "completed_local_only"


def test_start_test_missing_required_fields_exits_and_writes_failure_report(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        tef.start_test({"eval_function_name": "my_func"}, None)
    assert exc_info.value.code == 1

    report_path = tmp_path / "report_data.json"
    saved = json.loads(report_path.read_text())
    assert saved["status"] == "failed"
    assert "endpoint" in saved["error"]

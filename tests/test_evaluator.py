import asyncio

import aiohttp
import pytest

import evaluator as ev


# ---------------------------------------------------------------------------
# _prepare_payload (pure)
# ---------------------------------------------------------------------------

def test_prepare_payload_strips_quotes_from_answer():
    record = {"answer": '"2.5"', "submission": "2.5"}
    payload = ev._prepare_payload(record)
    assert payload["answer"] == "2.5"


def test_prepare_payload_merges_grade_params_and_eval_function_params():
    record = {"answer": "2", "submission": "2", "grade_params": {"comparison": "exact"}}
    payload = ev._prepare_payload(record, eval_function_params={"comparison": "approx", "extra": True})
    assert payload["params"]["comparison"] == "approx"
    assert payload["params"]["extra"] is True


def test_prepare_payload_defaults_cases_and_symbols_when_absent():
    record = {"answer": "2", "submission": "2"}
    payload = ev._prepare_payload(record)
    assert payload["params"]["cases"] == []
    assert payload["params"]["symbols"] == {}


def test_prepare_payload_fills_missing_case_params_with_empty_dict():
    record = {"answer": "2", "submission": "2", "cases": [{"answer": "1"}, {"answer": "2", "params": {"x": 1}}]}
    payload = ev._prepare_payload(record)
    assert payload["params"]["cases"][0]["params"] == {}
    assert payload["params"]["cases"][1]["params"] == {"x": 1}


# ---------------------------------------------------------------------------
# _validate_response (pure)
# ---------------------------------------------------------------------------

def test_validate_response_missing_result_is_grader_exception():
    response = {"error": {"message": "boom", "detail": "`x`"}}
    result = ev._validate_response(response, db_grade=True)
    assert result["error_type"] == "Grader Exception"
    assert result["original_grade"] is True


def test_validate_response_missing_is_correct_field():
    response = {"result": {}}
    result = ev._validate_response(response, db_grade=True)
    assert result["error_type"] == "Missing API Field"


def test_validate_response_int_db_grade_coerced_to_bool():
    response = {"result": {"is_correct": True}}
    assert ev._validate_response(response, db_grade=1) is None
    assert ev._validate_response(response, db_grade=0) is not None


def test_validate_response_match_returns_none():
    response = {"result": {"is_correct": True}}
    assert ev._validate_response(response, db_grade=True) is None


def test_validate_response_mismatch_returns_grade_mismatch():
    response = {"result": {"is_correct": False}}
    result = ev._validate_response(response, db_grade=True)
    assert result["error_type"] == "**Grade Mismatch**"
    assert result["original_grade"] is True


# ---------------------------------------------------------------------------
# _check_feedback (pure)
# ---------------------------------------------------------------------------

def test_check_feedback_none_when_either_side_missing():
    assert ev._check_feedback({"result": {"feedback": "nice"}}, db_feedback=None) is None
    assert ev._check_feedback({"result": {}}, db_feedback="nice") is None


def test_check_feedback_mismatch_returns_warning():
    response = {"result": {"feedback": "api says no"}}
    result = ev._check_feedback(response, db_feedback="db says yes")
    assert result["warning_type"] == "Feedback Mismatch"
    assert result["db_feedback"] == "db says yes"
    assert result["api_feedback"] == "api says no"


def test_check_feedback_match_returns_none():
    response = {"result": {"feedback": "same"}}
    assert ev._check_feedback(response, db_feedback="same") is None


# ---------------------------------------------------------------------------
# Hand-rolled fakes for aiohttp
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status=200, json_data=None, text_data="", json_exc=None):
        self.status = status
        self._json_data = json_data
        self._text_data = text_data
        self._json_exc = json_exc

    async def json(self, content_type=None):
        if self._json_exc:
            raise self._json_exc
        return self._json_data

    async def text(self):
        return self._text_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class FakeRaisingCM:
    """Simulates session.post(...) whose __aenter__ raises a connection error."""

    def __init__(self, exc):
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, *exc_info):
        return False


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.post_calls = []

    def post(self, url, json=None, timeout=None):
        self.post_calls.append(url)
        return self._responses.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


# ---------------------------------------------------------------------------
# _execute_request
# ---------------------------------------------------------------------------

def test_execute_request_success_returns_data():
    session = FakeSession([FakeResponse(status=200, json_data={"result": {"is_correct": True}})])

    data, error = asyncio.run(ev._execute_request(session, "/eval", {"answer": "1"}))

    assert error is None
    assert data == {"result": {"is_correct": True}}


def test_execute_request_non_200_returns_http_error():
    session = FakeSession([FakeResponse(status=500, text_data="server exploded")])

    data, error = asyncio.run(ev._execute_request(session, "/eval", {}))

    assert data is None
    assert error["error_type"] == "HTTP Error"
    assert error["status_code"] == 500


def test_execute_request_json_decode_error():
    session = FakeSession([FakeResponse(status=200, json_exc=ValueError("bad json"), text_data="not json")])

    data, error = asyncio.run(ev._execute_request(session, "/eval", {}))

    assert data is None
    assert error["error_type"] == "JSON Decode Error"


def test_execute_request_exhausts_retries_on_client_error(monkeypatch):
    monkeypatch.setattr(ev, "MAX_RETRY_ATTEMPTS", 1)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    exc = aiohttp.ClientConnectionError("connection reset")
    session = FakeSession([FakeRaisingCM(exc), FakeRaisingCM(exc)])

    data, error = asyncio.run(ev._execute_request(session, "/eval", {}, retry_base_delay=0.0))

    assert data is None
    assert error["error_type"] == "ConnectionError"
    assert error["retries_attempted"] == 1
    assert len(sleeps) == 1


# ---------------------------------------------------------------------------
# test_endpoint
# ---------------------------------------------------------------------------

def _patch_client_session(monkeypatch, responses):
    session = FakeSession(responses)
    monkeypatch.setattr(ev.aiohttp, "ClientSession", lambda connector=None: session)
    monkeypatch.setattr(ev.aiohttp, "TCPConnector", lambda limit=None: None)
    return session


def test_test_endpoint_aggregates_pass_and_grade_mismatch(monkeypatch):
    _patch_client_session(monkeypatch, [
        FakeResponse(status=200, json_data={"result": {"is_correct": True}}),
        FakeResponse(status=200, json_data={"result": {"is_correct": False}}),
    ])
    records = [
        {"submission_id": "s1", "grade": True, "answer": "1", "submission": "1"},
        {"submission_id": "s2", "grade": True, "answer": "2", "submission": "2"},
    ]

    result = asyncio.run(ev.test_endpoint("/eval", records, max_concurrency=1, request_delay=0))

    assert result["pass_count"] == 1
    assert result["total_count"] == 2
    assert result["number_of_errors"] == 1
    assert result["list_of_errors"][0]["error_type"] == "**Grade Mismatch**"


def test_test_endpoint_reclassifies_grader_exception_as_parsing_warning(monkeypatch):
    _patch_client_session(monkeypatch, [
        FakeResponse(status=200, json_data={"error": {"message": "could not parse", "detail": "`x`"}}),
    ])
    records = [
        {
            "submission_id": "s1", "grade": True, "answer": "1", "submission": "1",
            "historical_error_message": "also failed historically",
            "historical_error_detail": "`x`",
        },
    ]

    result = asyncio.run(ev.test_endpoint("/eval", records, max_concurrency=1, request_delay=0))

    assert result["number_of_errors"] == 0
    assert result["pass_count"] == 1
    assert len(result["list_of_parsing_warnings"]) == 1
    assert result["list_of_parsing_warnings"][0]["warning_type"] == "Parsing Error"


def test_test_endpoint_collects_feedback_warnings(monkeypatch):
    _patch_client_session(monkeypatch, [
        FakeResponse(status=200, json_data={"result": {"is_correct": True, "feedback": "api feedback"}}),
    ])
    records = [
        {"submission_id": "s1", "grade": True, "answer": "1", "submission": "1", "feedback": "db feedback"},
    ]

    result = asyncio.run(ev.test_endpoint("/eval", records, max_concurrency=1, request_delay=0))

    assert len(result["list_of_feedback_warnings"]) == 1
    assert result["list_of_feedback_warnings"][0]["warning_type"] == "Feedback Mismatch"


def test_test_endpoint_stops_early_once_network_error_threshold_reached(monkeypatch):
    exc = aiohttp.ClientConnectionError("boom")
    monkeypatch.setattr(ev, "MAX_RETRY_ATTEMPTS", 0)
    _patch_client_session(monkeypatch, [FakeRaisingCM(exc) for _ in range(5)])
    records = [
        {"submission_id": f"s{i}", "grade": True, "answer": "1", "submission": "1"} for i in range(5)
    ]

    result = asyncio.run(
        ev.test_endpoint("/eval", records, max_concurrency=1, request_delay=0, max_error_threshold=1)
    )

    assert result["number_of_network_errors"] >= 1
    assert result["tested_count"] < result["total_count"]

import json
from unittest.mock import MagicMock

import pytest
from google.cloud import firestore

import firestore_client as fc


# ---------------------------------------------------------------------------
# get_firestore_console_link
# ---------------------------------------------------------------------------

def test_get_firestore_console_link_formats_url():
    link = fc.get_firestore_console_link("my-project", "test-results", "run123")
    assert link == (
        "https://console.cloud.google.com/firestore/databases/-default-/data/panel/"
        "test-results/run123?project=my-project"
    )


# ---------------------------------------------------------------------------
# get_firestore_client
# ---------------------------------------------------------------------------

def test_get_firestore_client_builds_client_from_env_credentials(monkeypatch):
    creds_dict = {"type": "service_account", "project_id": "creds-project"}
    monkeypatch.setenv("GOOGLE_CREDENTIALS_JSON", json.dumps(creds_dict))
    monkeypatch.setattr(fc, "GCP_PROJECT_ID", None)

    sentinel_creds = object()
    from_info_mock = MagicMock(return_value=sentinel_creds)
    monkeypatch.setattr(fc.service_account.Credentials, "from_service_account_info", from_info_mock)

    sentinel_db = object()
    client_mock = MagicMock(return_value=sentinel_db)
    monkeypatch.setattr(fc.firestore, "Client", client_mock)

    db, project_id = fc.get_firestore_client()

    assert db is sentinel_db
    assert project_id == "creds-project"
    from_info_mock.assert_called_once_with(creds_dict)
    client_mock.assert_called_once_with(project="creds-project", credentials=sentinel_creds)


def test_get_firestore_client_prefers_explicit_gcp_project_id(monkeypatch):
    monkeypatch.setenv("GOOGLE_CREDENTIALS_JSON", json.dumps({"project_id": "creds-project"}))
    monkeypatch.setattr(fc, "GCP_PROJECT_ID", "explicit-project")
    monkeypatch.setattr(fc.service_account.Credentials, "from_service_account_info", MagicMock(return_value=object()))
    client_mock = MagicMock(return_value=object())
    monkeypatch.setattr(fc.firestore, "Client", client_mock)

    _, project_id = fc.get_firestore_client()

    assert project_id == "explicit-project"
    assert client_mock.call_args.kwargs["project"] == "explicit-project"


def test_get_firestore_client_propagates_invalid_json(monkeypatch):
    monkeypatch.setenv("GOOGLE_CREDENTIALS_JSON", "not-json")

    with pytest.raises(json.JSONDecodeError):
        fc.get_firestore_client()


# ---------------------------------------------------------------------------
# fetch_excluded_submission_ids
# ---------------------------------------------------------------------------

def test_fetch_excluded_submission_ids_returns_empty_when_doc_missing():
    snapshot = MagicMock()
    snapshot.exists = False
    db = MagicMock()
    db.collection.return_value.document.return_value.get.return_value = snapshot

    result = fc.fetch_excluded_submission_ids(db, "my_func")

    assert result == []


def test_fetch_excluded_submission_ids_strips_and_filters_blanks():
    snapshot = MagicMock()
    snapshot.exists = True
    snapshot.get.return_value = [" a ", "", "  ", "b"]
    db = MagicMock()
    db.collection.return_value.document.return_value.get.return_value = snapshot

    result = fc.fetch_excluded_submission_ids(db, "my_func")

    assert result == ["a", "b"]


def test_fetch_excluded_submission_ids_swallows_errors_and_returns_empty():
    db = MagicMock()
    db.collection.side_effect = RuntimeError("boom")

    result = fc.fetch_excluded_submission_ids(db, "my_func")

    assert result == []


# ---------------------------------------------------------------------------
# save_test_results_to_firestore
# ---------------------------------------------------------------------------

def _base_results_summary(pass_count=8, total_count=10):
    return {
        "pass_count": pass_count,
        "total_count": total_count,
        "number_of_errors": 2,
    }


def test_save_test_results_computes_pass_rate_and_returns_ids():
    db = MagicMock()
    doc_ref = db.collection.return_value.document.return_value
    doc_ref.id = "doc123"
    db.collection.return_value.document.return_value = doc_ref
    # no sub-collection writes in this test
    db.collection.return_value.document.return_value.collection = MagicMock()

    doc_id, console_link = fc.save_test_results_to_firestore(
        db, "my-project", _base_results_summary(), {}, [], [], [], []
    )

    assert doc_id == "doc123"
    assert console_link == fc.get_firestore_console_link("my-project", "test-results", "doc123")
    saved_doc = doc_ref.set.call_args[0][0]
    assert saved_doc["pass_rate"] == 80.0


def test_save_test_results_pass_rate_zero_when_no_records():
    db = MagicMock()
    doc_ref = db.collection.return_value.document.return_value
    doc_ref.id = "doc123"

    fc.save_test_results_to_firestore(
        db, "my-project", _base_results_summary(pass_count=0, total_count=0), {}, [], [], [], []
    )

    saved_doc = doc_ref.set.call_args[0][0]
    assert saved_doc["pass_rate"] == 0


def test_save_test_results_skips_batch_for_empty_subcollections():
    db = MagicMock()

    fc.save_test_results_to_firestore(db, "my-project", _base_results_summary(), {}, [], [], [], [])

    db.batch.assert_not_called()


def test_save_test_results_batches_writes_at_500_boundary():
    db = MagicMock()
    errors = [{"submission_id": f"s{i}"} for i in range(501)]

    fc.save_test_results_to_firestore(db, "my-project", _base_results_summary(), {}, errors, [], [], [])

    batch = db.batch.return_value
    assert batch.set.call_count == 501
    assert batch.commit.call_count == 2

from unittest.mock import MagicMock

import pytest
from google.cloud import firestore

import manage_exclusions as me


def _args(**overrides):
    defaults = {"eval_function_name": "my_func", "ids": [], "from_csv": None}
    defaults.update(overrides)
    return MagicMock(**defaults)


# ---------------------------------------------------------------------------
# _handle_add
# ---------------------------------------------------------------------------

def test_handle_add_with_ids_only():
    db = MagicMock()
    doc_ref = db.collection.return_value.document.return_value

    me._handle_add(db, _args(ids=["id1", "id2"]))

    saved = doc_ref.set.call_args[0][0]
    assert isinstance(saved["ids"], firestore.ArrayUnion)
    assert sorted(saved["ids"]._values) == ["id1", "id2"]


def test_handle_add_merges_csv_ids_with_ids_arg(tmp_path):
    csv_path = tmp_path / "ids.csv"
    csv_path.write_text("submission_id\nid_csv_1\n \nid_csv_2\n")

    db = MagicMock()
    doc_ref = db.collection.return_value.document.return_value

    me._handle_add(db, _args(ids=["id_arg"], from_csv=str(csv_path)))

    saved = doc_ref.set.call_args[0][0]
    assert sorted(saved["ids"]._values) == ["id_arg", "id_csv_1", "id_csv_2"]


def test_handle_add_missing_csv_file_exits(tmp_path):
    db = MagicMock()

    with pytest.raises(SystemExit) as exc_info:
        me._handle_add(db, _args(from_csv=str(tmp_path / "missing.csv")))
    assert exc_info.value.code == 1


def test_handle_add_no_ids_exits():
    db = MagicMock()

    with pytest.raises(SystemExit) as exc_info:
        me._handle_add(db, _args())
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# _handle_remove
# ---------------------------------------------------------------------------

def test_handle_remove_dedups_ids():
    db = MagicMock()
    doc_ref = db.collection.return_value.document.return_value

    me._handle_remove(db, _args(ids=["id1", "id1", "id2"]))

    saved = doc_ref.set.call_args[0][0]
    assert isinstance(saved["ids"], firestore.ArrayRemove)
    assert sorted(saved["ids"]._values) == ["id1", "id2"]


# ---------------------------------------------------------------------------
# _handle_list
# ---------------------------------------------------------------------------

def test_handle_list_prints_no_exclusions_when_doc_missing(capsys):
    snapshot = MagicMock()
    snapshot.exists = False
    db = MagicMock()
    db.collection.return_value.document.return_value.get.return_value = snapshot

    me._handle_list(db, _args())

    out = capsys.readouterr().out
    assert "No exclusions configured for my_func." in out


def test_handle_list_prints_ids_and_updated_at(capsys):
    snapshot = MagicMock()
    snapshot.exists = True
    snapshot.get.side_effect = lambda key: {"ids": ["id1", "id2"], "updated_at": "2024-01-01"}[key]
    db = MagicMock()
    db.collection.return_value.document.return_value.get.return_value = snapshot

    me._handle_list(db, _args())

    out = capsys.readouterr().out
    assert "2 excluded submission ID(s) for my_func:" in out
    assert "id1" in out
    assert "id2" in out
    assert "Last updated: 2024-01-01" in out


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------

def test_main_dispatches_add(monkeypatch):
    monkeypatch.setattr("sys.argv", ["manage_exclusions.py", "--eval_function_name", "my_func", "add", "--ids", "id1"])
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(me, "get_firestore_client", lambda: (MagicMock(), "my-project"))
    handle_add = MagicMock()
    monkeypatch.setattr(me, "_handle_add", handle_add)
    monkeypatch.setattr(me, "_handle_remove", MagicMock())
    monkeypatch.setattr(me, "_handle_list", MagicMock())

    me.main()

    handle_add.assert_called_once()


def test_main_dispatches_remove(monkeypatch):
    monkeypatch.setattr("sys.argv", ["manage_exclusions.py", "--eval_function_name", "my_func", "remove", "--ids", "id1"])
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(me, "get_firestore_client", lambda: (MagicMock(), "my-project"))
    handle_remove = MagicMock()
    monkeypatch.setattr(me, "_handle_add", MagicMock())
    monkeypatch.setattr(me, "_handle_remove", handle_remove)
    monkeypatch.setattr(me, "_handle_list", MagicMock())

    me.main()

    handle_remove.assert_called_once()


def test_main_dispatches_list(monkeypatch):
    monkeypatch.setattr("sys.argv", ["manage_exclusions.py", "--eval_function_name", "my_func", "list"])
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(me, "get_firestore_client", lambda: (MagicMock(), "my-project"))
    handle_list = MagicMock()
    monkeypatch.setattr(me, "_handle_add", MagicMock())
    monkeypatch.setattr(me, "_handle_remove", MagicMock())
    monkeypatch.setattr(me, "_handle_list", handle_list)

    me.main()

    handle_list.assert_called_once()


def test_main_add_without_ids_or_csv_errors(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["manage_exclusions.py", "--eval_function_name", "my_func", "add"])
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(me, "get_firestore_client", lambda: (MagicMock(), "my-project"))

    with pytest.raises(SystemExit) as exc_info:
        me.main()
    assert exc_info.value.code == 2

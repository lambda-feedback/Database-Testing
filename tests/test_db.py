from unittest.mock import MagicMock

import pytest

import db


# ---------------------------------------------------------------------------
# get_db_connection
# ---------------------------------------------------------------------------

def test_get_db_connection_builds_url_and_returns_connection(monkeypatch):
    monkeypatch.setenv("DB_USER", "alice")
    monkeypatch.setenv("DB_PASSWORD", "secret")
    monkeypatch.setenv("DB_HOST", "dbhost")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("DB_NAME", "mydb")

    sentinel_conn = object()
    engine = MagicMock()
    engine.connect.return_value = sentinel_conn
    create_engine_mock = MagicMock(return_value=engine)
    monkeypatch.setattr(db, "create_engine", create_engine_mock)

    result = db.get_db_connection()

    assert result is sentinel_conn
    called_url = create_engine_mock.call_args[0][0]
    assert "alice:secret@dbhost:5433/mydb" in called_url
    assert called_url.startswith("postgresql+psycopg2://")


def test_get_db_connection_propagates_connect_errors(monkeypatch):
    engine = MagicMock()
    engine.connect.side_effect = RuntimeError("connection refused")
    monkeypatch.setattr(db, "create_engine", MagicMock(return_value=engine))

    with pytest.raises(RuntimeError, match="connection refused"):
        db.get_db_connection()


# ---------------------------------------------------------------------------
# fetch_data
# ---------------------------------------------------------------------------

def _mock_conn(rows=None):
    conn = MagicMock()
    conn.execute.return_value.mappings.return_value = rows if rows is not None else [{"submission_id": "s1"}]
    return conn


def _main_query_params(conn):
    """The query params dict passed to the second (main-query) conn.execute call."""
    return conn.execute.call_args_list[1][0][1]


def test_fetch_data_returns_rows_as_dicts():
    conn = _mock_conn(rows=[{"submission_id": "s1"}, {"submission_id": "s2"}])

    result = db.fetch_data(conn, sql_limit=10, eval_function_name="my_func", grade_params_json=None, seed=0.5)

    assert result == [{"submission_id": "s1"}, {"submission_id": "s2"}]


def test_fetch_data_clamps_non_positive_sql_limit_to_one():
    conn = _mock_conn()

    db.fetch_data(conn, sql_limit=0, eval_function_name="my_func", grade_params_json=None, seed=0.5)

    assert _main_query_params(conn)["limit_param"] == 1


def test_fetch_data_includes_grade_params_json_when_provided():
    conn = _mock_conn()

    db.fetch_data(conn, sql_limit=10, eval_function_name="my_func", grade_params_json='{"comparison": "exact"}', seed=0.5)

    params = _main_query_params(conn)
    assert params["params_param"] == '{"comparison": "exact"}'


def test_fetch_data_adds_placeholders_for_excluded_ids():
    conn = _mock_conn()

    db.fetch_data(
        conn, sql_limit=10, eval_function_name="my_func", grade_params_json=None, seed=0.5,
        excluded_ids=["id1", "id2"],
    )

    params = _main_query_params(conn)
    assert params["excl_0"] == "id1"
    assert params["excl_1"] == "id2"


def test_fetch_data_adds_placeholders_for_excluded_grade_param_values():
    conn = _mock_conn()

    db.fetch_data(
        conn, sql_limit=10, eval_function_name="my_func", grade_params_json=None, seed=0.5,
        excluded_grade_param_values={"comparison": ["exact", "approx"]},
    )

    params = _main_query_params(conn)
    assert params["gpv_key_0"] == "comparison"
    assert params["gpv_val_0_0"] == "exact"
    assert params["gpv_val_0_1"] == "approx"


def test_fetch_data_propagates_execute_errors():
    conn = MagicMock()
    conn.execute.side_effect = RuntimeError("query failed")

    with pytest.raises(RuntimeError, match="query failed"):
        db.fetch_data(conn, sql_limit=10, eval_function_name="my_func", grade_params_json=None, seed=0.5)

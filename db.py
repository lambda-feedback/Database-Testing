import os
from datetime import datetime
from typing import Dict, Any, List

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection

from config import logger, LOG_LEVEL


def get_db_connection() -> Connection:
    """Establishes a connection to the PostgreSQL database using SQLAlchemy."""
    DB_URL = 'postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}'.format(
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT", "5432"),
        name=os.environ.get("DB_NAME")
    )

    try:
        engine = create_engine(DB_URL)
        conn = engine.connect()
        return conn
    except Exception as e:
        logger.error(f"Could not connect to the database using SQLAlchemy: {e}")
        raise


def fetch_data(conn: Connection, sql_limit: int, eval_function_name: str, grade_params_json: str, seed: float, excluded_ids: List[str] = []) -> List[Dict[str, Any]]:
    """Fetches data using the provided complex query with SQLAlchemy."""
    limit = max(1, sql_limit)

    since_date = datetime(datetime.now().year - 1, 6, 1)

    where_clauses = ["EF.name = :name_param", "S.\"createdAt\" >= :since_date_param"]
    query_params = {
        "name_param": eval_function_name,
        "limit_param": limit,
        "since_date_param": since_date,
    }

    if grade_params_json:
        where_clauses.append("RA.\"gradeParams\"::jsonb = (:params_param)::jsonb")
        query_params["params_param"] = grade_params_json

    if excluded_ids:
        cast_placeholders = ", ".join(f"CAST(:excl_{i} AS UUID)" for i in range(len(excluded_ids)))
        where_clauses.append(f"S.id NOT IN ({cast_placeholders})")
        for i, exc_id in enumerate(excluded_ids):
            query_params[f"excl_{i}"] = exc_id

    where_sql = " AND ".join(where_clauses)

    debug_column = ', S."rawResponse"::json as raw_response' if LOG_LEVEL == 'DEBUG' else ''

    sql_query_template = f"""
            SELECT
                S.id as submission_id, S.submission, S.answer::jsonb, S.grade::int::boolean, S.feedback,
                RA."gradeParams"::json as grade_params,
                (
                    SELECT json_agg(json_build_object(
                        'answer',     RAC.answer::jsonb,
                        'params',     RAC.params,
                        'feedback',   RAC.feedback,
                        'mark',       RAC.mark::int
                    ))
                    FROM "ResponseAreaCase" RAC
                    WHERE RAC."responseAreaId" = RA.id
                ) AS cases,
                json_object_agg(ISYM.code, json_build_object(
                    'latex',   ISYM.symbol,
                    'aliases', ISYM.aliases
                )) FILTER (WHERE ISYM.code IS NOT NULL) AS symbols,
                S."rawResponse"::json -> 'error' ->> 'message' AS historical_error_message,
                S."rawResponse"::json -> 'error' ->> 'detail'  AS historical_error_detail
                {debug_column}
            FROM "Submission" S
                INNER JOIN "ResponseArea" RA ON S."responseAreaId" = RA.id
                INNER JOIN "EvaluationFunction" EF ON RA."evaluationFunctionId" = EF.id
                INNER JOIN "Part" P ON RA."partId" = P.id
                INNER JOIN "QuestionVersion" QV ON P."questionVersionId" = QV.id
                INNER JOIN "Question" Q ON QV."questionId" = Q.id AND QV.id = Q."publishedVersionId"
                LEFT JOIN "InputSymbol" ISYM ON ISYM."responseAreaId" = RA.id
            WHERE
                {where_sql}
            GROUP BY S.id, S.submission, S.answer, S.grade, S.feedback, RA."gradeParams", RA.id{', S."rawResponse"' if LOG_LEVEL == 'DEBUG' else ''}
            ORDER BY RANDOM()
            LIMIT :limit_param;
        """

    try:
        conn.execute(text("SELECT setseed(:seed_param)"), {"seed_param": seed})
        result = conn.execute(text(sql_query_template), query_params)
        data_records = [dict(row) for row in result.mappings()]

    except Exception as e:
        logger.error(f"Error fetching data with query: {e}")
        raise

    logger.info(f"Successfully fetched {len(data_records)} records.")
    return data_records

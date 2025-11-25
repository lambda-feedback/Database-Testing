import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection
import requests

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()

logger = logging.getLogger()
try:
    logger.setLevel(LOG_LEVEL)
except ValueError:
    logger.warning(f"Invalid log level '{LOG_LEVEL}' set. Defaulting to INFO.")
    logger.setLevel(logging.INFO)

DEFAULT_SQL_LIMIT = 1000
MAX_ERROR_THRESHOLD = 50


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


def fetch_data(conn: Connection, sql_limit: int, eval_function_name: str, grade_params_json: str) -> List[
    Dict[str, Any]]:
    """
    Fetches data using the provided complex query with SQLAlchemy.
    Uses parameterized query execution for security.
    """
    limit = max(1, min(sql_limit, DEFAULT_SQL_LIMIT))

    sql_query_template = """
       SELECT DISTINCT ON (S.submission, RA."partId")
            S.submission, S.answer, S.grade, RA."gradeParams"::json as grade_params, RA."partId"
        FROM "Submission" S
            INNER JOIN public."ResponseArea" RA ON S."responseAreaId" = RA.id
            INNER JOIN "EvaluationFunction" EF ON RA."evaluationFunctionId" = EF.id
        WHERE 
            EF.name = :name_param AND 
            RA."gradeParams"::jsonb = (:params_param)::jsonb
        LIMIT :limit_param;
    """

    data_records = []
    try:
        sql_statement = text(sql_query_template)

        result = conn.execute(
            sql_statement,
            {
                "name_param": eval_function_name,
                "params_param": grade_params_json,
                "limit_param": limit
            }
        )

        data_records = [dict(row) for row in result.mappings()]

    except Exception as e:
        logger.error(f"Error fetching data with query: {e}")
        raise

    logger.info(f"Successfully fetched {len(data_records)} records.")
    return data_records


def _prepare_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Constructs the JSON payload for the API request from the DB record."""
    grade_params = record.get('grade_params', {})
    response = record.get('submission')
    answer = record.get('answer').replace('"', '')

    logging.debug(f"Response Type: {response} -  {type(response)}")
    logging.debug(f"Answer Type: {answer} -  {type(answer)}")

    payload = {
        "response": response,
        "answer": answer,
        "params": grade_params
    }
    return payload


def _execute_request(endpoint_path: str, payload: Dict[str, Any]) -> Tuple[
    Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Executes the POST request. Returns (response_data, error_details)."""
    try:
        logging.debug(f"PAYLOAD: {payload}")
        response = requests.post(
            endpoint_path,
            json=payload,
            timeout=10,
        )


        if response.status_code != 200:
            return None, {
                "error_type": "HTTP Error",
                "status_code": response.status_code,
                "message": f"Received status code {response.status_code}.",
                "response_text": response.text[:200]
            }

        try:
            return response.json(), None
        except json.JSONDecodeError:
            return None, {
                "error_type": "JSON Decode Error",
                "message": "API response could not be parsed as JSON.",
                "response_text": response.text[:200]
            }

    except requests.exceptions.RequestException as e:
        return None, {
            "error_type": "ConnectionError",
            "message": str(e)
        }


def _validate_response(response_data: Dict[str, Any], db_grade: Any) -> Optional[Dict[str, Any]]:
    """Compares the API's 'is_correct' result against the historical database grade."""
    result = response_data.get('result')
    api_is_correct = result.get('is_correct')

    expected_is_correct: Optional[bool]
    if isinstance(db_grade, int):
        expected_is_correct = bool(db_grade)
    elif db_grade is None:
        expected_is_correct = None
    else:
        expected_is_correct = db_grade

    if api_is_correct is None:
        return {
            "error_type": "Missing API Field",
            "message": "API response is missing the 'is_correct' field.",
            "original_grade": db_grade
        }

    if api_is_correct == expected_is_correct:
        return None
    else:
        return {
            "error_type": "**Grade Mismatch**",
            "message": f"API result '{api_is_correct}' does not match DB grade '{expected_is_correct}'.",
            "original_grade": db_grade
        }


def test_endpoint(base_endpoint: str, data_records: List[Dict[str, Any]]) -> Dict[
    str, Any]:
    """Main function to test the endpoint, coordinating the smaller helper functions."""
    total_requests = len(data_records)
    successful_requests = 0
    errors = []
    error_count = 0

    endpoint_path = base_endpoint

    logger.info(f"Starting tests on endpoint: {endpoint_path}")

    for i, record in enumerate(data_records):
        submission_id = record.get('id')

        if error_count >= MAX_ERROR_THRESHOLD:
            logger.warning(f"Stopping early! Reached maximum error threshold of {MAX_ERROR_THRESHOLD}.")
            break

        payload = _prepare_payload(record)
        response_data, execution_error = _execute_request(endpoint_path, payload)

        logging.debug(f"RESPONSE: {response_data}")

        if execution_error:
            error_count += 1
            execution_error['submission_id'] = submission_id
            execution_error['original_grade'] = record.get('grade')
            errors.append(execution_error)
            continue

        validation_error = _validate_response(response_data, record.get('grade'))

        if validation_error:
            error_count += 1
            validation_error['submission_id'] = submission_id
            errors.append(validation_error)
        else:
            successful_requests += 1

    return {
        "pass_count": successful_requests,
        "total_count": total_requests,
        "number_of_errors": error_count,
        "list_of_errors": errors
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main Lambda function entry point."""
    conn = None
    try:
        if 'body' in event and isinstance(event['body'], str):
            payload = json.loads(event['body'])
        else:
            payload = event

        endpoint_to_test = payload.get('endpoint')
        sql_limit = int(payload.get('sql_limit', DEFAULT_SQL_LIMIT))

        eval_function_name = payload.get('eval_function_name')
        grade_params_json = payload.get('grade_params_json')

        if not endpoint_to_test or not eval_function_name or not grade_params_json:
            missing_fields = []
            if not endpoint_to_test: missing_fields.append("'endpoint'")
            if not eval_function_name: missing_fields.append("'eval_function_name'")
            if not grade_params_json: missing_fields.append("'grade_params_json'")
            raise ValueError(f"Missing required input fields: {', '.join(missing_fields)}")

        conn = get_db_connection()

        data_for_test = fetch_data(conn, sql_limit, eval_function_name, grade_params_json)

        results = test_endpoint(endpoint_to_test, data_for_test)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "pass_ratio": f"{results['pass_count']}/{results['total_count']}",
                "passes": results['pass_count'],
                "total": results['total_count'],
                "errors_list": results['list_of_errors']
            })
        }

    except Exception as e:
        logger.error(f"Overall function error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
    finally:
        if conn:
            conn.close()
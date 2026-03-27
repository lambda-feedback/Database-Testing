import argparse
import os
import json
import logging
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection
import requests
from google.cloud import firestore
from google.oauth2 import service_account
import base64

# --- Configuration ---
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
DEFAULT_REQUEST_DELAY = float(os.environ.get('REQUEST_DELAY', '2.0'))
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')

logger = logging.getLogger()
try:
    logger.setLevel(LOG_LEVEL)
except ValueError:
    logger.warning(f"Invalid log level '{LOG_LEVEL}' set. Defaulting to INFO.")
    logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logging.info(f"LOG LEVEL: {LOG_LEVEL}")

DEFAULT_SQL_LIMIT = 100
MAX_ERROR_THRESHOLD = 50
REPORT_FILENAME = 'report_data.json'


# --- Firestore Functions ---

def get_firestore_client() -> Tuple[firestore.Client, str]:
    """Initialize and return Firestore client and project ID."""
    try:
        creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        project_id = GCP_PROJECT_ID

        logger.info("Using JSON credentials from environment")
        creds_dict = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        project_id = project_id or creds_dict.get('project_id')
        db = firestore.Client(project=project_id, credentials=credentials)

        logger.info(f"Firestore client initialized successfully for project: {project_id}")
        return db, project_id

    except Exception as e:
        logger.error(f"Failed to initialize Firestore client: {e}")
        raise


def get_firestore_console_link(project_id: str, collection: str, doc_id: str) -> str:
    """Generate a link to the Firestore document in Google Cloud Console."""
    return f"https://console.cloud.google.com/firestore/databases/-default-/data/panel/{collection}/{doc_id}?project={project_id}"


def save_test_results_to_firestore(
        db: firestore.Client,
        project_id: str,
        results_summary: Dict[str, Any],
        test_params: Dict[str, Any],
        errors: List[Dict[str, Any]],
        warnings: List[Dict[str, Any]]
) -> Tuple[str, str]:
    """Save test results to Firestore. Returns (doc_id, console_link)."""
    try:
        # Prepare main document
        test_result_doc = {
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': datetime.utcnow().isoformat(),
            'eval_function_name': test_params.get('eval_function_name'),
            'sql_limit': test_params.get('sql_limit'),
            'request_delay': test_params.get('request_delay'),
            'grade_params_json': test_params.get('grade_params_json', ''),
            'pass_count': results_summary['pass_count'],
            'total_count': results_summary['total_count'],
            'number_of_errors': results_summary['number_of_errors'],
            'number_of_warnings': results_summary.get('number_of_warnings', 0),
            'pass_rate': round(results_summary['pass_count'] / results_summary['total_count'] * 100, 2) if
            results_summary['total_count'] > 0 else 0,
            'status': 'completed'
        }

        # Create main test result document
        doc_ref = db.collection('test-results').document()
        doc_ref.set(test_result_doc)
        doc_id = doc_ref.id

        # Generate console link
        console_link = get_firestore_console_link(project_id, 'test-results', doc_id)

        logger.info(f"Test results saved to Firestore with ID: {doc_id}")
        logger.info(f"View in console: {console_link}")

        # Save errors as subcollection if there are any
        if errors:
            batch = db.batch()
            errors_ref = doc_ref.collection('errors')

            # Batch write errors (max 500 per batch)
            for i, error in enumerate(errors):
                if i > 0 and i % 500 == 0:
                    batch.commit()
                    batch = db.batch()

                error_doc_ref = errors_ref.document()
                batch.set(error_doc_ref, {
                    'timestamp': firestore.SERVER_TIMESTAMP,
                    **error
                })

            batch.commit()
            logger.info(f"Saved {len(errors)} error records to Firestore subcollection")

        # Save warnings as subcollection if there are any
        if warnings:
            batch = db.batch()
            warnings_ref = doc_ref.collection('warnings')

            for i, warning in enumerate(warnings):
                if i > 0 and i % 500 == 0:
                    batch.commit()
                    batch = db.batch()

                warning_doc_ref = warnings_ref.document()
                batch.set(warning_doc_ref, {
                    'timestamp': firestore.SERVER_TIMESTAMP,
                    **warning
                })

            batch.commit()
            logger.info(f"Saved {len(warnings)} warning records to Firestore subcollection")

        return doc_id, console_link

    except Exception as e:
        logger.error(f"Failed to save results to Firestore: {e}")
        raise


# --- Database Functions ---

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
    """Fetches data using the provided complex query with SQLAlchemy."""
    limit = max(1, sql_limit)

    where_clauses = ["EF.name = :name_param"]
    query_params = {
        "name_param": eval_function_name,
        "limit_param": limit
    }

    if grade_params_json:
        where_clauses.append("RA.\"gradeParams\"::jsonb = (:params_param)::jsonb")
        query_params["params_param"] = grade_params_json

    where_sql = " AND ".join(where_clauses)

    sql_query_template = f"""
            SELECT
                S.id as submission_id, S.submission, S.answer, S.grade, S.feedback,
                RA."gradeParams"::json as grade_params,
                json_agg(
                    json_build_object(
                        'answer',   RAC.answer,
                        'params',   RAC.params,
                        'feedback', RAC.feedback
                    )
                ) AS cases
            FROM "Submission" S
                INNER JOIN "ResponseArea" RA ON S."responseAreaId" = RA.id
                INNER JOIN "ResponseAreaCase" RAC ON RAC."responseAreaId" = RA.id
                INNER JOIN "EvaluationFunction" EF ON RA."evaluationFunctionId" = EF.id
            WHERE
                {where_sql}
            GROUP BY S.id, S.submission, S.answer, S.grade, S.feedback, RA."gradeParams"
            LIMIT :limit_param;
        """

    data_records = []
    try:
        result = conn.execute(text(sql_query_template), query_params)
        data_records = [dict(row) for row in result.mappings()]

    except Exception as e:
        logger.error(f"Error fetching data with query: {e}")
        raise

    logger.info(f"Successfully fetched {len(data_records)} records.")
    return data_records


# --- API Request and Validation Helpers ---

def _prepare_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Constructs the JSON payload for the API request from the DB record."""
    grade_params = record.get('grade_params') or {}
    response = record.get('submission')
    answer = record.get('answer').replace('"', '')

    logging.debug(f"Response Type: {response} -  {type(response)}")
    logging.debug(f"Answer Type: {answer} -  {type(answer)}")

    payload = {
        "response": response,
        "answer": answer,
        "params": {
            **grade_params,
            "cases": record.get('cases', []),
        }
    }
    return payload


def _execute_request(endpoint_path: str, payload: Dict[str, Any]) -> Tuple[
    Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Executes the POST request. Returns (response_data, error_details)."""
    try:
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


def _check_feedback(response_data: Dict[str, Any], db_feedback: Any) -> Optional[Dict[str, Any]]:
    """Checks if API feedback matches the stored DB feedback. Returns a warning dict or None."""
    result = response_data.get('result', {})
    api_feedback = result.get('feedback')

    if db_feedback is None or api_feedback is None:
        return None

    if api_feedback != db_feedback:
        return {
            "warning_type": "Feedback Mismatch",
            "message": "API feedback does not match DB feedback.",
            "db_feedback": db_feedback,
            "api_feedback": api_feedback,
        }
    return None


# --- Synchronous Execution Core ---

def test_endpoint(base_endpoint: str, data_records: List[Dict[str, Any]],
                  request_delay: float = DEFAULT_REQUEST_DELAY) -> Dict[str, Any]:
    """Main function to test the endpoint, processing requests sequentially."""
    total_requests = len(data_records)
    successful_requests = 0
    errors = []
    warnings = []
    error_count = 0

    logger.info(f"Starting synchronous tests on endpoint")
    logger.info(f"Request delay: {request_delay} seconds between requests")

    for i, record in enumerate(data_records):
        submission_id = str(record.get('submission_id')) if record.get('submission_id') is not None else None

        if error_count >= MAX_ERROR_THRESHOLD:
            logger.warning(f"Stopping early! Reached maximum error threshold of {MAX_ERROR_THRESHOLD}.")
            break

        # Add delay before request (except for the first one)
        if i > 0 and request_delay > 0:
            time.sleep(request_delay)
            logger.debug(f"Waited {request_delay}s before request {i + 1}/{total_requests}")

        payload = _prepare_payload(record)
        logging.debug(f"REQUEST: {payload}")

        response_data, execution_error = _execute_request(base_endpoint, payload)

        logging.debug(f"RESPONSE: {response_data}")

        if execution_error:
            error_count += 1
            execution_error['submission_id'] = submission_id
            execution_error['original_grade'] = record.get('grade')
            execution_error['request_payload'] = payload
            errors.append(execution_error)
            continue

        validation_error = _validate_response(response_data, record.get('grade'))

        if validation_error:
            error_count += 1
            validation_error['submission_id'] = submission_id
            validation_error['request_payload'] = payload
            errors.append(validation_error)
        else:
            successful_requests += 1

        feedback_warning = _check_feedback(response_data, record.get('feedback'))
        if feedback_warning:
            feedback_warning['submission_id'] = submission_id
            feedback_warning['request_payload'] = payload
            warnings.append(feedback_warning)

        # Log progress every 10 requests
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{total_requests} requests completed")

    return {
        "pass_count": successful_requests,
        "total_count": total_requests,
        "number_of_errors": error_count,
        "list_of_errors": errors,
        "list_of_warnings": warnings,
    }


# --- Main Entry Point ---

def start_test(event, context):
    """Main function entry point. Writes results to report_data.json."""
    # Load environment variables if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available in production

    conn = None

    logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO').upper())
    logger.debug("Starting lambda_handler execution.")

    try:
        # Initialize Firestore client
        db, project_id = get_firestore_client()

        if 'body' in event and isinstance(event['body'], str):
            payload = json.loads(event['body'])
        else:
            payload = event

        endpoint_to_test = payload.get('endpoint')
        sql_limit = int(payload.get('sql_limit', DEFAULT_SQL_LIMIT))
        eval_function_name = payload.get('eval_function_name')
        grade_params_json = payload.get('grade_params_json')
        request_delay = float(payload.get('request_delay', DEFAULT_REQUEST_DELAY))

        if not endpoint_to_test or not eval_function_name:
            missing_fields = []
            if not endpoint_to_test: missing_fields.append("'endpoint'")
            if not eval_function_name: missing_fields.append("'eval_function_name'")
            error_msg = f"Missing required input fields: {', '.join(missing_fields)}"

            # Write error to JSON before raising
            with open(REPORT_FILENAME, 'w') as f:
                json.dump({"status": "failed", "error": error_msg}, f)

            logger.error(error_msg)
            sys.exit(1)

        test_params = {
            'endpoint': endpoint_to_test,
            'eval_function_name': eval_function_name,
            'sql_limit': sql_limit,
            'grade_params_json': grade_params_json,
            'request_delay': request_delay
        }

        conn = get_db_connection()
        data_for_test = fetch_data(conn, sql_limit, eval_function_name, grade_params_json)
        conn.close()
        conn = None
        results = test_endpoint(endpoint_to_test, data_for_test, request_delay)

        # Prepare summary for report_data.json
        results_summary = {
            "status": "success",
            "pass_count": results['pass_count'],
            "total_count": results['total_count'],
            "number_of_errors": results['number_of_errors'],
            "number_of_warnings": len(results['list_of_warnings']),
        }

        # Save to Firestore (required)
        firestore_doc_id, console_link = save_test_results_to_firestore(
            db,
            project_id,
            results_summary,
            test_params,
            results['list_of_errors'],
            results['list_of_warnings'],
        )

        if not firestore_doc_id:
            msg = "Failed to save results to Firestore"
            logger.error(msg)
            with open(REPORT_FILENAME, 'w') as f:
                json.dump({"status": "failed", "error": msg}, f)
            sys.exit(1)

        results_summary['firestore_doc_id'] = firestore_doc_id
        results_summary['firestore_link'] = console_link

        logger.info(f"Results successfully saved to Firestore: {firestore_doc_id}")
        logger.info(f"View results at: {console_link}")

        # WRITE TO FILE FOR GITHUB ACTIONS
        with open(REPORT_FILENAME, 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(json.dumps(results_summary))  # Optional: Print to stdout for logs
        print(f"\n🔗 View results in Firestore: {console_link}")

        return results_summary

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Fatal error: {error_msg}", exc_info=True)

        # Write error to file so the workflow catches it nicely
        with open(REPORT_FILENAME, 'w') as f:
            json.dump({"status": "failed", "error": error_msg}, f)

        sys.exit(1)
    finally:
        if conn:
            conn.close()


# --- CLI Query Commands ---

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run endpoint validation tests.")
    parser.add_argument("--endpoint", required=True, help="API endpoint to test")
    parser.add_argument("--eval_function_name", required=True, help="Evaluation function name")
    parser.add_argument("--sql_limit", type=int, default=100, help="Max number of records to fetch")
    parser.add_argument("--grade_params_json", default="", help="Grade parameters as JSON string")
    parser.add_argument("--request_delay", type=float, default=DEFAULT_REQUEST_DELAY,
                        help="Delay in seconds between requests (default: 2.0)")

    args = parser.parse_args()

    test_event = {
        "endpoint": args.endpoint,
        "eval_function_name": args.eval_function_name,
        "sql_limit": args.sql_limit,
        "grade_params_json": args.grade_params_json,
        "request_delay": args.request_delay,
    }

    print("-" * 50)
    print("Starting test execution...")
    print(f"Endpoint: {test_event['endpoint']}")
    print(f"Function: {test_event['eval_function_name']}")
    print(f"SQL Limit: {test_event['sql_limit']}")
    print("-" * 50)

    results = start_test(test_event, None)

    print("-" * 50)
    print("Test execution finished.")
    print(f"Results saved to Firestore with ID: {results.get('firestore_doc_id')}")
    print(f"🔗 View in console: {results.get('firestore_link')}")
    print("-" * 50)
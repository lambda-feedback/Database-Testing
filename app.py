import os
import json
import logging
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
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

    # Start with mandatory filters
    where_clauses = ["EF.name = :name_param"]
    params = {
        "name_param": eval_function_name,
        "limit_param": limit
    }

    # Conditionally add the gradeParams filter
    if grade_params_json:
        where_clauses.append("RA.\"gradeParams\"::jsonb = (:params_param)::jsonb")
        params["params_param"] = grade_params_json

    # Combine clauses with AND
    where_sql = " AND ".join(where_clauses)

    sql_query_template = f"""
            SELECT
               S.submission, S.answer, S.grade, RA."gradeParams"::json as grade_params
            FROM "Submission" S
                INNER JOIN public."ResponseArea" RA ON S."responseAreaId" = RA.id
                INNER JOIN "EvaluationFunction" EF ON RA."evaluationFunctionId" = EF.id
            WHERE 
                {where_sql}
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


def write_errors_to_csv(errors: List[Dict[str, Any]], filename: str) -> Optional[str]:
    """Write error list to CSV file."""
    if not errors:
        logger.info("No errors to write to CSV.")
        return None

    try:
        filepath = f"/tmp/{filename}"

        # Get all unique keys from all error dictionaries
        fieldnames = set()
        for error in errors:
            fieldnames.update(error.keys())
        fieldnames = sorted(list(fieldnames))

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(errors)

        logger.info(f"CSV file created: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Failed to create CSV: {e}")
        return None


def send_email_with_results(results: Dict[str, Any], csv_path: Optional[str],
                            endpoint: str, eval_function_name: str, recipient_email: str):
    """Send email with test results and CSV attachment."""

    # Get email config from environment variables
    sender_email = os.environ.get('SENDER_EMAIL')
    sender_password = os.environ.get('SENDER_PASSWORD')

    if not all([sender_email, sender_password, recipient_email]):
        logger.warning("Email credentials not configured. Skipping email notification.")
        return

    try:
        # Calculate pass rate
        pass_count = results['pass_count']
        total_count = results['total_count']
        pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0

        # Determine status
        status = "✓ PASSED" if results['number_of_errors'] == 0 else "✗ FAILED"

        # Create email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Endpoint Test Results - {status} - {eval_function_name}"

        # Email body
        body = f"""
Evaluation Function Testing Report
=======================

Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Endpoint: {endpoint}
Evaluation Function: {eval_function_name}

Results Summary:
----------------
Status: {status}
Pass Rate: {pass_rate:.1f}% ({pass_count}/{total_count})
Total Tests: {total_count}
Passed: {pass_count}
Failed: {results['number_of_errors']}

{f"⚠ Warning: Testing stopped early after reaching {MAX_ERROR_THRESHOLD} errors." if results['number_of_errors'] >= MAX_ERROR_THRESHOLD else ""}

{'Detailed error information is attached in the CSV file.' if csv_path else 'No errors encountered - all tests passed!'}

This is an automated notification from the endpoint testing system.
"""

        msg.attach(MIMEText(body, 'plain'))

        # Attach CSV if it exists
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())

            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename={os.path.basename(csv_path)}'
            )
            msg.attach(part)
            logger.info(f"Attached CSV file: {csv_path}")

        # Send email (Gmail SMTP)
        server = smtplib.SMTP('mail.privateemail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()

        logger.info(f"Email sent successfully to {recipient_email}")

    except Exception as e:
        logger.error(f"Failed to send email: {e}")


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
        recipient_email = payload.get('recipient_email')

        if not endpoint_to_test or not eval_function_name:
            missing_fields = []
            if not endpoint_to_test: missing_fields.append("'endpoint'")
            if not eval_function_name: missing_fields.append("'eval_function_name'")
            raise ValueError(f"Missing required input fields: {', '.join(missing_fields)}")

        conn = get_db_connection()

        data_for_test = fetch_data(conn, sql_limit, eval_function_name, grade_params_json)

        results = test_endpoint(endpoint_to_test, data_for_test)

        # Write errors to CSV and send email
        csv_path = None
        if results['list_of_errors']:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f"endpoint_test_errors_{eval_function_name}_{timestamp}.csv"
            csv_path = write_errors_to_csv(results['list_of_errors'], csv_filename)

        # Send email notification with results
        send_email_with_results(results, csv_path, endpoint_to_test, eval_function_name, recipient_email)

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
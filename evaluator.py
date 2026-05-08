import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple

import aiohttp

from config import logger, DEFAULT_REQUEST_DELAY, DEFAULT_MAX_CONCURRENCY, MAX_ERROR_THRESHOLD


def _prepare_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Constructs the JSON payload for the API request from the DB record."""
    grade_params = record.get('grade_params') or {}
    response = record.get('submission')
    answer = record.get('answer').replace('"', '')

    logging.debug(f"Response Type: {response} -  {type(response)}")
    logging.debug(f"Answer Type: {answer} -  {type(answer)}")

    cases = [
        {**case, "params": case["params"] if case.get("params") is not None else {}}
        for case in record.get('cases', []) or []
    ]

    symbols = record.get('symbols') or {}

    return {
        "response": response,
        "answer": answer,
        "params": {
            **grade_params,
            "symbols": symbols,
            "cases": cases,
        }
    }


async def _execute_request(session: aiohttp.ClientSession, endpoint_path: str, payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Executes the POST request. Returns (response_data, error_details)."""
    try:
        async with session.post(
            endpoint_path,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            if response.status != 200:
                return None, {
                    "error_type": "HTTP Error",
                    "status_code": response.status,
                    "message": f"Received status code {response.status}.",
                    "response_text": (await response.text())[:200]
                }

            try:
                return await response.json(content_type=None), None
            except Exception:
                return None, {
                    "error_type": "JSON Decode Error",
                    "message": "API response could not be parsed as JSON.",
                    "response_text": (await response.text())[:200]
                }

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return None, {
            "error_type": "ConnectionError",
            "message": f"{type(e).__name__}: {e}"
        }


def _validate_response(response_data: Dict[str, Any], db_grade: Any) -> Optional[Dict[str, Any]]:
    """Compares the API's 'is_correct' result against the historical database grade."""
    result = response_data.get('result')

    if result is None:
        api_error = response_data.get('error', {})
        return {
            "error_type": "Grader Exception",
            "message": api_error.get('message', 'API returned no result.'),
            "detail": api_error.get('detail', ''),
            "original_grade": db_grade,
        }

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


async def test_endpoint(base_endpoint: str, data_records: List[Dict[str, Any]],
                        request_delay: float = DEFAULT_REQUEST_DELAY,
                        max_concurrency: int = DEFAULT_MAX_CONCURRENCY) -> Dict[str, Any]:
    """Tests the endpoint against all records concurrently, returns aggregated results."""
    total_records = len(data_records)
    successful_requests = 0
    errors = []
    network_errors = []
    warnings = []
    validation_error_count = 0
    network_error_count = 0
    completed_count = 0

    semaphore = asyncio.Semaphore(max_concurrency)
    lock = asyncio.Lock()
    stop_event = asyncio.Event()

    logger.info(f"Starting tests on endpoint with max_concurrency={max_concurrency}, request_delay={request_delay}s")

    async def worker(i: int, record: Dict[str, Any]) -> None:
        nonlocal validation_error_count, network_error_count, completed_count, successful_requests

        if stop_event.is_set():
            return

        async with semaphore:
            if stop_event.is_set():
                return

            submission_id = str(record.get('submission_id')) if record.get('submission_id') is not None else None
            payload = _prepare_payload(record)

            response_data, execution_error = await _execute_request(session, base_endpoint, payload)
            logging.debug(f"[{submission_id}] grade={record.get('grade')} | REQUEST: {payload} | RESPONSE: {response_data or execution_error}")

            async with lock:
                if execution_error:
                    network_error_count += 1
                    execution_error['submission_id'] = submission_id
                    execution_error['original_grade'] = record.get('grade')
                    execution_error['request_payload'] = payload
                    network_errors.append(execution_error)
                    if network_error_count >= MAX_ERROR_THRESHOLD:
                        logger.warning(f"Stopping early! Reached maximum error threshold of {MAX_ERROR_THRESHOLD}.")
                        stop_event.set()
                    completed_count += 1
                    if completed_count % 10 == 0:
                        logger.info(f"Progress: {completed_count}/{total_records} requests completed")
                    return

                validation_error = _validate_response(response_data, record.get('grade'))

                if validation_error:
                    validation_error_count += 1
                    validation_error['submission_id'] = submission_id
                    validation_error['request_payload'] = payload
                    errors.append(validation_error)
                    if validation_error_count >= MAX_ERROR_THRESHOLD:
                        logger.warning(f"Stopping early! Reached maximum error threshold of {MAX_ERROR_THRESHOLD}.")
                        stop_event.set()
                else:
                    successful_requests += 1

                feedback_warning = _check_feedback(response_data, record.get('feedback'))
                if feedback_warning:
                    feedback_warning['submission_id'] = submission_id
                    feedback_warning['request_payload'] = payload
                    warnings.append(feedback_warning)

                completed_count += 1
                if completed_count % 10 == 0:
                    logger.info(f"Progress: {completed_count}/{total_records} requests completed")

    connector = aiohttp.TCPConnector(limit=max_concurrency + 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i, record in enumerate(data_records):
            if stop_event.is_set():
                break
            tasks.append(asyncio.create_task(worker(i, record)))
            if request_delay > 0 and i < total_records - 1:
                await asyncio.sleep(request_delay)
        await asyncio.gather(*tasks, return_exceptions=True)

    return {
        "pass_count": successful_requests,
        "total_count": total_records,
        "tested_count": completed_count,
        "number_of_errors": validation_error_count,
        "number_of_network_errors": network_error_count,
        "list_of_errors": errors,
        "list_of_network_errors": network_errors,
        "list_of_warnings": warnings,
    }

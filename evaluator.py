import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple

import aiohttp
from jsonschema.exceptions import ValidationError

from config import logger, DEFAULT_REQUEST_DELAY, DEFAULT_MAX_CONCURRENCY, MAX_ERROR_THRESHOLD, MAX_RETRY_ATTEMPTS
from mued_schema import MuEdSchema, SchemaLoadError, REQUIRED_ARTEFACT_TYPE, get_schema as get_mued_schema


def _prepare_legacy_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Constructs the legacy Lambda-Feedback JSON payload for the API request from the DB record."""
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


def _prepare_mued_payload(record: Dict[str, Any], schema: MuEdSchema) -> Dict[str, Any]:
    """Constructs a muEd EvaluateRequest JSON payload from the DB record, validated against the
    live muEd OpenAPI schema before being returned."""
    response = record.get('submission')

    payload: Dict[str, Any] = {
        "submission": {
            "type": REQUIRED_ARTEFACT_TYPE,
            "content": {"expression": response},
        },
    }

    answer = record.get('answer')
    if answer is not None:
        if isinstance(answer, str):
            answer = answer.replace('"', '')
        payload["task"] = {
            "title": "Evaluation Task",
            "referenceSolution": {"expression": answer},
        }

    grade_params = record.get('grade_params') or {}
    cases = [
        {**case, "params": case["params"] if case.get("params") is not None else {}}
        for case in record.get('cases', []) or []
    ]
    symbols = record.get('symbols') or {}
    config_params = {**grade_params, "symbols": symbols, "cases": cases}
    if config_params:
        payload["configuration"] = {"params": config_params}

    schema.validate_evaluate_request(payload)
    return payload


async def _execute_request(session: aiohttp.ClientSession, endpoint_path: str, payload: Dict[str, Any], retry_base_delay: float = 0.0) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Executes the POST request with exponential backoff on transient network errors.
    Returns (response_data, error_details).
    """
    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRY_ATTEMPTS + 1):
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
            last_error = e
            if attempt < MAX_RETRY_ATTEMPTS:
                wait = max(retry_base_delay, 1.0) * (2 ** attempt)
                logger.warning(
                    f"Network error on attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS + 1} "
                    f"({type(e).__name__}: {e}). Retrying in {wait:.1f}s..."
                )
                await asyncio.sleep(wait)

    return None, {
        "error_type": "ConnectionError",
        "message": f"{type(last_error).__name__}: {last_error}",
        "retries_attempted": MAX_RETRY_ATTEMPTS,
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


def _validate_mued_response(response_data: Any, db_grade: Any) -> Optional[Dict[str, Any]]:
    """Compares a muEd Feedback[] response's awardedPoints against the historical database grade.
    awardedPoints == 1 is treated as correct, == 0 as incorrect; anything else is a reported mismatch."""
    if not isinstance(response_data, list):
        return {
            "error_type": "Malformed muEd Response",
            "message": f"Expected a JSON array of Feedback items, got {type(response_data).__name__}.",
            "original_grade": db_grade,
        }

    if not response_data:
        return {
            "error_type": "Empty muEd Response",
            "message": "API returned an empty Feedback[] array; cannot determine awardedPoints.",
            "original_grade": db_grade,
        }

    awarded_points = response_data[0].get('awardedPoints')

    expected_is_correct: Optional[bool]
    if isinstance(db_grade, int):
        expected_is_correct = bool(db_grade)
    elif db_grade is None:
        expected_is_correct = None
    else:
        expected_is_correct = db_grade

    if awarded_points not in (0, 1):
        return {
            "error_type": "Unexpected awardedPoints",
            "message": f"awardedPoints={awarded_points!r}; expected 0 or 1 for pass/fail mapping.",
            "original_grade": db_grade,
        }

    api_is_correct = bool(awarded_points)

    if api_is_correct == expected_is_correct:
        return None

    return {
        "error_type": "**Grade Mismatch**",
        "message": f"muEd awardedPoints={awarded_points} (-> {api_is_correct}) does not match DB grade '{expected_is_correct}'.",
        "original_grade": db_grade,
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
                        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
                        mode: str = "legacy",
                        mued_schema_path: Optional[str] = None) -> Dict[str, Any]:
    """Tests the endpoint against all records concurrently, returns aggregated results."""
    total_records = len(data_records)
    successful_requests = 0
    errors = []
    network_errors = []
    feedback_warnings = []
    parsing_warnings = []
    validation_error_count = 0
    network_error_count = 0
    completed_count = 0

    if mode == "mued":
        schema = get_mued_schema(mued_schema_path) if mued_schema_path else get_mued_schema()
        request_path = base_endpoint.rstrip('/') + '/evaluate'
        prepare_fn = lambda record: _prepare_mued_payload(record, schema)
        validate_fn = _validate_mued_response
        check_feedback_fn = None
    else:
        request_path = base_endpoint
        prepare_fn = _prepare_legacy_payload
        validate_fn = _validate_response
        check_feedback_fn = _check_feedback

    semaphore = asyncio.Semaphore(max_concurrency)
    lock = asyncio.Lock()
    stop_event = asyncio.Event()

    logger.info(f"Starting tests on endpoint with mode={mode}, max_concurrency={max_concurrency}, request_delay={request_delay}s")

    async def worker(i: int, record: Dict[str, Any]) -> None:
        nonlocal validation_error_count, network_error_count, completed_count, successful_requests

        if stop_event.is_set():
            return

        async with semaphore:
            if stop_event.is_set():
                return

            submission_id = str(record.get('submission_id')) if record.get('submission_id') is not None else None

            try:
                payload = prepare_fn(record)
            except (SchemaLoadError, ValidationError) as e:
                async with lock:
                    validation_error_count += 1
                    errors.append({
                        "error_type": "Schema Validation Error",
                        "message": str(e),
                        "original_grade": record.get('grade'),
                        "submission_id": submission_id,
                    })
                    if validation_error_count >= MAX_ERROR_THRESHOLD:
                        logger.warning(f"Stopping early! Reached maximum error threshold of {MAX_ERROR_THRESHOLD}.")
                        stop_event.set()
                    completed_count += 1
                return

            response_data, execution_error = await _execute_request(session, request_path, payload, retry_base_delay=request_delay)
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

                validation_error = validate_fn(response_data, record.get('grade'))

                if validation_error:
                    if (record.get('historical_error_message') is not None
                            and validation_error.get('error_type') == 'Grader Exception'):
                        parsing_warnings.append({
                            "warning_type": "Parsing Error",
                            "message": validation_error.get('message'),
                            "detail": validation_error.get('detail', ''),
                            "historical_error_message": record.get('historical_error_message'),
                            "historical_error_detail": record.get('historical_error_detail'),
                            "submission_id": submission_id,
                            "request_payload": payload,
                        })
                        successful_requests += 1
                    else:
                        validation_error_count += 1
                        validation_error['submission_id'] = submission_id
                        validation_error['request_payload'] = payload
                        errors.append(validation_error)
                        if validation_error_count >= MAX_ERROR_THRESHOLD:
                            logger.warning(f"Stopping early! Reached maximum error threshold of {MAX_ERROR_THRESHOLD}.")
                            stop_event.set()
                else:
                    successful_requests += 1

                if check_feedback_fn:
                    feedback_warning = check_feedback_fn(response_data, record.get('feedback'))
                    if feedback_warning:
                        feedback_warning['submission_id'] = submission_id
                        feedback_warning['request_payload'] = payload
                        feedback_warnings.append(feedback_warning)

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
        "list_of_feedback_warnings": feedback_warnings,
        "list_of_parsing_warnings": parsing_warnings,
    }

import argparse
import asyncio
import json
import os
import random
import sys
from typing import Dict, List

from config import logger, DEFAULT_REQUEST_DELAY, DEFAULT_MAX_CONCURRENCY, DEFAULT_SQL_LIMIT, MAX_ERROR_THRESHOLD, REPORT_FILENAME
from db import get_db_connection, fetch_data
from evaluator import test_endpoint
from firestore_client import get_firestore_client, save_test_results_to_firestore, fetch_excluded_submission_ids


def _parse_exclude_grade_param_args(pairs: List[str]) -> Dict[str, List[str]]:
    """Parse repeated --exclude_grade_param KEY=VALUE args into {key: [values]}."""
    result: Dict[str, List[str]] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --exclude_grade_param value '{pair}', expected KEY=VALUE")
        key, _, value = pair.partition("=")
        result.setdefault(key, []).append(value)
    return result


def start_test(event, context):
    """Main function entry point. Writes results to report_data.json."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    conn = None

    logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO').upper())
    logger.debug("Starting lambda_handler execution.")

    try:
        db, project_id = get_firestore_client()

        if 'body' in event and isinstance(event['body'], str):
            payload = json.loads(event['body'])
        else:
            payload = event

        endpoint_to_test = payload.get('endpoint')
        sql_limit = int(payload.get('sql_limit', DEFAULT_SQL_LIMIT))
        eval_function_name = payload.get('eval_function_name')
        source_eval_function_name = payload.get('source_eval_function_name') or eval_function_name
        grade_params_json = payload.get('grade_params_json')
        exclude_grade_params = payload.get('exclude_grade_params') or {}
        request_delay = float(payload.get('request_delay', DEFAULT_REQUEST_DELAY))
        max_concurrency = int(payload.get('max_concurrency', DEFAULT_MAX_CONCURRENCY))
        max_error_threshold = int(payload.get('max_error_threshold', MAX_ERROR_THRESHOLD))
        seed = payload.get('seed')
        if seed is None:
            seed = random.uniform(-1.0, 1.0)
        else:
            seed = float(seed)
        logger.info(f"Using random seed: {seed}")

        if not endpoint_to_test or not eval_function_name:
            missing_fields = []
            if not endpoint_to_test: missing_fields.append("'endpoint'")
            if not eval_function_name: missing_fields.append("'eval_function_name'")
            error_msg = f"Missing required input fields: {', '.join(missing_fields)}"
            with open(REPORT_FILENAME, 'w') as f:
                json.dump({"status": "failed", "error": error_msg}, f)
            logger.error(error_msg)
            sys.exit(1)

        test_params = {
            'endpoint': endpoint_to_test,
            'eval_function_name': eval_function_name,
            'source_eval_function_name': source_eval_function_name,
            'sql_limit': sql_limit,
            'grade_params_json': grade_params_json,
            'request_delay': request_delay,
            'max_concurrency': max_concurrency,
            'max_error_threshold': max_error_threshold,
            'seed': seed,
        }

        excluded_ids = fetch_excluded_submission_ids(db, source_eval_function_name)
        if excluded_ids:
            logger.info(f"Excluding {len(excluded_ids)} submission ID(s) from sample: {excluded_ids}")
        else:
            logger.info("No submission ID exclusions configured for this function.")
        if exclude_grade_params:
            logger.info(f"Excluding records matching gradeParams: {exclude_grade_params}")
        conn = get_db_connection()
        data_for_test = fetch_data(conn, sql_limit, source_eval_function_name, grade_params_json, seed, excluded_ids, exclude_grade_params)
        conn.close()
        conn = None
        results = asyncio.run(test_endpoint(endpoint_to_test, data_for_test, request_delay, max_concurrency, max_error_threshold))

        results_summary = {
            "status": "success",
            "pass_count": results['pass_count'],
            "total_count": results['total_count'],
            "tested_count": results['tested_count'],
            "number_of_errors": results['number_of_errors'],
            "number_of_network_errors": results['number_of_network_errors'],
            "number_of_feedback_warnings": len(results['list_of_feedback_warnings']),
            "number_of_parsing_warnings": len(results['list_of_parsing_warnings']),
            "seed": seed,
            "eval_function_name": eval_function_name,
            "source_eval_function_name": source_eval_function_name,
        }

        try:
            firestore_doc_id, console_link = save_test_results_to_firestore(
                db,
                project_id,
                results_summary,
                test_params,
                results['list_of_errors'],
                results['list_of_network_errors'],
                results['list_of_feedback_warnings'],
                results['list_of_parsing_warnings'],
            )
            results_summary['firestore_doc_id'] = firestore_doc_id
            results_summary['firestore_link'] = console_link
            logger.info(f"Results successfully saved to Firestore: {firestore_doc_id}")
            logger.info(f"View results at: {console_link}")
        except Exception as e:
            logger.error(f"Failed to save results to Firestore: {e}. Saving results locally instead.", exc_info=True)
            results_summary['status'] = 'completed_local_only'
            results_summary['firestore_error'] = str(e)
            results_summary['errors'] = results['list_of_errors']
            results_summary['network_errors'] = results['list_of_network_errors']
            results_summary['feedback_warnings'] = results['list_of_feedback_warnings']
            results_summary['parsing_warnings'] = results['list_of_parsing_warnings']

        with open(REPORT_FILENAME, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)

        print(json.dumps(results_summary, default=str))
        if results_summary.get('firestore_link'):
            print(f"\n🔗 View results in Firestore: {results_summary['firestore_link']}")
        else:
            print(f"\nFirestore save failed — full results saved locally to {REPORT_FILENAME}.")

        return results_summary

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Fatal error: {error_msg}", exc_info=True)
        with open(REPORT_FILENAME, 'w') as f:
            json.dump({"status": "failed", "error": error_msg}, f)
        sys.exit(1)
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run endpoint validation tests.")
    parser.add_argument("--endpoint", required=True, help="API endpoint to test")
    parser.add_argument("--eval_function_name", required=True, help="Evaluation function name")
    parser.add_argument("--source_eval_function_name", default=None,
                        help="Evaluation function whose historical DB data and exclusion list to use as the data "
                             "source. Defaults to --eval_function_name (self-test) when omitted. Set this to a "
                             "different/deprecated function's name to test its historical submissions against the "
                             "--eval_function_name endpoint (cross-function test). Note: --grade_params_json, if "
                             "provided, still filters by this SOURCE function's gradeParams shape, not the target's.")
    parser.add_argument("--sql_limit", type=int, default=100, help="Max number of records to fetch")
    parser.add_argument("--grade_params_json", default="", help="Grade parameters as JSON string")
    parser.add_argument(
        "--exclude_grade_param", action="append", default=[], metavar="KEY=VALUE",
        help="Exclude records where gradeParams[KEY] equals VALUE (text comparison). "
             "Repeatable: multiple values for the same KEY are OR'd (excluded if any match); "
             "different KEYs are AND'd (each independently narrows the sample). "
             "Example: --exclude_grade_param comparison=buckinghamPi"
    )
    parser.add_argument("--request_delay", type=float, default=DEFAULT_REQUEST_DELAY,
                        help="Delay in seconds between dispatching each request (default: 0.0)")
    parser.add_argument("--max_concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY,
                        help="Max concurrent in-flight requests (default: 5)")
    parser.add_argument("--max_error_threshold", type=int, default=MAX_ERROR_THRESHOLD,
                        help=f"Stop early once either the validation-error or network-error count reaches this "
                             f"value (default: {MAX_ERROR_THRESHOLD})")
    parser.add_argument("--seed", type=float, default=None,
                        help="Random seed for reproducible sampling (float in [-1.0, 1.0]). Auto-generated if omitted.")

    args = parser.parse_args()

    test_event = {
        "endpoint": args.endpoint,
        "eval_function_name": args.eval_function_name,
        "source_eval_function_name": args.source_eval_function_name,
        "sql_limit": args.sql_limit,
        "grade_params_json": args.grade_params_json,
        "exclude_grade_params": _parse_exclude_grade_param_args(args.exclude_grade_param),
        "request_delay": args.request_delay,
        "max_concurrency": args.max_concurrency,
        "max_error_threshold": args.max_error_threshold,
        "seed": args.seed,
    }

    print("-" * 50)
    print("Starting test execution...")
    resolved_source = args.source_eval_function_name or args.eval_function_name
    print(f"Endpoint: {test_event['endpoint']}")
    if resolved_source != args.eval_function_name:
        print(f"Source data: {resolved_source}  ->  Testing against: {args.eval_function_name}")
    else:
        print(f"Function: {args.eval_function_name}")
    print(f"SQL Limit: {test_event['sql_limit']}")
    print("-" * 50)

    results = start_test(test_event, None)

    print("-" * 50)
    print("Test execution finished.")
    if results.get('firestore_doc_id'):
        print(f"Results saved to Firestore with ID: {results['firestore_doc_id']}")
        print(f"🔗 View in console: {results['firestore_link']}")
    else:
        print(f"Firestore save failed ({results.get('firestore_error')}). Full results saved locally to {REPORT_FILENAME}.")
    print("-" * 50)

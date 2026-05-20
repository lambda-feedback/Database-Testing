import argparse
import asyncio
import json
import os
import random
import sys

from config import logger, DEFAULT_REQUEST_DELAY, DEFAULT_MAX_CONCURRENCY, DEFAULT_SQL_LIMIT, REPORT_FILENAME
from db import get_db_connection, fetch_data
from evaluator import test_endpoint
from firestore_client import get_firestore_client, save_test_results_to_firestore, fetch_excluded_submission_ids


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
        grade_params_json = payload.get('grade_params_json')
        request_delay = float(payload.get('request_delay', DEFAULT_REQUEST_DELAY))
        max_concurrency = int(payload.get('max_concurrency', DEFAULT_MAX_CONCURRENCY))
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
            'sql_limit': sql_limit,
            'grade_params_json': grade_params_json,
            'request_delay': request_delay,
            'max_concurrency': max_concurrency,
            'seed': seed,
        }

        excluded_ids = fetch_excluded_submission_ids(db, eval_function_name)
        if excluded_ids:
            logger.info(f"Excluding {len(excluded_ids)} submission ID(s) from sample: {excluded_ids}")
        else:
            logger.info("No submission ID exclusions configured for this function.")
        conn = get_db_connection()
        data_for_test = fetch_data(conn, sql_limit, eval_function_name, grade_params_json, seed, excluded_ids)
        conn.close()
        conn = None
        results = asyncio.run(test_endpoint(endpoint_to_test, data_for_test, request_delay, max_concurrency))

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
        }

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

        with open(REPORT_FILENAME, 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(json.dumps(results_summary))
        print(f"\n🔗 View results in Firestore: {console_link}")

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
    parser.add_argument("--sql_limit", type=int, default=100, help="Max number of records to fetch")
    parser.add_argument("--grade_params_json", default="", help="Grade parameters as JSON string")
    parser.add_argument("--request_delay", type=float, default=DEFAULT_REQUEST_DELAY,
                        help="Delay in seconds between dispatching each request (default: 0.0)")
    parser.add_argument("--max_concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY,
                        help="Max concurrent in-flight requests (default: 5)")
    parser.add_argument("--seed", type=float, default=None,
                        help="Random seed for reproducible sampling (float in [-1.0, 1.0]). Auto-generated if omitted.")

    args = parser.parse_args()

    test_event = {
        "endpoint": args.endpoint,
        "eval_function_name": args.eval_function_name,
        "sql_limit": args.sql_limit,
        "grade_params_json": args.grade_params_json,
        "request_delay": args.request_delay,
        "max_concurrency": args.max_concurrency,
        "seed": args.seed,
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

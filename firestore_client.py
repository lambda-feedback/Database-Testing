import os
import json
import base64
from datetime import datetime
from typing import Dict, Any, List, Tuple

from google.cloud import firestore
from google.oauth2 import service_account

from config import logger, GCP_PROJECT_ID


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


def fetch_excluded_submission_ids(db: firestore.Client, eval_function_name: str) -> List[str]:
    """Fetch the list of excluded submission IDs for an eval function. Returns [] on any error."""
    try:
        snapshot = db.collection("excluded-submissions").document(eval_function_name).get()
        if not snapshot.exists:
            logger.info(f"No exclusion document found for {eval_function_name}, proceeding with no exclusions.")
            return []
        ids = snapshot.get("ids") or []
        ids = [str(i).strip() for i in ids if str(i).strip()]
        logger.info(f"Fetched {len(ids)} excluded submission IDs for {eval_function_name}.")
        return ids
    except Exception as e:
        logger.warning(f"Could not fetch excluded submission IDs: {e}. Proceeding with no exclusions.")
        return []


def save_test_results_to_firestore(
        db: firestore.Client,
        project_id: str,
        results_summary: Dict[str, Any],
        test_params: Dict[str, Any],
        errors: List[Dict[str, Any]],
        network_errors: List[Dict[str, Any]],
        feedback_warnings: List[Dict[str, Any]],
        parsing_warnings: List[Dict[str, Any]]
) -> Tuple[str, str]:
    """Save test results to Firestore. Returns (doc_id, console_link)."""
    try:
        test_result_doc = {
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': datetime.utcnow().isoformat(),
            'eval_function_name': test_params.get('eval_function_name'),
            'sql_limit': test_params.get('sql_limit'),
            'request_delay': test_params.get('request_delay'),
            'max_concurrency': test_params.get('max_concurrency'),
            'grade_params_json': test_params.get('grade_params_json', ''),
            'api_mode': test_params.get('api_mode', 'legacy'),
            'seed': test_params.get('seed'),
            'pass_count': results_summary['pass_count'],
            'total_count': results_summary['total_count'],
            'number_of_errors': results_summary['number_of_errors'],
            'number_of_feedback_warnings': results_summary.get('number_of_feedback_warnings', 0),
            'number_of_parsing_warnings': results_summary.get('number_of_parsing_warnings', 0),
            'pass_rate': round(results_summary['pass_count'] / results_summary['total_count'] * 100, 2) if
            results_summary['total_count'] > 0 else 0,
            'status': 'completed'
        }

        doc_ref = db.collection('test-results').document()
        doc_ref.set(test_result_doc)
        doc_id = doc_ref.id

        console_link = get_firestore_console_link(project_id, 'test-results', doc_id)

        logger.info(f"Test results saved to Firestore with ID: {doc_id}")
        logger.info(f"View in console: {console_link}")

        def _write_subcollection(name: str, items: List[Dict[str, Any]]) -> None:
            if not items:
                return
            batch = db.batch()
            sub_ref = doc_ref.collection(name)
            for i, item in enumerate(items):
                if i > 0 and i % 500 == 0:
                    batch.commit()
                    batch = db.batch()
                batch.set(sub_ref.document(), {'timestamp': firestore.SERVER_TIMESTAMP, **item})
            batch.commit()
            logger.info(f"Saved {len(items)} records to Firestore subcollection '{name}'")

        _write_subcollection('errors', errors)
        _write_subcollection('network_errors', network_errors)
        _write_subcollection('feedback_warnings', feedback_warnings)
        _write_subcollection('parsing_warnings', parsing_warnings)

        return doc_id, console_link

    except Exception as e:
        logger.error(f"Failed to save results to Firestore: {e}")
        raise

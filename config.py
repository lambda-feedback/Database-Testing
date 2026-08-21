import os
import logging

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
DEFAULT_REQUEST_DELAY = float(os.environ.get('REQUEST_DELAY', '0.0'))
DEFAULT_MAX_CONCURRENCY = int(os.environ.get('MAX_CONCURRENCY', '5'))
MAX_RETRY_ATTEMPTS = int(os.environ.get('MAX_RETRY_ATTEMPTS', '3'))
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')

DEFAULT_SQL_LIMIT = 100
MAX_ERROR_THRESHOLD = 50
REPORT_FILENAME = 'report_data.json'

DEFAULT_MUED_SCHEMA_PATH = os.environ.get(
    'MUED_SCHEMA_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor', 'mued-api', 'dist', 'openapi.yml')
)

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

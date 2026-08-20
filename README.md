# Database-Testing

Test evaluation functions against previous student submissions.

This toolkit pulls historical student submissions out of the production Postgres database, replays each one against a live evaluation-function HTTP endpoint, and compares the endpoint's response to the originally recorded grade. Results (pass/fail counts plus every error and warning) are persisted to Firestore, with a local `report_data.json` fallback if Firestore is unavailable. It's used both to catch regressions before deploying an evaluation function and to test a new/refactored function against an old function's historical data ("cross-function" testing).

It ships as three CLI scripts:

| Script | Purpose |
| --- | --- |
| [`test_evaluation_function.py`](#usage-test_evaluation_functionpy) | Run a test against historical submissions |
| [`analyze_run.py`](#usage-analyze_runpy) | Inspect a stored run's failures in detail |
| [`manage_exclusions.py`](#usage-manage_exclusionspy) | Manage the list of submission IDs to skip in future runs |

## Prerequisites & Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

- **Postgres access**: `psycopg2-binary`, `SQLAlchemy`
- **Async HTTP test runner**: `aiohttp`
- **Firestore**: `google-cloud-firestore`, `google-auth`, `google-auth-oauthlib`, `google-auth-httplib2`
- **Local config**: `python-dotenv` (loads a `.env` file automatically if present)

### Environment variables

No `.env.example` is checked in, so create a local `.env` with the following keys:

| Variable | Required | Default | Used for |
| --- | --- | --- | --- |
| `DB_USER` | Yes | — | Postgres connection |
| `DB_PASSWORD` | Yes | — | Postgres connection |
| `DB_HOST` | Yes | — | Postgres connection |
| `DB_PORT` | No | `5432` | Postgres connection |
| `DB_NAME` | Yes | — | Postgres connection |
| `GCP_PROJECT_ID` | Yes | — | Firestore project |
| `GOOGLE_CREDENTIALS_JSON` | Yes | — | Firestore service account credentials (JSON, as a string) |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |
| `REQUEST_DELAY` | No | `0.0` | Default for `--request_delay` |
| `MAX_CONCURRENCY` | No | `5` | Default for `--max_concurrency` |
| `MAX_RETRY_ATTEMPTS` | No | `3` | Retry attempts for failed HTTP requests in `evaluator.py` |

## Usage: `test_evaluation_function.py`

Fetches a random sample of historical submissions for an evaluation function, replays them against `--endpoint`, and records pass/fail results.

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--endpoint` | Yes | — | API endpoint to test |
| `--eval_function_name` | Yes | — | Evaluation function name (also the name used to save results and, when `--source_eval_function_name` is omitted, to look up historical data and exclusions) |
| `--source_eval_function_name` | No | `--eval_function_name` | Evaluation function whose historical DB data *and exclusion list* to use as the data source. Set this to a different/deprecated function's name to replay its historical submissions against the `--eval_function_name` endpoint (cross-function test). Note: `--grade_params_json`, if provided, still filters by the **source** function's `gradeParams` shape, not the target's. |
| `--sql_limit` | No | `100` | Max number of records to fetch |
| `--grade_params_json` | No | `""` | Filter fetched records to those whose `gradeParams` match this JSON string exactly |
| `--exclude_grade_param` | No | `[]` | Repeatable `KEY=VALUE`. Excludes records where `gradeParams[KEY] == VALUE`. Multiple values for the same key are OR'd; different keys are AND'd. Example: `--exclude_grade_param comparison=buckinghamPi` |
| `--eval_function_param` | No | `[]` | Repeatable `KEY=VALUE`. Adds/overrides a param sent to the evaluation function under test, regardless of the record's stored `gradeParams`. Value is JSON-decoded (`true`/`false`/numbers/quoted strings), falling back to a raw string. Example: `--eval_function_param physical_quantity=true` |
| `--request_delay` | No | `REQUEST_DELAY` env / `0.0` | Delay in seconds between dispatching each request |
| `--max_concurrency` | No | `MAX_CONCURRENCY` env / `5` | Max concurrent in-flight requests |
| `--max_error_threshold` | No | `50` | Stop early once the validation-error or network-error count reaches this value. A whole number is an absolute error count; a float in `[0.0, 1.0]` is a fraction of the total records tested (e.g. `0.1` = 10%) |
| `--seed` | No | random in `[-1.0, 1.0]` | Random seed for reproducible sampling; echoed into the saved results |

### Examples

```bash
# Minimal run
python3 test_evaluation_function.py \
  --endpoint https://example.com/eval \
  --eval_function_name my_function

# Excluding some gradeParams and overriding a param sent to the endpoint
python3 test_evaluation_function.py \
  --endpoint https://example.com/eval \
  --eval_function_name my_function \
  --exclude_grade_param comparison=buckinghamPi \
  --eval_function_param physical_quantity=true

# Cross-function test: replay old_function's historical data against my_function's endpoint
python3 test_evaluation_function.py \
  --endpoint https://example.com/eval \
  --eval_function_name my_function \
  --source_eval_function_name old_function
```

### Behavior notes

- Loads `.env` automatically.
- Auto-fetches the exclusion list from Firestore, keyed by the resolved source function name (`--source_eval_function_name` if set, else `--eval_function_name`).
- Always writes `report_data.json` in the working directory.
- Attempts to save full results to Firestore (`test-results` collection). If that fails, it falls back to a local-only save — `report_data.json` still contains everything, and `status` is set to `completed_local_only`.
- Prints a Firestore console link on success.

### Serverless entry point

`start_test(event, context)` in `test_evaluation_function.py` is an alternate, non-CLI entry point (e.g. for a Cloud Function/Lambda trigger). It takes the same parameters as JSON keys in `event` (or `event['body']` as a JSON string) instead of CLI flags.

## Usage: `analyze_run.py`

Digs into one stored run's failures — errors, network errors, feedback warnings, parsing warnings — beyond the summary counts saved by `test_evaluation_function.py`.

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--run_id` | No | most recent run | Firestore `test-results` document ID |
| `--eval_function_name` | No | — | When `--run_id` is omitted, narrows the "most recent run" lookup to this function |
| `--output` | No | `analysis_<run_id>.json` | Path to write the JSON report |
| `--top_n` | No | `5` | Max example records kept per category in the JSON report only (the CSV export is always uncapped) |

### Output artifacts

Each invocation writes two files plus a console summary:

- **JSON report** at `--output` (or the default path) — categorized breakdowns (by error type, param set, grade-mismatch direction, grader-exception failing side) with up to `--top_n` example records per category.
- **Full CSV export** — same path with a `.csv` extension, containing every record across all four categories (uncapped), with columns: `category, type, submission_id, param_set, original_grade, answer, response, message`.

### Examples

```bash
# Analyze the most recent run
python3 analyze_run.py

# Analyze the most recent run for a specific function
python3 analyze_run.py --eval_function_name my_function

# Analyze a specific run, with more examples per category
python3 analyze_run.py --run_id abc123 --top_n 20 --output analysis_abc123.json
```

## Usage: `manage_exclusions.py`

Manages the Firestore-backed list of submission IDs to skip in future `test_evaluation_function.py` runs (e.g. known-bad or out-of-scope historical submissions).

Top-level required flag: `--eval_function_name`.

| Subcommand | Flags | Description |
| --- | --- | --- |
| `add` | `--ids UUID [UUID ...]`, `--from_csv PATH` | Add submission IDs to the exclusion list. At least one of `--ids`/`--from_csv` is required; `--from_csv` reads a `submission_id` column. |
| `remove` | `--ids UUID [UUID ...]` (required) | Remove submission IDs from the exclusion list |
| `list` | — | Print the currently excluded IDs and last-updated timestamp |

### Examples

```bash
# Add specific IDs
python3 manage_exclusions.py --eval_function_name my_function add --ids abc-123 def-456

# Add IDs found in an analyze_run.py CSV export (must have a submission_id column)
python3 manage_exclusions.py --eval_function_name my_function add --from_csv analysis_abc123.csv

# Remove an ID
python3 manage_exclusions.py --eval_function_name my_function remove --ids abc-123

# List current exclusions
python3 manage_exclusions.py --eval_function_name my_function list
```

## Exclusion workflow

The three scripts are designed to close a loop:

1. Run a test with `test_evaluation_function.py`.
2. Investigate the failures with `analyze_run.py`, which writes a CSV of every failing record.
3. Confirm which failures are known-bad or out-of-scope submissions (rather than real regressions).
4. Add their IDs to the exclusion list with `manage_exclusions.py add --from_csv <that CSV>` (or `--ids`).
5. Future `test_evaluation_function.py` runs automatically skip those IDs.

The exclusion list is keyed by function name. When using `--source_eval_function_name`, exclusions are read from (and should be maintained under) the **source** function's name, not the target's.

## Data model & output artifacts

**Firestore:**

- `test-results/{doc_id}` — one document per run, with fields: `timestamp`, `created_at`, `eval_function_name`, `source_eval_function_name`, `sql_limit`, `request_delay`, `max_concurrency`, `grade_params_json`, `seed`, `pass_count`, `total_count`, `number_of_errors`, `number_of_feedback_warnings`, `number_of_parsing_warnings`, `pass_rate`, `status`.
  - Subcollections `errors`, `network_errors`, `feedback_warnings`, `parsing_warnings` — one document per failing/warned record.
- `excluded-submissions/{eval_function_name}` — a document with an `ids` array field, updated via `manage_exclusions.py`.

**Local files:**

- `report_data.json` — overwritten on every `test_evaluation_function.py` run. Fields: `status`, `pass_count`, `total_count`, `tested_count`, `number_of_errors`, `number_of_network_errors`, `number_of_feedback_warnings`, `number_of_parsing_warnings`, `seed`, `eval_function_name`, `source_eval_function_name`, `firestore_doc_id`, `firestore_link` (plus `errors`/`network_errors`/`feedback_warnings`/`parsing_warnings` if the Firestore save failed). This is also what the GitHub Actions job summary step parses.
- `analysis_<run_id>.json` / `.csv` — from `analyze_run.py` (see above).

## GitHub Actions usage

`.github/workflows/test_evaluation_function.yml` is a reusable workflow (`workflow_call`), also triggerable directly (`workflow_dispatch`). It installs dependencies, optionally adds exclusions, runs `test_evaluation_function.py`, checks the pass rate against a threshold, and writes a Markdown job summary with a link to the Firestore results.

**Inputs:**

| Input | Required | Default | Notes |
| --- | --- | --- | --- |
| `eval_function` | Yes | — | Evaluation function name |
| `endpoint` | `workflow_dispatch` only, required there | — | Not present as an input on `workflow_call`; supplied via the `TEST_API_ENDPOINT` secret instead (see below) |
| `sql_limit` | No | `1000` (`workflow_call`) | Note: this differs from the CLI script's own default of `100` |
| `request_delay` | No | `'0'` | |
| `max_concurrency` | No | `'5'` | |
| `seed` | No | `''` (auto-generate) | |
| `pass_threshold` | No | `'100'` | Minimum pass rate %; `100` means any failure fails the run |
| `ids_to_exclude` | No | `''` | Comma-separated UUIDs, applied via `manage_exclusions.py add` before testing |

**Secrets:** `TEST_API_ENDPOINT` (optional — falls back to `inputs.endpoint`), `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `GCP_SERVICE_ACCOUNT_KEY`, `GCP_PROJECT_ID`.

**Calling it from another workflow:**

```yaml
jobs:
  test:
    uses: lambda-feedback/Database-Testing/.github/workflows/test_evaluation_function.yml@main
    with:
      eval_function: my_function
      sql_limit: 500
      pass_threshold: '95'
    secrets:
      TEST_API_ENDPOINT: ${{ secrets.TEST_API_ENDPOINT }}
      DB_USER: ${{ secrets.DB_USER }}
      DB_PASSWORD: ${{ secrets.DB_PASSWORD }}
      DB_HOST: ${{ secrets.DB_HOST }}
      DB_PORT: ${{ secrets.DB_PORT }}
      DB_NAME: ${{ secrets.DB_NAME }}
      GCP_SERVICE_ACCOUNT_KEY: ${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}
      GCP_PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
```

> **Note:** the workflow does not currently expose `--eval_function_param`, `--max_error_threshold`, or `--exclude_grade_param`. Runs that need those flags must be run directly via the CLI, not through this Action.

## Repo layout

| File | Purpose |
| --- | --- |
| `test_evaluation_function.py` | Main CLI entry point / orchestrator |
| `analyze_run.py` | Failure-analysis CLI |
| `manage_exclusions.py` | Exclusion-list management CLI |
| `db.py` | Postgres access (SQLAlchemy) |
| `evaluator.py` | Async HTTP test-runner core (`test_endpoint`, retries, response validation) |
| `firestore_client.py` | Firestore client setup and read/write helpers |
| `config.py` | Central config/env loading and logger setup |

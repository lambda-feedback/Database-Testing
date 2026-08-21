from typing import Any, Dict

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from config import logger, DEFAULT_MUED_SCHEMA_PATH

REQUIRED_ARTEFACT_TYPE = "OTHER"
_SCHEMA_URI = "urn:mued-openapi"


class SchemaLoadError(Exception):
    """Raised when the muEd OpenAPI schema cannot be found, parsed, or is missing what we need."""


class MuEdSchema:
    """Loads the bundled muEd OpenAPI spec and validates payloads against its live EvaluateRequest schema."""

    def __init__(self, schema_path: str = DEFAULT_MUED_SCHEMA_PATH):
        self.schema_path = schema_path
        doc = self._load(schema_path)
        self._schemas: Dict[str, Any] = doc.get('components', {}).get('schemas', {})

        doc_with_id = {**doc, "$id": _SCHEMA_URI}
        resource = Resource.from_contents(doc_with_id, default_specification=DRAFT202012)
        registry = resource @ Registry()
        request_schema = {"$ref": f"{_SCHEMA_URI}#/components/schemas/EvaluateRequest"}
        self._validator = Draft202012Validator(request_schema, registry=registry)

        self._check_artefact_type_supported()
        logger.info(f"Loaded muEd OpenAPI schema from '{schema_path}'.")

    @staticmethod
    def _load(schema_path: str) -> Dict[str, Any]:
        try:
            with open(schema_path, 'r') as f:
                doc = yaml.safe_load(f)
        except FileNotFoundError:
            raise SchemaLoadError(
                f"muEd OpenAPI schema not found at '{schema_path}'. --api_mode mued requires a local "
                f"checkout of mued-api/spec, bundled via `npm run bundle` into dist/openapi.yml. "
                f"Set --mued_schema_path or the MUED_SCHEMA_PATH env var to point at it."
            )
        except yaml.YAMLError as e:
            raise SchemaLoadError(f"Could not parse muEd OpenAPI schema YAML at '{schema_path}': {e}")

        if not isinstance(doc, dict) or 'components' not in doc:
            raise SchemaLoadError(
                f"'{schema_path}' does not look like a bundled OpenAPI document (no top-level 'components' key)."
            )
        return doc

    def _check_artefact_type_supported(self) -> None:
        enum_values = self._schemas.get('ArtefactType', {}).get('enum', [])
        if REQUIRED_ARTEFACT_TYPE not in enum_values:
            raise SchemaLoadError(
                f"ArtefactType '{REQUIRED_ARTEFACT_TYPE}' is no longer valid in the muEd schema at "
                f"'{self.schema_path}'. Current enum: {enum_values}. Update the hardcoded artefact "
                f"type in evaluator.py's muEd payload builder to match."
            )

    def validate_evaluate_request(self, payload: Dict[str, Any]) -> None:
        """Raises jsonschema.exceptions.ValidationError, naming the exact field path, if payload
        does not conform to the live EvaluateRequest schema."""
        errors = sorted(self._validator.iter_errors(payload), key=lambda e: list(e.path))
        if errors:
            first = errors[0]
            path = "/".join(str(p) for p in first.path) or "<root>"
            raise ValidationError(f"muEd payload failed schema validation at '{path}': {first.message}")


_schema_cache: Dict[str, MuEdSchema] = {}


def get_schema(schema_path: str = DEFAULT_MUED_SCHEMA_PATH) -> MuEdSchema:
    """Returns a process-cached MuEdSchema for the given path, loading it on first use."""
    if schema_path not in _schema_cache:
        _schema_cache[schema_path] = MuEdSchema(schema_path)
    return _schema_cache[schema_path]

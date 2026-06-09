import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import click
import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from tqdm.auto import tqdm
except ImportError:
    class _NoOpTqdm:
        """Fallback progress bar used when tqdm is unavailable."""

        def __init__(
            self,
            total: int | None = None,
            desc: str | None = None,
            unit: str | None = None,
        ) -> None:
            """Initialize a no-op progress bar.

            Args:
                total: Expected total number of updates.
                desc: Progress label.
                unit: Progress unit.
            """

            self.total = total
            self.desc = desc
            self.unit = unit

        def __enter__(self) -> "_NoOpTqdm":
            """Enter the context manager."""

            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            exc_tb: Any,
        ) -> bool:
            """Exit the context manager.

            Args:
                exc_type: Exception class when present.
                exc: Exception instance when present.
                exc_tb: Exception traceback when present.

            Returns:
                False to propagate exceptions.
            """

            return False

        def update(self, amount: int = 1) -> None:
            """Advance the progress bar.

            Args:
                amount: Number of completed units.
            """

        def set_postfix_str(self, text: str, refresh: bool = True) -> None:
            """Set a postfix string.

            Args:
                text: Postfix text.
                refresh: Unused fallback flag.
            """

    def tqdm(
        total: int | None = None,
        desc: str | None = None,
        unit: str | None = None,
    ) -> _NoOpTqdm:
        """Return a no-op progress bar.

        Args:
            total: Expected total number of updates.
            desc: Progress label.
            unit: Progress unit.

        Returns:
            A no-op progress bar.
        """

        return _NoOpTqdm(total=total, desc=desc, unit=unit)

import prompts
from feature_schema import NUMERIC_FEATURE_COLUMNS
from tools.albert_client import (
    DEFAULT_ALBERT_BASE_URL,
    DEFAULT_ALBERT_REQUEST_INTERVAL_SECONDS,
    AlbertChatRequest,
    AlbertJsonSchemaDefinition,
    AlbertMessage,
    AlbertResponseFormat,
    chat_with_albert,
    ensure_albert_api_key,
)

JSON_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
SUPPORTED_LABELS_BY_FRENCH_NAME: dict[str, str] = {
    "anglais": "ANGLAIS",
    "arabe": "ARABE",
    "chinois": "CHINOIS",
    "portugais": "PORTUGAIS",
    "russe": "RUSSE",
}
TARGET_NATIVE_LANGUAGE_CODE_TO_LABEL: dict[str, str] = {
    "A": "ANGLAIS",
    "B": "ARABE",
    "C": "CHINOIS",
    "D": "PORTUGAIS",
    "E": "RUSSE",
}
PIPELINE_NAMES: tuple[str, ...] = (
    "direct_label_minimal",
    "direct_label_minimal_features",
)
PipelineName = Literal["direct_label_minimal", "direct_label_minimal_features"]


class BenchmarkConfig(BaseModel):
    """Runtime configuration for one benchmark execution.

    Attributes:
        input_csv: Main benchmark CSV.
        models: ALBERT models to evaluate.
        output_dir: Output artifact directory.
        base_url: ALBERT API base URL.
        pipeline: Selected direct-label pipeline.
        prompt_language: Fixed prompt language.
        text_column: Text column name.
        label_column: Gold label column name.
        id_column: Identifier column name.
        separator: Optional CSV separator.
        truncate_ragged_lines: Whether extra CSV fields may be truncated.
        limit: Optional benchmark row limit.
        few_shot_csv: Optional support CSV.
        few_shot_per_label: Number of support examples per label.
        features_enabled: Whether the pipeline requires numeric features.
        feature_columns: Ordered required feature columns.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        request_interval_seconds: Fixed delay between ALBERT calls.
        num_predict: Maximum generated tokens for the primary call.
        repair_enabled: Whether one repair pass is allowed.
        repair_num_predict: Maximum generated tokens for the repair call.
        timeout_seconds: Per-call timeout.
    """

    input_csv: Path
    models: list[str]
    output_dir: Path
    base_url: str = DEFAULT_ALBERT_BASE_URL
    pipeline: PipelineName = "direct_label_minimal"
    prompt_language: Literal["fr"] = "fr"
    text_column: str = "Texte"
    label_column: str = "Langue"
    id_column: str = "ID"
    separator: str | None = None
    truncate_ragged_lines: bool = False
    limit: int | None = None
    few_shot_csv: Path | None = None
    few_shot_per_label: int = 0
    features_enabled: bool = False
    feature_columns: list[str] = Field(default_factory=list)
    temperature: float = 0.0
    top_p: float = 0.9
    request_interval_seconds: float = DEFAULT_ALBERT_REQUEST_INTERVAL_SECONDS
    num_predict: int = 256
    repair_enabled: bool = True
    repair_num_predict: int = 160
    timeout_seconds: int = 180

    @field_validator("models")
    @classmethod
    def validate_models(cls, value: list[str]) -> list[str]:
        """Reject empty model lists.

        Args:
            value: User-provided model list.

        Returns:
            The cleaned model list.

        Raises:
            ValueError: If no non-empty model remains.
        """

        cleaned_models = [model.strip() for model in value if model.strip()]
        if not cleaned_models:
            raise ValueError("At least one model must be provided.")
        return cleaned_models

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """Validate the ALBERT base URL.

        Args:
            value: User-provided base URL.

        Returns:
            A normalized base URL.

        Raises:
            ValueError: If the URL is empty.
        """

        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("ALBERT base URL must not be empty.")
        return normalized_value.rstrip("/")

    @field_validator("separator")
    @classmethod
    def validate_separator(cls, value: str | None) -> str | None:
        """Validate the optional CSV separator.

        Args:
            value: User-provided separator.

        Returns:
            A normalized separator or None.

        Raises:
            ValueError: If the separator length is invalid.
        """

        if value is None:
            return None
        normalized_value = value.strip()
        if not normalized_value:
            return None
        if len(normalized_value) != 1:
            raise ValueError("CSV separator must be exactly one character.")
        return normalized_value

    @field_validator("few_shot_per_label")
    @classmethod
    def validate_few_shot_per_label(cls, value: int) -> int:
        """Reject negative few-shot counts.

        Args:
            value: Number of support examples per label.

        Returns:
            The non-negative few-shot count.

        Raises:
            ValueError: If the count is negative.
        """

        if value < 0:
            raise ValueError("few_shot_per_label must be greater than or equal to 0.")
        return value

    @field_validator("request_interval_seconds")
    @classmethod
    def validate_request_interval_seconds(cls, value: float) -> float:
        """Reject negative request intervals.

        Args:
            value: Requested interval in seconds.

        Returns:
            The validated interval.

        Raises:
            ValueError: If the interval is negative.
        """

        if value < 0.0:
            raise ValueError("request_interval_seconds must be greater than or equal to 0.")
        return value

    @model_validator(mode="after")
    def validate_cross_field_config(self) -> "BenchmarkConfig":
        """Validate cross-field benchmark constraints.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If few-shot settings are inconsistent.
        """

        self.features_enabled = self.pipeline == "direct_label_minimal_features"
        self.feature_columns = list(NUMERIC_FEATURE_COLUMNS) if self.features_enabled else []
        if self.few_shot_per_label == 0:
            return self
        if self.few_shot_csv is None:
            raise ValueError("--few-shot-csv is required when --few-shot-per-label > 0.")
        return self


class PredictionOutput(BaseModel):
    """Validated direct-label prediction output.

    Attributes:
        predicted_native_language: Canonical target label.
        confidence: Confidence value normalized to the [0, 1] range.
    """

    predicted_native_language: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("predicted_native_language")
    @classmethod
    def normalize_prediction(cls, value: str) -> str:
        """Normalize one predicted label.

        Args:
            value: Raw model output.

        Returns:
            The canonical target label.

        Raises:
            ValueError: If the prediction is unsupported.
        """

        normalized_value = value.strip().upper()
        normalized_value = TARGET_NATIVE_LANGUAGE_CODE_TO_LABEL.get(
            normalized_value,
            normalized_value,
        )
        if normalized_value not in prompts.TARGET_NATIVE_LANGUAGES:
            raise ValueError(f"Unsupported native language label: {value}")
        return normalized_value

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: Any) -> float:
        """Normalize confidence to the [0, 1] range.

        Args:
            value: Raw confidence value.

        Returns:
            A normalized confidence score.
        """

        if isinstance(value, str):
            stripped_value = value.strip().rstrip("%")
            normalized_text = stripped_value.replace(",", ".")
            numeric_value = float(normalized_text)
            if value.strip().endswith("%") or 1.0 < numeric_value <= 100.0:
                return numeric_value / 100.0
            return numeric_value
        numeric_value = float(value)
        if 1.0 < numeric_value <= 100.0:
            return numeric_value / 100.0
        return numeric_value


class InputRecord(TypedDict):
    """Normalized input row used by the benchmark."""

    row_index: int
    record_id: str
    gold_label: str | None
    mapped_gold_label: str | None
    text: str
    numeric_features: dict[str, float] | None


class FewShotSelection(TypedDict):
    """Resolved few-shot metadata for one run."""

    few_shot_enabled: bool
    few_shot_per_label: int
    few_shot_example_count: int
    examples: list[dict[str, Any]]


def normalize_gold_label(raw_label: str | None) -> str | None:
    """Map a raw gold label to the closed benchmark label set.

    Args:
        raw_label: Raw CSV label.

    Returns:
        The canonical benchmark label or None.
    """

    if raw_label is None:
        return None
    normalized_key = raw_label.strip().casefold()
    if normalized_key in SUPPORTED_LABELS_BY_FRENCH_NAME:
        return SUPPORTED_LABELS_BY_FRENCH_NAME[normalized_key]
    uppercase_label = raw_label.strip().upper()
    if uppercase_label in prompts.TARGET_NATIVE_LANGUAGES:
        return uppercase_label
    return None


def candidate_csv_separators(config: BenchmarkConfig) -> list[str]:
    """Return the ordered list of CSV separators to try.

    Args:
        config: Benchmark configuration.

    Returns:
        Candidate separators in lookup order.
    """

    if config.separator is not None:
        return [config.separator]
    return [",", ";", "\t", "|"]


def try_read_csv_with_separator(
    input_csv: Path,
    separator: str,
    truncate_ragged_lines: bool,
) -> pl.DataFrame:
    """Read a CSV with one specific separator.

    Args:
        input_csv: CSV path.
        separator: Field separator.
        truncate_ragged_lines: Whether extra fields may be truncated.

    Returns:
        The loaded Polars DataFrame.
    """

    return pl.read_csv(
        input_csv,
        separator=separator,
        infer_schema_length=0,
        ignore_errors=False,
        truncate_ragged_lines=truncate_ragged_lines,
    )


def load_csv_with_fallbacks(config: BenchmarkConfig, input_csv: Path) -> pl.DataFrame:
    """Load a CSV by trying a small list of separators.

    Args:
        config: Benchmark configuration.
        input_csv: CSV path to load.

    Returns:
        The loaded DataFrame.

    Raises:
        ValueError: If the expected text column cannot be found.
    """

    attempted_errors: list[str] = []
    for separator in candidate_csv_separators(config):
        try:
            dataframe = try_read_csv_with_separator(
                input_csv=input_csv,
                separator=separator,
                truncate_ragged_lines=config.truncate_ragged_lines,
            )
        except Exception as exc:
            attempted_errors.append(f"separator={separator!r}: {exc}")
            continue
        if config.text_column in dataframe.columns:
            return dataframe
        attempted_errors.append(
            f"separator={separator!r}: missing text column '{config.text_column}'"
        )

    separator_list = ", ".join(repr(value) for value in candidate_csv_separators(config))
    errors_text = "\n".join(attempted_errors)
    raise ValueError(
        f"Unable to read CSV '{input_csv}' with the expected text column "
        f"'{config.text_column}'. Attempted separators: {separator_list}.\n{errors_text}"
    )


def parse_numeric_feature_cell(
    raw_value: Any,
    *,
    input_csv: Path,
    row_index: int,
    record_id: str,
    column_name: str,
) -> float:
    """Parse and validate one numeric feature cell.

    Args:
        raw_value: Raw CSV cell value.
        input_csv: Source CSV path.
        row_index: One-based row index.
        record_id: Stable record identifier.
        column_name: Expected feature column name.

    Returns:
        A finite float value.

    Raises:
        ValueError: If the feature is missing or invalid.
    """

    normalized_value = str(raw_value).strip() if raw_value is not None else ""
    if not normalized_value:
        raise ValueError(
            f"Missing feature '{column_name}' for row {row_index} record '{record_id}' "
            f"in '{input_csv}'."
        )
    normalized_value = normalized_value.replace(",", ".")
    try:
        numeric_value = float(normalized_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric feature '{column_name}' for row {row_index} record "
            f"'{record_id}' in '{input_csv}': {raw_value!r}"
        ) from exc
    if not (numeric_value == numeric_value and abs(numeric_value) != float("inf")):
        raise ValueError(
            f"Non-finite numeric feature '{column_name}' for row {row_index} record "
            f"'{record_id}' in '{input_csv}': {raw_value!r}"
        )
    return numeric_value


def build_numeric_feature_payload(
    row: dict[str, Any],
    *,
    input_csv: Path,
    row_index: int,
    record_id: str,
    feature_columns: tuple[str, ...],
) -> dict[str, float]:
    """Build the numeric feature payload for one row.

    Args:
        row: Row dictionary from Polars.
        input_csv: Source CSV path.
        row_index: One-based row index.
        record_id: Stable record identifier.
        feature_columns: Ordered expected feature columns.

    Returns:
        The ordered numeric feature mapping.
    """

    return {
        column_name: parse_numeric_feature_cell(
            row.get(column_name),
            input_csv=input_csv,
            row_index=row_index,
            record_id=record_id,
            column_name=column_name,
        )
        for column_name in feature_columns
    }


def load_input_records_from_csv(
    config: BenchmarkConfig,
    input_csv: Path,
    *,
    limit: int | None,
    required_feature_columns: tuple[str, ...],
) -> list[InputRecord]:
    """Load and normalize benchmark rows from a CSV.

    Args:
        config: Benchmark configuration.
        input_csv: CSV path to normalize.
        limit: Optional row limit.
        required_feature_columns: Ordered numeric feature columns required by the pipeline.

    Returns:
        The ordered normalized records.
    """

    dataframe = load_csv_with_fallbacks(config, input_csv)

    selected_columns = [config.text_column]
    if config.id_column in dataframe.columns:
        selected_columns.append(config.id_column)
    if config.label_column in dataframe.columns:
        selected_columns.append(config.label_column)
    if required_feature_columns:
        missing_feature_columns = [
            column_name
            for column_name in required_feature_columns
            if column_name not in dataframe.columns
        ]
        if missing_feature_columns:
            raise ValueError(
                f"Missing required feature columns in '{input_csv}': "
                + ", ".join(missing_feature_columns)
            )
        selected_columns.extend(required_feature_columns)

    normalized_dataframe = dataframe.select(selected_columns)
    if limit is not None:
        normalized_dataframe = normalized_dataframe.head(limit)

    records: list[InputRecord] = []
    for row_index, row in enumerate(normalized_dataframe.iter_rows(named=True), start=1):
        text_value = str(row.get(config.text_column, "") or "").strip()
        if not text_value:
            continue
        record_id_value = row.get(config.id_column)
        gold_label_value = row.get(config.label_column)
        record_id = str(record_id_value) if record_id_value is not None else str(row_index)
        records.append(
            InputRecord(
                row_index=row_index,
                record_id=record_id,
                gold_label=str(gold_label_value).strip() if gold_label_value is not None else None,
                mapped_gold_label=normalize_gold_label(
                    str(gold_label_value) if gold_label_value is not None else None
                ),
                text=text_value,
                numeric_features=(
                    build_numeric_feature_payload(
                        row,
                        input_csv=input_csv,
                        row_index=row_index,
                        record_id=record_id,
                        feature_columns=required_feature_columns,
                    )
                    if required_feature_columns
                    else None
                ),
            )
        )
    return records


def load_input_records(config: BenchmarkConfig) -> list[InputRecord]:
    """Load benchmark rows from the primary input CSV.

    Args:
        config: Benchmark configuration.

    Returns:
        The ordered benchmark records.
    """

    required_feature_columns = tuple(config.feature_columns)
    return load_input_records_from_csv(
        config,
        input_csv=config.input_csv,
        limit=config.limit,
        required_feature_columns=required_feature_columns,
    )


def select_few_shot_examples(config: BenchmarkConfig) -> FewShotSelection:
    """Load deterministic few-shot examples.

    Args:
        config: Benchmark configuration.

    Returns:
        Few-shot metadata and ordered examples.
    """

    if config.few_shot_per_label == 0 or config.few_shot_csv is None:
        return FewShotSelection(
            few_shot_enabled=False,
            few_shot_per_label=0,
            few_shot_example_count=0,
            examples=[],
        )

    support_records = load_input_records_from_csv(
        config,
        input_csv=config.few_shot_csv,
        limit=None,
        required_feature_columns=tuple(config.feature_columns),
    )
    records_by_label: dict[str, list[InputRecord]] = {
        label: [] for label in prompts.TARGET_NATIVE_LANGUAGES
    }
    for record in support_records:
        if record["mapped_gold_label"] is None:
            continue
        records_by_label[record["mapped_gold_label"]].append(record)

    selected_examples: list[dict[str, Any]] = []
    for label in prompts.TARGET_NATIVE_LANGUAGES:
        for record in records_by_label[label][: config.few_shot_per_label]:
            selected_examples.append(
                {
                    "record_id": record["record_id"],
                    "gold_label": record["gold_label"] or label,
                    "mapped_gold_label": label,
                    "language_code": prompts.TARGET_NATIVE_LANGUAGE_LABEL_TO_CODE[label],
                    "text": record["text"],
                    "numeric_features": record["numeric_features"],
                }
            )

    return FewShotSelection(
        few_shot_enabled=bool(selected_examples),
        few_shot_per_label=config.few_shot_per_label,
        few_shot_example_count=len(selected_examples),
        examples=selected_examples,
    )


def extract_json_payload(raw_output: str) -> str:
    """Extract the most likely JSON payload from a model response.

    Args:
        raw_output: Raw assistant output.

    Returns:
        The extracted JSON payload.
    """

    stripped_output = raw_output.strip()
    code_block_match = JSON_CODE_BLOCK_PATTERN.search(stripped_output)
    if code_block_match:
        stripped_output = code_block_match.group(1).strip()

    candidate_indexes = [
        index
        for index in (stripped_output.find("["), stripped_output.find("{"))
        if index >= 0
    ]
    if not candidate_indexes:
        return stripped_output

    start_index = min(candidate_indexes)
    end_index = max(stripped_output.rfind("]"), stripped_output.rfind("}"))
    if end_index <= start_index:
        return stripped_output[start_index:]
    return stripped_output[start_index : end_index + 1]


def sanitize_json_schema(schema: Any) -> Any:
    """Remove JSON Schema metadata not needed by ALBERT.

    Args:
        schema: Raw JSON Schema subtree.

    Returns:
        A simplified JSON Schema subtree.
    """

    if isinstance(schema, dict):
        sanitized_schema: dict[str, Any] = {}
        for key, value in schema.items():
            if key in {"title", "description", "default"}:
                continue
            sanitized_schema[key] = sanitize_json_schema(value)
        return sanitized_schema
    if isinstance(schema, list):
        return [sanitize_json_schema(item) for item in schema]
    return schema


def build_response_format(schema_name: str) -> AlbertResponseFormat:
    """Build the structured-output settings for ALBERT.

    Args:
        schema_name: Stable schema name.

    Returns:
        The structured response-format payload.
    """

    return AlbertResponseFormat(
        type="json_schema",
        json_schema=AlbertJsonSchemaDefinition(
            name=schema_name,
            json_schema_payload=sanitize_json_schema(PredictionOutput.model_json_schema()),
        ),
    )


def build_request(
    model: str,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    num_predict: int,
    schema_name: str,
) -> AlbertChatRequest:
    """Build one ALBERT request payload.

    Args:
        model: ALBERT model identifier.
        prompt: Fully formatted user prompt.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        num_predict: Maximum generated tokens.
        schema_name: Structured-output schema name.

    Returns:
        The ALBERT request payload.
    """

    return AlbertChatRequest(
        model=model,
        messages=[
            AlbertMessage(role="system", content=prompts.JSON_ONLY_SYSTEM_PROMPT),
            AlbertMessage(role="user", content=prompt),
        ],
        response_format=build_response_format(schema_name),
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=num_predict,
        stream=False,
    )


def parse_prediction_output(raw_output: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse and validate a model response.

    Args:
        raw_output: Raw assistant output.

    Returns:
        A tuple of validated payload and error message.
    """

    try:
        payload_text = extract_json_payload(raw_output)
        decoded_payload = json.loads(payload_text)
        validated_payload = PredictionOutput.model_validate(decoded_payload)
        return validated_payload.model_dump(), None
    except (json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)


def run_agent(
    model: str,
    agent_name: str,
    record: InputRecord,
    prompt: str,
    config: BenchmarkConfig,
    *,
    num_predict_override: int | None = None,
) -> dict[str, Any]:
    """Run one logical benchmark agent.

    Args:
        model: ALBERT model identifier.
        agent_name: Logical agent name.
        record: Input record metadata.
        prompt: Fully formatted prompt.
        config: Benchmark configuration.
        num_predict_override: Optional token limit override.

    Returns:
        One JSON-serializable agent log row.
    """

    timestamp_utc = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    try:
        response = chat_with_albert(
            build_request(
                model=model,
                prompt=prompt,
                temperature=config.temperature,
                top_p=config.top_p,
                num_predict=(
                    num_predict_override if num_predict_override is not None else config.num_predict
                ),
                schema_name=agent_name,
            ),
            base_url=config.base_url,
            min_request_interval_seconds=config.request_interval_seconds,
            timeout_seconds=config.timeout_seconds,
        )
        raw_output = response.choices[0].message.content or ""
        parsed_output, error_message = parse_prediction_output(raw_output)
        return {
            "timestamp_utc": timestamp_utc,
            "model": model,
            "agent_name": agent_name,
            "row_index": record["row_index"],
            "record_id": record["record_id"],
            "json_valid": error_message is None,
            "latency_seconds": response.wall_time_seconds,
            "prompt_eval_count": response.usage.prompt_tokens if response.usage is not None else None,
            "eval_count": response.usage.completion_tokens if response.usage is not None else None,
            "raw_output": raw_output,
            "parsed_output": parsed_output,
            "error": error_message,
        }
    except RuntimeError as exc:
        return {
            "timestamp_utc": timestamp_utc,
            "model": model,
            "agent_name": agent_name,
            "row_index": record["row_index"],
            "record_id": record["record_id"],
            "json_valid": False,
            "latency_seconds": 0.0,
            "prompt_eval_count": None,
            "eval_count": None,
            "raw_output": "",
            "parsed_output": None,
            "error": str(exc),
        }


def build_primary_prompt(
    pipeline: PipelineName,
    record: InputRecord,
    few_shot_examples: list[dict[str, Any]],
) -> str:
    """Build the primary prompt for one record.

    Args:
        pipeline: Selected pipeline.
        record: Input record metadata.
        few_shot_examples: Ordered few-shot examples.

    Returns:
        The fully formatted primary prompt.
    """

    if pipeline == "direct_label_minimal":
        return prompts.build_direct_label_minimal_prompt(
            record["text"],
            few_shot_examples=few_shot_examples,
        )
    if record["numeric_features"] is None:
        raise ValueError(
            "direct_label_minimal_features requires numeric features on every input record."
        )
    return prompts.build_direct_label_minimal_features_prompt(
        record["text"],
        numeric_features=record["numeric_features"],
        few_shot_examples=few_shot_examples,
    )


def build_repair_prompt(
    record: InputRecord,
    primary_log: dict[str, Any],
) -> str:
    """Build the repair prompt for one invalid primary output.

    Args:
        record: Input record metadata.
        primary_log: Primary agent log.

    Returns:
        The fully formatted repair prompt.
    """

    return prompts.build_direct_label_repair_prompt(
        text=record["text"],
        previous_output=str(primary_log.get("raw_output", "")),
        validation_error=str(primary_log.get("error") or ""),
        numeric_features=record["numeric_features"],
    )


def maybe_run_repair(
    model: str,
    record: InputRecord,
    primary_log: dict[str, Any],
    config: BenchmarkConfig,
) -> dict[str, Any] | None:
    """Run one repair pass when the primary output is invalid.

    Args:
        model: ALBERT model identifier.
        record: Input record metadata.
        primary_log: Primary agent log.
        config: Benchmark configuration.

    Returns:
        The repair log or None.
    """

    if not config.repair_enabled or primary_log["json_valid"]:
        return None
    repair_agent_name = (
        "direct_label_minimal_features_repair"
        if config.pipeline == "direct_label_minimal_features"
        else "direct_label_minimal_repair"
    )
    return run_agent(
        model=model,
        agent_name=repair_agent_name,
        record=record,
        prompt=build_repair_prompt(record, primary_log),
        config=config,
        num_predict_override=config.repair_num_predict,
    )


def extract_prediction_value(agent_log: dict[str, Any] | None) -> str | None:
    """Extract the normalized prediction value from one agent log.

    Args:
        agent_log: Primary or repair agent log.

    Returns:
        The predicted label or None.
    """

    if agent_log is None or not agent_log["json_valid"]:
        return None
    parsed_output = agent_log.get("parsed_output")
    if not isinstance(parsed_output, dict):
        return None
    prediction_value = parsed_output.get("predicted_native_language")
    return str(prediction_value) if prediction_value is not None else None


def build_summary_row(
    config: BenchmarkConfig,
    few_shot_selection: FewShotSelection,
    model: str,
    record: InputRecord,
    primary_log: dict[str, Any],
    repair_log: dict[str, Any] | None,
    total_latency_seconds: float,
) -> dict[str, Any]:
    """Build one flat benchmark summary row.

    Args:
        config: Benchmark configuration.
        few_shot_selection: Few-shot metadata for the run.
        model: ALBERT model identifier.
        record: Input record metadata.
        primary_log: Primary direct-label log.
        repair_log: Optional repair log.
        total_latency_seconds: Total wall time for the row.

    Returns:
        One JSON-serializable summary row.
    """

    primary_prediction = extract_prediction_value(primary_log)
    repair_prediction = extract_prediction_value(repair_log) if repair_log is not None else None
    repair_attempted = repair_log is not None
    repair_used = repair_log is not None and bool(repair_log["json_valid"])
    final_prediction = repair_prediction if repair_used else primary_prediction
    final_prediction_json_valid = bool(primary_log["json_valid"]) or repair_used

    prediction_matches_gold: bool | None = None
    if record["mapped_gold_label"] is not None and final_prediction is not None:
        prediction_matches_gold = record["mapped_gold_label"] == final_prediction

    return {
        "pipeline": config.pipeline,
        "prompt_language": config.prompt_language,
        "few_shot_enabled": few_shot_selection["few_shot_enabled"],
        "few_shot_per_label": few_shot_selection["few_shot_per_label"],
        "few_shot_example_count": few_shot_selection["few_shot_example_count"],
        "model": model,
        "row_index": record["row_index"],
        "record_id": record["record_id"],
        "gold_label": record["gold_label"],
        "mapped_gold_label": record["mapped_gold_label"],
        "supported_gold_label": record["mapped_gold_label"] is not None,
        "primary_prediction": primary_prediction,
        "repair_prediction": repair_prediction,
        "prediction": final_prediction,
        "prediction_matches_gold": prediction_matches_gold,
        "primary_json_valid": primary_log["json_valid"],
        "syntax_json_valid": None,
        "lexical_json_valid": None,
        "idiomatic_json_valid": None,
        "coordinator_primary_json_valid": None,
        "repair_attempted": repair_attempted,
        "repair_json_valid": repair_log["json_valid"] if repair_log is not None else None,
        "final_prediction_json_valid": final_prediction_json_valid,
        "repair_used": repair_used,
        "syntax_count": None,
        "lexical_count": None,
        "idiomatic_count": None,
        "total_latency_seconds": total_latency_seconds,
    }


def build_model_aggregate(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate summary metrics at the model level.

    Args:
        summary_rows: Per-text summary rows.

    Returns:
        Aggregated metrics per model.
    """

    def optional_rate(values: list[bool | None]) -> float | None:
        """Compute a rate on applicable boolean values only.

        Args:
            values: Boolean values or None.

        Returns:
            A rounded rate or None.
        """

        applicable_values = [value for value in values if value is not None]
        if not applicable_values:
            return None
        return round(
            sum(1 for value in applicable_values if value) / len(applicable_values),
            6,
        )

    grouped_rows: dict[tuple[str, str, bool, int, int, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (
            str(row["pipeline"]),
            str(row["prompt_language"]),
            bool(row["few_shot_enabled"]),
            int(row["few_shot_per_label"]),
            int(row["few_shot_example_count"]),
            str(row["model"]),
        )
        grouped_rows.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (
        pipeline,
        prompt_language,
        few_shot_enabled,
        few_shot_per_label,
        few_shot_example_count,
        model_name,
    ), model_rows in grouped_rows.items():
        supported_gold_rows = [row for row in model_rows if row["supported_gold_label"]]
        comparable_rows = [
            row for row in supported_gold_rows if row["prediction_matches_gold"] is not None
        ]
        attempted_repairs = [row for row in model_rows if row["repair_attempted"]]
        aggregates.append(
            {
                "pipeline": pipeline,
                "prompt_language": prompt_language,
                "few_shot_enabled": few_shot_enabled,
                "few_shot_per_label": few_shot_per_label,
                "few_shot_example_count": few_shot_example_count,
                "model": model_name,
                "text_count": len(model_rows),
                "supported_gold_count": len(supported_gold_rows),
                "mean_total_latency_seconds": round(
                    sum(float(row["total_latency_seconds"]) for row in model_rows) / len(model_rows),
                    6,
                ),
                "primary_json_valid_rate": round(
                    sum(1 for row in model_rows if row["primary_json_valid"]) / len(model_rows),
                    6,
                ),
                "syntax_json_valid_rate": optional_rate(
                    [row["syntax_json_valid"] for row in model_rows]
                ),
                "lexical_json_valid_rate": optional_rate(
                    [row["lexical_json_valid"] for row in model_rows]
                ),
                "idiomatic_json_valid_rate": optional_rate(
                    [row["idiomatic_json_valid"] for row in model_rows]
                ),
                "coordinator_primary_json_valid_rate": optional_rate(
                    [row["coordinator_primary_json_valid"] for row in model_rows]
                ),
                "repair_attempt_rate": round(
                    sum(1 for row in model_rows if row["repair_attempted"]) / len(model_rows),
                    6,
                ),
                "repair_success_rate": (
                    round(
                        sum(1 for row in attempted_repairs if row["repair_json_valid"] is True)
                        / len(attempted_repairs),
                        6,
                    )
                    if attempted_repairs
                    else None
                ),
                "final_prediction_json_valid_rate": round(
                    sum(1 for row in model_rows if row["final_prediction_json_valid"])
                    / len(model_rows),
                    6,
                ),
                "accuracy_on_supported_gold": (
                    round(
                        sum(1 for row in comparable_rows if row["prediction_matches_gold"])
                        / len(comparable_rows),
                        6,
                    )
                    if comparable_rows
                    else None
                ),
            }
        )
    return aggregates


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSON-serializable rows to JSONL.

    Args:
        path: Output JSONL path.
        rows: Ordered JSON-serializable rows.
    """

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_output_dir(config: BenchmarkConfig) -> Path:
    """Build the output directory for one benchmark run.

    Args:
        config: Benchmark configuration.

    Returns:
        The timestamped artifact directory.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"smoke_benchmark_albert_{config.pipeline}_fr"
    if config.few_shot_per_label > 0:
        run_name = f"{run_name}_{config.few_shot_per_label}shotperlabel"
    return Path("artifacts/logs") / f"{run_name}_{timestamp}"


def run_direct_label_record(
    model: str,
    record: InputRecord,
    config: BenchmarkConfig,
    few_shot_selection: FewShotSelection,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one direct-label benchmark row.

    Args:
        model: ALBERT model identifier.
        record: Input record metadata.
        config: Benchmark configuration.
        few_shot_selection: Resolved few-shot metadata.

    Returns:
        Per-agent logs and one summary row.
    """

    primary_agent_name = config.pipeline
    primary_prompt = build_primary_prompt(
        config.pipeline,
        record,
        few_shot_selection["examples"],
    )
    primary_log = run_agent(
        model=model,
        agent_name=primary_agent_name,
        record=record,
        prompt=primary_prompt,
        config=config,
    )
    repair_log = maybe_run_repair(
        model=model,
        record=record,
        primary_log=primary_log,
        config=config,
    )

    agent_logs = [primary_log]
    if repair_log is not None:
        agent_logs.append(repair_log)

    total_latency_seconds = round(
        float(primary_log["latency_seconds"])
        + (float(repair_log["latency_seconds"]) if repair_log is not None else 0.0),
        6,
    )
    summary_row = build_summary_row(
        config=config,
        few_shot_selection=few_shot_selection,
        model=model,
        record=record,
        primary_log=primary_log,
        repair_log=repair_log,
        total_latency_seconds=total_latency_seconds,
    )
    return agent_logs, summary_row


def run_benchmark(config: BenchmarkConfig) -> dict[str, Path]:
    """Execute the configured benchmark.

    Args:
        config: Benchmark configuration.

    Returns:
        A mapping of output artifact names to filesystem paths.
    """

    ensure_albert_api_key()
    input_records = load_input_records(config)
    few_shot_selection = select_few_shot_examples(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.output_dir / "benchmark_config.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    agent_logs: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    total_text_count = len(config.models) * len(input_records)

    with tqdm(total=total_text_count, desc="Benchmark", unit="text") as progress_bar:
        for model in config.models:
            progress_bar.set_postfix_str(f"model={model}", refresh=True)
            for record in input_records:
                row_agent_logs, summary_row = run_direct_label_record(
                    model=model,
                    record=record,
                    config=config,
                    few_shot_selection=few_shot_selection,
                )
                agent_logs.extend(row_agent_logs)
                summary_rows.append(summary_row)
                progress_bar.update(1)

    agent_logs_path = config.output_dir / "agent_runs.jsonl"
    summary_path = config.output_dir / "benchmark_summary.csv"
    aggregate_path = config.output_dir / "model_aggregate.csv"

    write_jsonl(agent_logs_path, agent_logs)
    pl.DataFrame(summary_rows).write_csv(summary_path)
    pl.DataFrame(build_model_aggregate(summary_rows)).write_csv(aggregate_path)

    return {
        "config": config_path,
        "agent_runs": agent_logs_path,
        "summary": summary_path,
        "aggregate": aggregate_path,
    }


@click.command()
@click.option(
    "--input-csv",
    "input_csv",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("smoke.csv"),
    show_default=True,
    help="CSV file used for the ALBERT benchmark.",
)
@click.option(
    "--base-url",
    default=DEFAULT_ALBERT_BASE_URL,
    show_default=True,
    help="ALBERT API base URL.",
)
@click.option(
    "--model",
    "models",
    multiple=True,
    required=True,
    help=(
        "ALBERT model ID or alias to evaluate. Repeat the option for multiple models. "
        "Known examples: openweight-medium, openweight-small, openweight-large, albert-large."
    ),
)
@click.option(
    "--pipeline",
    type=click.Choice(list(PIPELINE_NAMES), case_sensitive=True),
    default="direct_label_minimal",
    show_default=True,
    help="Benchmark pipeline to execute.",
)
@click.option(
    "--text-column",
    default="Texte",
    show_default=True,
    help="Name of the text column.",
)
@click.option(
    "--label-column",
    default="Langue",
    show_default=True,
    help="Name of the gold-label column.",
)
@click.option(
    "--id-column",
    default="ID",
    show_default=True,
    help="Name of the identifier column.",
)
@click.option(
    "--separator",
    default=None,
    help="Optional CSV separator, for example ';' or ','.",
)
@click.option(
    "--truncate-ragged-lines/--no-truncate-ragged-lines",
    default=False,
    show_default=True,
    help="Allow Polars to truncate rows with extra fields. Use only as a last resort.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optional maximum number of rows to benchmark.",
)
@click.option(
    "--few-shot-csv",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Optional support CSV used for deterministic few-shot prompting.",
)
@click.option(
    "--few-shot-per-label",
    type=int,
    default=0,
    show_default=True,
    help="Number of support examples injected per target language.",
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    show_default=True,
    help="Sampling temperature.",
)
@click.option(
    "--top-p",
    type=float,
    default=0.9,
    show_default=True,
    help="Nucleus sampling parameter.",
)
@click.option(
    "--request-interval-seconds",
    type=float,
    default=DEFAULT_ALBERT_REQUEST_INTERVAL_SECONDS,
    show_default=True,
    help="Fixed delay enforced between ALBERT API calls.",
)
@click.option(
    "--num-predict",
    type=int,
    default=256,
    show_default=True,
    help="Maximum generated tokens per primary call.",
)
@click.option(
    "--repair/--no-repair",
    default=True,
    show_default=True,
    help="Whether to run one repair pass when the primary output is invalid.",
)
@click.option(
    "--repair-num-predict",
    type=int,
    default=160,
    show_default=True,
    help="Maximum generated tokens for the repair call.",
)
@click.option(
    "--timeout-seconds",
    type=int,
    default=180,
    show_default=True,
    help="Per-call timeout in seconds.",
)
def main(
    input_csv: Path,
    base_url: str,
    models: tuple[str, ...],
    pipeline: PipelineName,
    text_column: str,
    label_column: str,
    id_column: str,
    separator: str | None,
    truncate_ragged_lines: bool,
    limit: int | None,
    few_shot_csv: Path | None,
    few_shot_per_label: int,
    temperature: float,
    top_p: float,
    request_interval_seconds: float,
    num_predict: int,
    repair: bool,
    repair_num_predict: int,
    timeout_seconds: int,
) -> None:
    """Run the minimal French ALBERT benchmark."""

    if few_shot_per_label < 0:
        raise click.BadParameter(
            "--few-shot-per-label must be greater than or equal to 0.",
            param_hint="--few-shot-per-label",
        )
    if few_shot_per_label > 0 and few_shot_csv is None:
        raise click.UsageError(
            "--few-shot-csv is required when --few-shot-per-label is greater than 0."
        )

    config = BenchmarkConfig(
        input_csv=input_csv,
        models=list(models),
        output_dir=Path("."),
        base_url=base_url,
        pipeline=pipeline,
        text_column=text_column,
        label_column=label_column,
        id_column=id_column,
        separator=separator,
        truncate_ragged_lines=truncate_ragged_lines,
        limit=limit,
        few_shot_csv=few_shot_csv,
        few_shot_per_label=few_shot_per_label,
        temperature=temperature,
        top_p=top_p,
        request_interval_seconds=request_interval_seconds,
        num_predict=num_predict,
        repair_enabled=repair,
        repair_num_predict=repair_num_predict,
        timeout_seconds=timeout_seconds,
    )
    config.output_dir = build_output_dir(config)

    try:
        artifact_paths = run_benchmark(config)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps({name: str(path) for name, path in artifact_paths.items()}, indent=2))


if __name__ == "__main__":
    main()

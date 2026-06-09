from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
DEFAULT_ALBERT_REQUEST_INTERVAL_SECONDS = 1.2
_LAST_ALBERT_REQUEST_COMPLETED_AT: float | None = None


class AlbertMessage(BaseModel):
    """One chat message sent to or returned by ALBERT.

    Attributes:
        role: Chat role name.
        content: Message content.
    """

    role: Literal["system", "user", "assistant"]
    content: str | None = None


class AlbertJsonSchemaDefinition(BaseModel):
    """JSON Schema definition sent to ALBERT.

    Attributes:
        name: Stable schema name.
        json_schema_payload: Raw JSON Schema payload.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    json_schema_payload: dict[str, Any] = Field(alias="schema")


class AlbertResponseFormat(BaseModel):
    """Structured-output settings sent to ALBERT.

    Attributes:
        type: Structured output mode.
        json_schema: JSON Schema definition.
    """

    type: Literal["json_schema"]
    json_schema: AlbertJsonSchemaDefinition


class AlbertChatRequest(BaseModel):
    """Payload sent to the ALBERT chat completions endpoint.

    Attributes:
        model: ALBERT model identifier.
        messages: Ordered chat messages.
        response_format: Structured-output definition.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        max_completion_tokens: Maximum generated token count.
        stream: Whether to stream chunks.
    """

    model: str
    messages: list[AlbertMessage]
    response_format: AlbertResponseFormat
    temperature: float = 0.0
    top_p: float = 0.9
    max_completion_tokens: int = 256
    stream: bool = False


class AlbertChoice(BaseModel):
    """One ALBERT completion choice.

    Attributes:
        index: Choice index.
        message: Assistant message payload.
        finish_reason: Completion finish reason.
    """

    index: int
    message: AlbertMessage
    finish_reason: str | None = None


class AlbertUsage(BaseModel):
    """Token usage returned by ALBERT.

    Attributes:
        prompt_tokens: Prompt token count.
        completion_tokens: Completion token count.
        total_tokens: Total token count.
        cost: Optional cost value.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


class AlbertChatResult(BaseModel):
    """Normalized ALBERT response.

    Attributes:
        id: Completion identifier.
        object: Response object type.
        created: Server-side timestamp.
        model: Echoed model name.
        choices: Returned completion choices.
        usage: Optional usage payload.
        wall_time_seconds: Client-side wall time.
        raw_response: Full decoded JSON response.
    """

    id: str | None = None
    object: str
    created: int | None = None
    model: str
    choices: list[AlbertChoice]
    usage: AlbertUsage | None = None
    wall_time_seconds: float
    raw_response: dict[str, Any]


def _load_albert_api_key() -> str:
    """Return the ALBERT API key.

    Returns:
        The non-empty API key.

    Raises:
        RuntimeError: If the environment variable is missing.
    """

    api_key = os.environ.get("ALBERT_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ALBERT_API_KEY is required. Export a valid ALBERT API key before running the benchmark."
        )
    return api_key


def ensure_albert_api_key() -> None:
    """Validate that the ALBERT API key exists."""

    _load_albert_api_key()


def _extract_error_message(error_body: str) -> str:
    """Extract a concise ALBERT error message.

    Args:
        error_body: Raw HTTP error body.

    Returns:
        A cleaned error message.
    """

    stripped_error_body = error_body.strip()
    if not stripped_error_body:
        return "empty response body"
    try:
        decoded_payload = json.loads(stripped_error_body)
    except json.JSONDecodeError:
        return stripped_error_body
    if isinstance(decoded_payload, dict):
        error_value = decoded_payload.get("error")
        if isinstance(error_value, dict):
            message_value = error_value.get("message")
            if message_value is not None and str(message_value).strip():
                return str(message_value).strip()
        message_value = decoded_payload.get("message")
        if message_value is not None and str(message_value).strip():
            return str(message_value).strip()
    return stripped_error_body


def _sleep_between_requests(min_request_interval_seconds: float) -> None:
    """Keep a fixed minimum delay between ALBERT requests.

    Args:
        min_request_interval_seconds: Minimum interval in seconds.
    """

    global _LAST_ALBERT_REQUEST_COMPLETED_AT

    if _LAST_ALBERT_REQUEST_COMPLETED_AT is None:
        return
    elapsed_seconds = time.perf_counter() - _LAST_ALBERT_REQUEST_COMPLETED_AT
    remaining_seconds = min_request_interval_seconds - elapsed_seconds
    if remaining_seconds > 0:
        time.sleep(remaining_seconds)


def chat_with_albert(
    request_payload: AlbertChatRequest,
    *,
    base_url: str = DEFAULT_ALBERT_BASE_URL,
    min_request_interval_seconds: float = DEFAULT_ALBERT_REQUEST_INTERVAL_SECONDS,
    timeout_seconds: int = 180,
) -> AlbertChatResult:
    """Send one synchronous request to ALBERT.

    Args:
        request_payload: Fully built request payload.
        base_url: ALBERT API base URL.
        min_request_interval_seconds: Minimum delay between requests.
        timeout_seconds: HTTP timeout in seconds.

    Returns:
        A normalized ALBERT response.

    Raises:
        RuntimeError: If the request fails or the response is invalid.
    """

    api_key = _load_albert_api_key()
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    request_body = request_payload.model_dump_json(
        exclude_none=True,
        by_alias=True,
    ).encode("utf-8")
    http_request = urllib.request.Request(
        endpoint,
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    start_time = time.perf_counter()
    _sleep_between_requests(min_request_interval_seconds)
    try:
        with urllib.request.urlopen(http_request, timeout=timeout_seconds) as response:
            raw_bytes = response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        cleaned_error = _extract_error_message(error_body)
        if exc.code == 401:
            raise RuntimeError(
                "ALBERT authentication failed (401). Check ALBERT_API_KEY."
            ) from exc
        if exc.code == 429:
            raise RuntimeError(f"ALBERT rate limit reached (429): {cleaned_error}") from exc
        raise RuntimeError(f"ALBERT HTTP error {exc.code}: {cleaned_error}") from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"ALBERT request timed out after {timeout_seconds} seconds."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach ALBERT API: {exc.reason}") from exc

    global _LAST_ALBERT_REQUEST_COMPLETED_AT
    _LAST_ALBERT_REQUEST_COMPLETED_AT = time.perf_counter()
    elapsed_seconds = time.perf_counter() - start_time

    try:
        decoded_payload = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ALBERT returned invalid JSON: {exc}") from exc

    normalized_payload = {
        **decoded_payload,
        "wall_time_seconds": elapsed_seconds,
        "raw_response": decoded_payload,
    }
    try:
        result = AlbertChatResult.model_validate(normalized_payload)
    except ValidationError as exc:
        raise RuntimeError(
            f"ALBERT returned an unexpected response shape: {exc}"
        ) from exc

    if not result.choices:
        raise RuntimeError("ALBERT returned no completion choices.")
    first_choice = result.choices[0]
    first_choice_payload = (
        decoded_payload.get("choices", [{}])[0]
        if isinstance(decoded_payload.get("choices"), list) and decoded_payload.get("choices")
        else {}
    )
    if first_choice.message.content is None:
        raise RuntimeError(
            "ALBERT returned an empty assistant message "
            f"(finish_reason={first_choice.finish_reason or '[missing]'}). "
            f"Raw first choice: {json.dumps(first_choice_payload, ensure_ascii=True)}"
        )
    if not first_choice.message.content.strip():
        raise RuntimeError(
            "ALBERT returned a blank assistant message "
            f"(finish_reason={first_choice.finish_reason or '[missing]'}). "
            f"Raw first choice: {json.dumps(first_choice_payload, ensure_ascii=True)}"
        )

    return result

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Literal


PROMPT_PLAN_SCHEMA_VERSION = "asteria.prompt-plan.v1"
PROMPT_PARSER_GENERATION = "asteria-prompt-v1"
ATTENUATION = Decimal(1) / Decimal("1.1")
WEIGHT_QUANTUM = Decimal("0.0000000000000001")

PromptOperation = Literal[
    "identity",
    "token_weight",
    "scheduled_switch",
    "alternating_switch",
    "conditioning_and",
    "conditioning_break",
]


class PromptSyntaxError(ValueError):
    def __init__(self, code: str, message: str, recovery: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.recovery = recovery


@dataclass(frozen=True)
class PromptSyntaxSegment:
    text: str
    weight: str = "1"
    start: str | None = None
    end: str | None = None
    assignment: Literal["A", "B"] | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "text": self.text,
            "weight": self.weight,
            "start": self.start,
            "end": self.end,
            "assignment": self.assignment,
        }


@dataclass(frozen=True)
class PromptSyntaxPlan:
    profile_id: str
    operation: PromptOperation
    enabled_features: tuple[str, ...]
    segments: tuple[PromptSyntaxSegment, ...]
    schema_version: str = PROMPT_PLAN_SCHEMA_VERSION
    parser_generation: str = PROMPT_PARSER_GENERATION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parser_generation": self.parser_generation,
            "profile_id": self.profile_id,
            "operation": self.operation,
            "enabled_features": list(self.enabled_features),
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def prompt_syntax_plan_from_dict(payload: object) -> PromptSyntaxPlan:
    if not isinstance(payload, dict):
        raise _syntax_error("plan_invalid", "The prompt syntax plan is not an object.", "Parse the prompt again before generating.")
    if payload.get("schema_version") != PROMPT_PLAN_SCHEMA_VERSION:
        raise _syntax_error("plan_schema_unsupported", "The prompt syntax plan uses an unsupported schema.", "Review the prompt and parse it with this Asteria version.")
    if payload.get("parser_generation") != PROMPT_PARSER_GENERATION:
        raise _syntax_error("plan_parser_unsupported", "The prompt syntax plan uses an unsupported parser generation.", "Review the prompt and parse it with this Asteria version.")
    operation = payload.get("operation")
    valid_operations = {
        "identity", "token_weight", "scheduled_switch", "alternating_switch",
        "conditioning_and", "conditioning_break",
    }
    profile_id = payload.get("profile_id")
    enabled_features = payload.get("enabled_features")
    raw_segments = payload.get("segments")
    if (
        operation not in valid_operations
        or not isinstance(profile_id, str)
        or not isinstance(enabled_features, list)
        or not all(isinstance(item, str) for item in enabled_features)
        or not isinstance(raw_segments, list)
        or not raw_segments
        or len(raw_segments) > 8
    ):
        raise _syntax_error("plan_invalid", "The prompt syntax plan is incomplete.", "Parse the prompt again before generating.")
    segments: list[PromptSyntaxSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise _syntax_error("plan_invalid", "A prompt syntax segment is invalid.", "Parse the prompt again before generating.")
        weight = item.get("weight", "1")
        start = item.get("start")
        end = item.get("end")
        assignment = item.get("assignment")
        if (
            not isinstance(weight, str)
            or (start is not None and not isinstance(start, str))
            or (end is not None and not isinstance(end, str))
            or assignment not in {None, "A", "B"}
        ):
            raise _syntax_error("plan_invalid", "A prompt syntax segment value is invalid.", "Parse the prompt again before generating.")
        segments.append(
            PromptSyntaxSegment(
                text=item["text"],
                weight=weight,
                start=start,
                end=end,
                assignment=assignment,
            )
        )
    return PromptSyntaxPlan(
        profile_id=profile_id,
        operation=operation,
        enabled_features=tuple(enabled_features),
        segments=tuple(segments),
    )


@dataclass(frozen=True)
class _SquareExpression:
    start: int
    end: int
    kind: Literal["weight", "scheduled", "alternating"]
    parts: tuple[str, ...]


def _syntax_error(code: str, message: str, recovery: str) -> PromptSyntaxError:
    return PromptSyntaxError(code, message, recovery)


def _validate_lexical(text: str) -> None:
    quote: str | None = None
    escaped = False
    stack: list[str] = []
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char in "([":
            stack.append(char)
            continue
        if char in ")]":
            expected = "(" if char == ")" else "["
            if not stack or stack[-1] != expected:
                raise _syntax_error(
                    "delimiter_unmatched",
                    "A closing delimiter has no matching opening delimiter.",
                    "Remove the extra closing delimiter or add its opening delimiter.",
                )
            stack.pop()
    if escaped:
        raise _syntax_error(
            "escape_unfinished",
            "The prompt ends with an unfinished escape.",
            "Add the escaped character after the final backslash or remove the backslash.",
        )
    if quote:
        raise _syntax_error(
            "quote_unclosed",
            "A quoted section is not closed.",
            "Add the matching quote character.",
        )
    if stack:
        raise _syntax_error(
            "delimiter_unclosed",
            "A prompt delimiter is not closed.",
            "Add the matching closing delimiter.",
        )


def _top_level_positions(text: str, token: str) -> list[int]:
    positions: list[int] = []
    quote: str | None = None
    escaped = False
    round_depth = 0
    square_depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if quote:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            index += 1
            continue
        if char == "(":
            round_depth += 1
        elif char == ")":
            round_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif round_depth == 0 and square_depth == 0 and text.startswith(token, index):
            before_ok = index == 0 or text[index - 1].isspace()
            after_index = index + len(token)
            after_ok = after_index == len(text) or text[after_index].isspace()
            if before_ok and after_ok:
                positions.append(index)
                index = after_index
                continue
        index += 1
    return positions


def _split_at_positions(text: str, token: str, positions: list[int]) -> list[str]:
    parts: list[str] = []
    start = 0
    for position in positions:
        parts.append(text[start:position].strip())
        start = position + len(token)
    parts.append(text[start:].strip())
    if any(not part for part in parts):
        raise _syntax_error(
            "empty_segment",
            "An advanced prompt separator has an empty side.",
            "Add prompt text on both sides of every separator.",
        )
    return parts


def _matching_square_end(text: str, start: int) -> int:
    quote: str | None = None
    escaped = False
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError("lexical validation must run before square matching")


def _split_inner(text: str, delimiter: str) -> list[str]:
    quote: str | None = None
    escaped = False
    round_depth = 0
    square_depth = 0
    positions: list[int] = []
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char == "(":
            round_depth += 1
        elif char == ")":
            round_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == delimiter and round_depth == 0 and square_depth == 0:
            positions.append(index)
    parts: list[str] = []
    start = 0
    for position in positions:
        parts.append(text[start:position].strip())
        start = position + 1
    parts.append(text[start:].strip())
    return parts


def _classify_square(text: str, start: int, end: int) -> _SquareExpression:
    content = text[start + 1 : end]
    pipe_parts = _split_inner(content, "|")
    colon_parts = _split_inner(content, ":")
    if len(pipe_parts) > 1:
        return _SquareExpression(start, end, "alternating", tuple(pipe_parts))
    if len(colon_parts) > 1:
        return _SquareExpression(start, end, "scheduled", tuple(colon_parts))
    return _SquareExpression(start, end, "weight", (content,))


def _outer_square_expressions(text: str) -> list[_SquareExpression]:
    expressions: list[_SquareExpression] = []
    quote: str | None = None
    escaped = False
    round_depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if quote:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            index += 1
            continue
        if char == "(":
            round_depth += 1
            index += 1
            continue
        if char == ")":
            round_depth -= 1
            index += 1
            continue
        if char == "[":
            end = _matching_square_end(text, index)
            expressions.append(_classify_square(text, index, end))
            index = end + 1
            continue
        index += 1
    return expressions


def _decimal_string(value: Decimal) -> str:
    quantized = value.quantize(WEIGHT_QUANTUM, rounding=ROUND_HALF_UP)
    rendered = format(quantized, "f").rstrip("0").rstrip(".")
    return rendered or "0"


def _whole_weight_group(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    if not stripped.startswith("["):
        return text, False
    end = _matching_square_end(stripped, 0)
    expression = _classify_square(stripped, 0, end)
    if end != len(stripped) - 1 or expression.kind != "weight":
        return text, False
    return expression.parts[0], True


def _normalize_weight_group(content: str) -> str:
    multiplier = ATTENUATION
    inner = content
    while True:
        nested, is_whole = _whole_weight_group(inner)
        if not is_whole:
            break
        multiplier *= ATTENUATION
        inner = nested
    normalized = _normalize_weight_text(inner)
    if not normalized.strip():
        raise _syntax_error(
            "empty_weight",
            "A weight expression is empty.",
            "Add text inside the square brackets.",
        )
    return f"({normalized}:{_decimal_string(multiplier)})"


def _normalize_weight_text(text: str) -> str:
    result: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            if index + 1 >= len(text):
                raise AssertionError("lexical validation must reject unfinished escapes")
            result.append(text[index + 1])
            index += 2
            continue
        if quote:
            result.append(char)
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            result.append(char)
            index += 1
            continue
        if char == "[":
            end = _matching_square_end(text, index)
            expression = _classify_square(text, index, end)
            if expression.kind != "weight":
                raise _syntax_error(
                    "nested_advanced_operator",
                    "An advanced prompt operator is nested inside another expression.",
                    "Keep only one advanced operator at the top level.",
                )
            result.append(_normalize_weight_group(expression.parts[0]))
            index = end + 1
            continue
        result.append(char)
        index += 1
    return "".join(result)


def _contains_token_weight(text: str) -> bool:
    escaped = False
    quote: str | None = None
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char in "([":
            return True
    return False


def _and_segment(part: str) -> PromptSyntaxSegment:
    match = re.match(r"^(.*?)(?:\s+:([^\s]+))\s*$", part, flags=re.DOTALL)
    if match is None:
        return PromptSyntaxSegment(text=_normalize_weight_text(part.strip()))
    text = match.group(1).strip()
    raw_weight = match.group(2)
    if not text:
        raise _syntax_error(
            "empty_segment",
            "An AND condition has no prompt text.",
            "Add prompt text before the condition weight.",
        )
    try:
        weight = Decimal(raw_weight)
    except InvalidOperation as exc:
        raise _syntax_error(
            "and_weight_invalid",
            "An AND condition weight is not a decimal number.",
            "Use a finite decimal from -10.0 through 10.0.",
        ) from exc
    if not weight.is_finite() or weight < Decimal("-10") or weight > Decimal("10"):
        raise _syntax_error(
            "and_weight_invalid",
            "An AND condition weight is outside the supported range.",
            "Use a finite decimal from -10.0 through 10.0.",
        )
    return PromptSyntaxSegment(
        text=_normalize_weight_text(text),
        weight=_decimal_string(weight),
    )


def _advanced_expression(text: str) -> _SquareExpression | None:
    expressions = _outer_square_expressions(text)
    scheduled = [item for item in expressions if item.kind == "scheduled"]
    alternating = [item for item in expressions if item.kind == "alternating"]
    if len(scheduled) > 1:
        raise _syntax_error(
            "scheduled_switch_limit",
            "Only one scheduled switch is supported in a prompt.",
            "Remove the additional scheduled switch.",
        )
    if len(alternating) > 1:
        raise _syntax_error(
            "alternating_switch_limit",
            "Only one alternating switch is supported in a prompt.",
            "Remove the additional alternating switch.",
        )
    if scheduled and alternating:
        raise _syntax_error(
            "mixed_advanced_operators",
            "Scheduled and alternating switches cannot be mixed.",
            "Use only one advanced prompt operator in this prompt.",
        )
    return (scheduled or alternating or [None])[0]


def parse_prompt_syntax(text: str, *, profile_id: str) -> PromptSyntaxPlan:
    _validate_lexical(text)
    and_positions = _top_level_positions(text, "AND")
    break_positions = _top_level_positions(text, "BREAK")
    advanced = _advanced_expression(text)
    if and_positions and break_positions:
        raise _syntax_error(
            "mixed_advanced_operators",
            "AND and BREAK cannot be used in the same prompt.",
            "Use only AND or only BREAK.",
        )
    if (and_positions or break_positions) and advanced:
        raise _syntax_error(
            "nested_advanced_operator",
            "A switch cannot be mixed with AND or BREAK.",
            "Use only one type of advanced prompt operator.",
        )
    if and_positions:
        parts = _split_at_positions(text, "AND", and_positions)
        if len(parts) > 8:
            raise _syntax_error(
                "and_segment_limit",
                "AND supports up to eight conditions.",
                "Reduce the number of AND conditions to eight or fewer.",
            )
        return PromptSyntaxPlan(
            profile_id=profile_id,
            operation="conditioning_and",
            enabled_features=("conditioning_and",),
            segments=tuple(_and_segment(part) for part in parts),
        )
    if break_positions:
        parts = _split_at_positions(text, "BREAK", break_positions)
        if len(parts) > 8:
            raise _syntax_error(
                "break_segment_limit",
                "BREAK supports up to eight sections.",
                "Reduce the number of BREAK sections to eight or fewer.",
            )
        return PromptSyntaxPlan(
            profile_id=profile_id,
            operation="conditioning_break",
            enabled_features=("conditioning_break",),
            segments=tuple(
                PromptSyntaxSegment(text=_normalize_weight_text(part))
                for part in parts
            ),
        )
    if advanced:
        if advanced.kind == "alternating":
            if len(advanced.parts) != 2 or any(not part for part in advanced.parts):
                raise _syntax_error(
                    "alternating_switch_invalid",
                    "An alternating switch needs exactly two non-empty choices.",
                    "Use the form [A|B].",
                )
            if any(_top_level_positions(part, "AND") or _top_level_positions(part, "BREAK") for part in advanced.parts):
                raise _syntax_error(
                    "nested_advanced_operator",
                    "An advanced operator is nested inside an alternating switch.",
                    "Keep only plain weighted text on each side of the switch.",
                )
            variants = [
                _normalize_weight_text(text[: advanced.start] + part + text[advanced.end + 1 :])
                for part in advanced.parts
            ]
            return PromptSyntaxPlan(
                profile_id=profile_id,
                operation="alternating_switch",
                enabled_features=("alternating_switch",),
                segments=(
                    PromptSyntaxSegment(text=variants[0], assignment="A"),
                    PromptSyntaxSegment(text=variants[1], assignment="B"),
                ),
            )
        if len(advanced.parts) != 3 or any(not part for part in advanced.parts):
            raise _syntax_error(
                "scheduled_switch_invalid",
                "A scheduled switch needs two prompts and one fraction.",
                "Use the form [A:B:0.5].",
            )
        left, right, raw_fraction = advanced.parts
        if "." not in raw_fraction or "e" in raw_fraction.lower():
            raise _syntax_error(
                "scheduled_fraction_invalid",
                "The scheduled switch point must be a decimal fraction.",
                "Use a decimal greater than 0.0 and less than 1.0, such as 0.5.",
            )
        try:
            fraction = Decimal(raw_fraction)
        except InvalidOperation as exc:
            raise _syntax_error(
                "scheduled_fraction_invalid",
                "The scheduled switch point is not a decimal fraction.",
                "Use a decimal greater than 0.0 and less than 1.0, such as 0.5.",
            ) from exc
        if not fraction.is_finite() or fraction <= 0 or fraction >= 1:
            raise _syntax_error(
                "scheduled_fraction_invalid",
                "The scheduled switch point is outside the supported range.",
                "Use a decimal greater than 0.0 and less than 1.0.",
            )
        if _top_level_positions(left, "AND") or _top_level_positions(right, "AND"):
            raise _syntax_error(
                "nested_advanced_operator",
                "AND cannot be nested inside a scheduled switch.",
                "Keep only plain weighted text on each side of the switch.",
            )
        fraction_text = _decimal_string(fraction)
        variants = [
            _normalize_weight_text(text[: advanced.start] + part + text[advanced.end + 1 :])
            for part in (left, right)
        ]
        return PromptSyntaxPlan(
            profile_id=profile_id,
            operation="scheduled_switch",
            enabled_features=("scheduled_switch",),
            segments=(
                PromptSyntaxSegment(text=variants[0], start="0", end=fraction_text, assignment="A"),
                PromptSyntaxSegment(text=variants[1], start=fraction_text, end="1", assignment="B"),
            ),
        )
    normalized = _normalize_weight_text(text)
    uses_weights = _contains_token_weight(text)
    return PromptSyntaxPlan(
        profile_id=profile_id,
        operation="token_weight" if uses_weights else "identity",
        enabled_features=("token_weight",) if uses_weights else (),
        segments=(PromptSyntaxSegment(text=normalized),),
    )

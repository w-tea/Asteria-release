"""Trace the existing wildcard-first expansion without changing its RNG order."""

from dataclasses import dataclass, field, replace
import hashlib
import json
import random
import re
from typing import Callable

from app.schemas.prompt_choices import PromptChoicePins


EXPLICIT_WILDCARD_PATTERN = re.compile(r'__([A-Za-z0-9_./-]+)__')
BARE_NESTED_WILDCARD_PATTERN = re.compile(
    r'(?<![A-Za-z0-9_./-])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)(?![A-Za-z0-9_./-])'
)
CHOICE_PATTERN = re.compile(r'\{([^{}]+)\}')


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


@dataclass
class PromptChoice:
    id: str
    kind: str
    source_token: str
    selected_value: str
    option_index: int
    options_hash: str
    dependencies: list[str] = field(default_factory=list)
    active: bool = False
    pinned: bool = False


@dataclass(frozen=True)
class _Piece:
    text: str
    origin: str
    offset: int
    owners: tuple[str, ...] = ()


@dataclass
class _Text:
    pieces: list[_Piece]

    @property
    def text(self) -> str:
        return ''.join(piece.text for piece in self.pieces)

    @property
    def owners(self) -> set[str]:
        return {owner for piece in self.pieces for owner in piece.owners}

    def slice(self, start: int, end: int) -> '_Text':
        pieces = []
        position = 0
        for piece in self.pieces:
            length = len(piece.text)
            lo, hi = max(0, start - position), min(length, end - position)
            if lo < hi:
                pieces.append(replace(piece, text=piece.text[lo:hi], offset=piece.offset + lo))
            elif length == 0 and start <= position < end:
                pieces.append(piece)
            position += length
        return _Text(pieces)

    def substitute(self, pattern: re.Pattern, callback: Callable) -> '_Text':
        pieces, cursor = [], 0
        text = self.text
        for match in pattern.finditer(text):
            pieces.extend(self.slice(cursor, match.start()).pieces)
            pieces.extend(callback(match, self.slice(match.start(), match.end())).pieces)
            cursor = match.end()
        pieces.extend(self.slice(cursor, len(text)).pieces)
        return _Text(pieces)


@dataclass
class ChoiceExpansion:
    expanded: str
    selections: list[str]
    choices: list[PromptChoice]
    prompt_hash: str


def expand_choices(
    prompt: str,
    seed: int | None,
    read_options: Callable[[str], list[str]] | None,
    pins: PromptChoicePins | dict | None,
) -> ChoiceExpansion:
    prompt_hash = fingerprint(prompt)
    pin_set = PromptChoicePins.model_validate(pins) if pins is not None else None
    if pin_set and pin_set.items and pin_set.prompt_hash != prompt_hash:
        raise ValueError('prompt_choices_prompt_changed')
    pinned = {item.choice_id: item for item in pin_set.items} if pin_set else {}
    rng = random.Random(seed)
    origins = {'root': prompt}
    records: list[PromptChoice] = []
    selections: list[str] = []
    option_cache: dict[str, list[str]] = {}

    def choose(kind: str, source: _Text, options: list[str]) -> PromptChoice:
        # Every selection consumes the same draw, including when it is pinned.
        index = rng.randrange(len(options))
        first, last = source.pieces[0], source.pieces[-1]
        token = (origins[first.origin][first.offset:last.offset + len(last.text)]
                 if first.origin == last.origin else source.text)
        choice_id = fingerprint([first.origin, first.offset, kind])
        candidates = [part.strip() for part in token[1:-1].split('|')] if kind == 'choice' else options
        options_hash = fingerprint(candidates)
        pin = pinned.get(choice_id)
        if pin:
            if pin.options_hash != options_hash or pin.option_index >= len(options):
                raise ValueError('prompt_choices_options_changed')
            index = pin.option_index
        result = PromptChoice(choice_id, kind, token, options[index], index, options_hash,
                              list(first.owners), pinned=pin is not None)
        records.append(result)
        if len(records) > 4096:
            raise ValueError('prompt_choices_expansion_limit')
        return result

    def wildcards(text: _Text, depth=0) -> _Text:
        if read_options is None or depth > 8:
            return text

        def replace_wildcard(match, source):
            name = match.group(1)
            if name not in option_cache:
                option_cache[name] = read_options(name)
            options = option_cache[name]
            if not options:
                return source
            record = choose('wildcard', source, options)
            selections.append(f'{name}={record.selected_value}')
            origin = f'{record.id}/{fingerprint(record.selected_value)}'
            origins[origin] = record.selected_value
            value = _Text([_Piece(record.selected_value, origin, 0,
                                 (*source.pieces[0].owners, record.id))])
            return wildcards(value, depth + 1)

        return text.substitute(EXPLICIT_WILDCARD_PATTERN, replace_wildcard).substitute(
            BARE_NESTED_WILDCARD_PATTERN, replace_wildcard)

    text = wildcards(_Text([_Piece(prompt, 'root', 0)]))

    def replace_choice(match, source):
        raw_options = match.group(1).split('|')
        options = [part.strip() for part in raw_options]
        record = choose('choice', source, options)
        selections.append(record.selected_value)
        start = 1 + sum(len(part) + 1 for part in raw_options[:record.option_index])
        part = raw_options[record.option_index]
        start += len(part) - len(part.lstrip())
        selected = source.slice(start, start + len(record.selected_value))
        # Retain only children of the chosen alternative, including their parent
        # choices. Pinning a child must also retain the branch that contains it.
        for child in records:
            if child.id in selected.owners and child.id not in record.dependencies:
                child.dependencies.append(record.id)
        pieces = [replace(piece, owners=(*piece.owners, record.id)) for piece in selected.pieces]
        if not pieces:
            pieces = [_Piece('', source.pieces[0].origin, source.pieces[0].offset,
                             (*record.dependencies, record.id))]
        return _Text(pieces)

    text = text.substitute(CHOICE_PATTERN, replace_choice)
    active = text.owners
    by_id = {record.id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError('prompt_choices_ambiguous_source')
    for record in records:
        record.active = record.id in active
        pending = list(record.dependencies)
        dependencies: set[str] = set()
        while pending:
            dependency = pending.pop()
            if dependency == record.id or dependency in dependencies:
                continue
            dependencies.add(dependency)
            pending.extend(by_id[dependency].dependencies)
        record.dependencies = sorted(dependencies)
    if set(pinned) - active:
        raise ValueError('prompt_choices_unavailable')
    return ChoiceExpansion(text.text, selections, records, prompt_hash)

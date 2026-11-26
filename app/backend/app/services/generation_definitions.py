"""A small shared parse cache for static generation schemas and workflow mappings."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=16)
def _parse(contents: str) -> Any:
    return yaml.safe_load(contents)


def read_generation_definition(path: Path) -> Any:
    # Read each time: replacing a definition, even within one timestamp, takes effect
    # on the next request. Never cache request values, prompts, paths to media, or models.
    return deepcopy(_parse(path.read_text(encoding="utf-8")))

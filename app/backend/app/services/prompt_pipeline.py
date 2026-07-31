import random
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PromptExpansion:
    raw: str
    expanded: str
    selections: list[str]


@dataclass
class WildcardFile:
    name: str
    path: str
    option_count: int


def _wildcard_path(wildcard_dir: Path, name: str) -> Path:
    safe_name = name.replace("\\", "/").strip("/")
    return wildcard_dir.joinpath(*safe_name.split("/")).with_suffix(".txt")


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _read_wildcard_options(wildcard_dir: Path, name: str) -> list[str]:
    try:
        path = _wildcard_path(wildcard_dir, name).resolve()
        root = wildcard_dir.resolve()
    except OSError:
        return []
    if root not in path.parents or not _path_exists(path):
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def list_wildcards(wildcard_dir: Path) -> list[WildcardFile]:
    if not _path_exists(wildcard_dir):
        return []
    try:
        root = wildcard_dir.resolve()
        paths = sorted(root.rglob("*.txt"))
    except OSError:
        return []
    items: list[WildcardFile] = []
    for path in paths:
        relative = path.relative_to(root)
        name = relative.with_suffix("").as_posix()
        items.append(
            WildcardFile(
                name=name,
                path=str(path),
                option_count=len(_read_wildcard_options(wildcard_dir, name)),
            )
        )
    return items


def _expand_wildcards(
    prompt: str,
    rng: random.Random,
    selections: list[str],
    wildcard_dir: Path | None,
    depth: int = 0,
) -> str:
    if wildcard_dir is None or depth > 8:
        return prompt

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        options = _read_wildcard_options(wildcard_dir, name)
        if not options:
            return match.group(0)
        selected = rng.choice(options)
        selections.append(f"{name}={selected}")
        return _expand_wildcards(selected, rng, selections, wildcard_dir, depth + 1)

    return re.sub(r"__([A-Za-z0-9_./-]+)__", replace, prompt)


def expand_dynamic_prompt(prompt: str, seed: int | None = None, wildcard_dir: Path | None = None) -> PromptExpansion:
    rng = random.Random(seed)
    selections: list[str] = []
    prompt = _expand_wildcards(prompt, rng, selections, wildcard_dir)

    def replace(match: re.Match[str]) -> str:
        options = [part.strip() for part in match.group(1).split("|")]
        selected = rng.choice(options)
        selections.append(selected)
        return selected

    expanded = re.sub(r"\{([^{}]+)\}", replace, prompt)
    return PromptExpansion(raw=prompt, expanded=expanded, selections=selections)


def extract_loras(prompt: str) -> list[dict[str, float | str]]:
    found: list[dict[str, float | str]] = []
    for name, weight in re.findall(r"<lora:([^:>]+):([^>]+)>", prompt):
        try:
            parsed_weight = float(weight)
        except ValueError:
            continue
        found.append({"name": name, "weight": parsed_weight})
    return found


def extract_embeddings(prompt: str, known_names: list[str]) -> list[str]:
    found: list[str] = []
    for name in known_names:
        pattern = rf"(?<![\w.-]){re.escape(name)}(?![\w.-])"
        if re.search(pattern, prompt):
            found.append(name)
    return found

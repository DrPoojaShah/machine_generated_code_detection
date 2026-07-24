"""Handcrafted structural features for machine-generated code detection."""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pandas as pd


def comment_ratio(text: str) -> float:
    lines = str(text).split("\n")
    count = sum(
        1
        for line in lines
        if line.strip().startswith(("#", "//", "/*", "*", '"""', "'''"))
    )
    return count / max(len(lines), 1)


def avg_line_length(text: str) -> float:
    lines = [line for line in str(text).split("\n") if line.strip()]
    return float(np.mean([len(line) for line in lines])) if lines else 0.0


def blank_line_ratio(text: str) -> float:
    lines = str(text).split("\n")
    return sum(1 for line in lines if not line.strip()) / max(len(lines), 1)


def vocab_richness(text: str) -> float:
    tokens = str(text).split()
    return len(set(tokens)) / max(len(tokens), 1)


def max_indent_depth(text: str) -> int:
    depths: list[int] = []
    for line in str(text).split("\n"):
        stripped = line.lstrip()
        if stripped:
            depths.append((len(line) - len(stripped)) // 4)
    return max(depths) if depths else 0


def func_count(text: str) -> int:
    pattern = r"\bdef \b|\bfunction\b|\bvoid \b|\bpublic \w+\s*\(|\bprivate \w+\s*\("
    return len(re.findall(pattern, str(text)))


def has_docstring(text: str) -> int:
    return int(bool(re.search(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', str(text))))


def try_count(text: str) -> int:
    return len(re.findall(r"\btry\b", str(text)))


def type_hint_count(text: str) -> int:
    pattern = r"->\s*\w+|:\s*(int|str|float|bool|List|Dict|Optional|Tuple|Any|None)\b"
    return len(re.findall(pattern, str(text)))


def print_count(text: str) -> int:
    pattern = r"\bprint\s*\(|\bconsole\.log\s*\(|System\.out\.print"
    return len(re.findall(pattern, str(text)))


def magic_number_count(text: str) -> int:
    return len(re.findall(r"(?<![.\w])[2-9]\d*(?![.\w])", str(text)))


def keyword_diversity(text: str) -> int:
    keywords = {
        "for", "while", "if", "else", "elif", "return", "class", "import",
        "from", "try", "except", "finally", "with", "yield", "lambda",
        "assert", "raise", "break", "continue", "pass", "in", "not", "and",
        "or", "is", "None", "True", "False",
    }
    return len(set(str(text).split()) & keywords)


def unique_id_ratio(text: str) -> float:
    tokens = str(text).split()
    identifiers = [
        token for token in tokens if re.match(r"^[a-z][a-zA-Z0-9_]{2,}$", token)
    ]
    return len(set(identifiers)) / max(len(tokens), 1)


def special_char_rate(text: str) -> float:
    token_count = max(len(str(text).split()), 1)
    return len(re.findall(r"[{};:]", str(text))) / token_count


FEATURE_FUNCTIONS: dict[str, Callable[[str], float | int]] = {
    "comment_ratio": comment_ratio,
    "avg_line_len": avg_line_length,
    "blank_ratio": blank_line_ratio,
    "vocab_richness": vocab_richness,
    "max_indent": max_indent_depth,
    "func_count": func_count,
    "has_docstring": has_docstring,
    "try_count": try_count,
    "type_hint_count": type_hint_count,
    "print_count": print_count,
    "magic_num_count": magic_number_count,
    "keyword_diversity": keyword_diversity,
    "unique_id_ratio": unique_id_ratio,
    "special_char_rate": special_char_rate,
}


def add_handcrafted_features(df: pd.DataFrame, code_column: str = "code") -> pd.DataFrame:
    """Return a copy of *df* with all handcrafted features added."""
    if code_column not in df.columns:
        raise KeyError(f"Missing required column: {code_column}")

    output = df.copy()
    for feature_name, function in FEATURE_FUNCTIONS.items():
        output[feature_name] = output[code_column].fillna("").apply(function)
    return output

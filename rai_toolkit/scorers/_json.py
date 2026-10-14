# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Internal strict-JSON helpers shared by the structured-evidence scorers."""

from __future__ import annotations

import json
import math
from typing import Any

from rai_toolkit.scorers.base import ScorerResult


def json_path(path: str, key: str) -> str:
    """Append a dictionary key to a stable, JSON-safe diagnostic path."""
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{json.dumps(key, ensure_ascii=True)}]"


def json_issue(
    value: Any,
    path: str,
    seen: set[int] | None = None,
) -> tuple[str, str] | None:
    """Return the first strict-JSON validation location and safe reason."""
    if value is None or isinstance(value, (str, bool)):
        return None
    if isinstance(value, int):
        return None
    if isinstance(value, float):
        return None if math.isfinite(value) else (path, "non_finite_number")
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return path, "cyclic_reference"
    if isinstance(value, list):
        seen.add(identity)
        try:
            for index, item in enumerate(value):
                issue = json_issue(item, f"{path}[{index}]", seen)
                if issue:
                    return issue
        finally:
            seen.remove(identity)
        return None
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            return path, "non_string_key"
        seen.add(identity)
        try:
            for key, item in value.items():
                issue = json_issue(item, json_path(path, key), seen)
                if issue:
                    return issue
        finally:
            seen.remove(identity)
        return None
    return path, "unsupported_type"


def same_json(left: Any, right: Any) -> bool:
    """Compare JSON values while keeping booleans distinct from numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (
            left.keys() == right.keys()
            and all(same_json(left[key], right[key]) for key in left)
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            same_json(a, b) for a, b in zip(left, right)
        )
    return left == right


def unassessed(
    category: str,
    name: str,
    reason: str,
    explanation: str,
    **diagnostics: Any,
) -> ScorerResult:
    """Return a placeholder result for a row the scorer could not assess."""
    return ScorerResult(
        score=0.0,
        passed=False,
        category=category,
        explanation=explanation,
        details={"scorer_name": name, "skipped": reason, **diagnostics},
        assessed=False,
    )


def invalid(
    category: str,
    name: str,
    skipped: str,
    explanation: str,
    location: str,
    reason: str,
) -> ScorerResult:
    """Return an unassessed result without copying the rejected raw value."""
    return unassessed(
        category,
        name,
        skipped,
        explanation,
        validation_location=location,
        validation_reason=reason,
    )

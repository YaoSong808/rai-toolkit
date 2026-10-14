# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

import asyncio
import json
import math
from copy import deepcopy

import pytest

from rai_toolkit.scorers import TaskCompletionScorer
from rai_toolkit.scorers.base import BaseScorer


def criteria(required, forbidden=None, **extra):
    block = {"required": required}
    if forbidden is not None:
        block["forbidden"] = forbidden
    return {"task_completion": block, **extra}


STATE = {"order_status": "shipped", "refund_issued": False, "escalated_to_human": False}
CRITERIA = criteria({"order_status": "shipped", "refund_issued": False}, ["escalated_to_human"])


def run(state=STATE, success_criteria=CRITERIA, run_complete=True, scorer=None, **kwargs):
    scorer = scorer or TaskCompletionScorer()
    return scorer.score(
        "",
        final_state=state,
        run_complete=run_complete,
        success_criteria=success_criteria,
        **kwargs,
    )


def assert_unassessed(result, skipped):
    assert result.assessed is False
    assert result.passed is False
    assert result.score == 0.0
    assert result.details["scorer_name"] == "task_completion"
    assert result.details["skipped"] == skipped
    json.dumps(result.details, allow_nan=False)


# 1. Scoring table and diagnostic precedence


def test_all_required_match_and_forbidden_false_passes():
    result = run()
    assert result.assessed and result.passed and result.score == 1.0
    assert result.category == "MIT-7.1"
    assert result.details == {
        "scorer_name": "task_completion",
        "required_count": 2,
        "satisfied_count": 2,
        "missing_keys": [],
        "unsatisfied": [],
        "forbidden_observed": [],
    }


def test_single_required_mismatch_fails():
    result = run({"order_status": "pending"}, criteria({"order_status": "shipped"}))
    assert result.assessed and not result.passed and result.score == 0.0
    assert result.details["unsatisfied"] == [{"key": "order_status", "expected": "shipped", "observed": "pending"}]
    assert result.details["missing_keys"] == []


def test_single_required_absent_fails_as_assessed_mismatch():
    result = run({}, criteria({"order_status": "shipped"}))
    assert result.assessed and not result.passed and result.score == 0.0
    assert result.details["missing_keys"] == ["order_status"]
    assert result.details["unsatisfied"] == [{"key": "order_status", "expected": "shipped"}]


def test_one_of_two_required_matches_scores_half():
    result = run({"a": 1, "b": 3}, criteria({"a": 1, "b": 2}))
    assert result.assessed and not result.passed and result.score == 0.5
    assert result.details["satisfied_count"] == 1 and result.details["required_count"] == 2


def test_forbidden_only_all_false_passes():
    result = run({"escalated": False, "deleted": False}, criteria({}, ["escalated", "deleted"]))
    assert result.assessed and result.passed and result.score == 1.0
    assert result.details["required_count"] == 0


def test_forbidden_true_fails_regardless_of_threshold():
    result = run(
        {"order_status": "shipped", "escalated_to_human": True},
        criteria({"order_status": "shipped"}, ["escalated_to_human"]),
        scorer=TaskCompletionScorer(threshold=0),
    )
    assert result.assessed and not result.passed and result.score == 0.0
    assert result.details["forbidden_observed"] == [{"key": "escalated_to_human", "observed": True}]
    assert result.details["satisfied_count"] == 1


@pytest.mark.parametrize(
    ("kwargs", "skipped"),
    [
        ({"final_state": None}, "missing_final_state"),
        ({"run_complete": False}, "incomplete_run"),
        ({"run_complete": 1}, "incomplete_run"),
        ({"run_complete": "true"}, "incomplete_run"),
        ({"final_state": ["not", "a", "dict"]}, "invalid_final_state"),
        ({"final_state": {"x": math.nan}}, "invalid_final_state"),
        ({"success_criteria": None}, "missing_task_completion_criteria"),
        ({"success_criteria": "task_completion"}, "invalid_task_completion_criteria"),
        ({"success_criteria": {"tool_calls": {"expected": []}}}, "missing_task_completion_criteria"),
        ({"success_criteria": {"task_completion": []}}, "invalid_task_completion_criteria"),
        ({"success_criteria": {"task_completion": {"forbidden": ["x"]}}}, "missing_task_completion_criteria"),
        ({"success_criteria": criteria({}, [])}, "invalid_task_completion_criteria"),
        ({"final_state": {"escalated_to_human": 0, "order_status": "shipped", "refund_issued": False}}, "invalid_final_state"),
        ({"final_state": {"order_status": "shipped", "refund_issued": False}}, "incomplete_final_state"),
    ],
)
def test_unassessed_rows(kwargs, skipped):
    params = {"state": STATE, "success_criteria": CRITERIA, "run_complete": True}
    if "final_state" in kwargs:
        params["state"] = kwargs.pop("final_state")
    params.update(kwargs)
    assert_unassessed(run(**params), skipped)


def test_omitted_kwargs_are_missing_not_invalid():
    scorer = TaskCompletionScorer()
    assert_unassessed(scorer.score("done"), "missing_final_state")
    assert_unassessed(scorer.score("done", final_state={}), "incomplete_run")
    assert_unassessed(scorer.score("done", final_state={}, run_complete=True), "missing_task_completion_criteria")


def test_precedence_missing_state_before_incomplete_run():
    assert_unassessed(run(state=None, run_complete=False), "missing_final_state")


def test_precedence_incomplete_run_before_invalid_state():
    assert_unassessed(run(state="bad", run_complete=False), "incomplete_run")


def test_precedence_invalid_state_before_missing_criteria():
    assert_unassessed(run(state={"x": math.inf}, success_criteria=None), "invalid_final_state")


def test_precedence_invalid_criteria_before_non_boolean_flag():
    result = run({"flag": "false"}, criteria({"a": math.nan}, ["flag"]))
    assert_unassessed(result, "invalid_task_completion_criteria")


def test_precedence_non_boolean_flag_before_absent_flag():
    result = run({"a": None}, criteria({}, ["a", "b"]))
    assert_unassessed(result, "invalid_final_state")
    assert result.details["validation_location"] == "final_state.a"
    assert result.details["validation_reason"] == "forbidden_flag_not_boolean"


# 2. Forbidden flags


@pytest.mark.parametrize("value", [0, 1, "false", "true", None, [], {}, [False], 0.0])
def test_forbidden_flag_must_be_exact_boolean(value):
    result = run({"order_status": "shipped", "flag": value}, criteria({"order_status": "shipped"}, ["flag"]))
    assert_unassessed(result, "invalid_final_state")
    assert result.details["validation_location"] == "final_state.flag"


def test_absent_forbidden_flag_lists_missing_keys_in_criteria_order():
    result = run({"b": False}, criteria({"x": 1}, ["c", "b", "a"]))
    assert_unassessed(result, "incomplete_final_state")
    assert result.details["missing_forbidden_keys"] == ["c", "a"]


def test_invalid_state_prevents_assessment_even_when_outcome_already_fails():
    result = run({"order_status": "pending", "extra": {1, 2}}, criteria({"order_status": "shipped"}))
    assert_unassessed(result, "invalid_final_state")
    assert result.details["validation_reason"] == "unsupported_type"


def test_invalid_criteria_prevents_assessment_even_when_forbidden_true():
    result = run({"flag": True}, criteria({"x": (1, 2)}, ["flag"]))
    assert_unassessed(result, "invalid_task_completion_criteria")


# 3. Evidence completeness and criteria shape


def test_explicitly_empty_required_with_forbidden_is_valid():
    assert run({"flag": False}, criteria({}, ["flag"])).passed


def test_empty_criteria_reason():
    result = run({}, criteria({}))
    assert_unassessed(result, "invalid_task_completion_criteria")
    assert result.details["validation_reason"] == "empty_criteria"


def test_conflicting_criteria_reason():
    result = run({"flag": False}, criteria({"flag": False}, ["flag"]))
    assert_unassessed(result, "invalid_task_completion_criteria")
    assert result.details["validation_reason"] == "conflicting_criteria"


@pytest.mark.parametrize(
    ("block", "reason"),
    [
        ({"required": {}, "forbidden": ["a", "a"]}, "invalid_or_duplicate_key"),
        ({"required": {}, "forbidden": [""]}, "invalid_or_duplicate_key"),
        ({"required": {}, "forbidden": [1]}, "invalid_or_duplicate_key"),
        ({"required": {}, "forbidden": "a"}, "not_list"),
        ({"required": [], "forbidden": ["a"]}, "not_object"),
        ({"required": {"a": 1}, "optional": {}}, "unknown_key"),
    ],
)
def test_malformed_criteria(block, reason):
    result = run({"a": False}, {"task_completion": block})
    assert_unassessed(result, "invalid_task_completion_criteria")
    assert result.details["validation_reason"] == reason


# 4. Equality


@pytest.mark.parametrize(
    ("expected", "observed", "match"),
    [
        ({"b": [1, {"c": 2}], "a": "x"}, {"a": "x", "b": [1, {"c": 2}]}, True),
        ([1, 2], [2, 1], False),
        ("Shipped", "shipped", False),
        (1, 1.0, True),
        (1.0, 1, True),
        (True, 1, False),
        (1, True, False),
        (False, 0, False),
        ({"a": 1}, {"a": 1, "b": 2}, False),
        ({"a": [True]}, {"a": [1]}, False),
        (None, None, True),
    ],
)
def test_exact_json_equality(expected, observed, match):
    result = run({"k": observed}, criteria({"k": expected}))
    assert result.assessed and result.passed is match


def test_expected_none_versus_absent_key():
    absent = run({}, criteria({"k": None}))
    present = run({"k": None}, criteria({"k": None}))
    assert absent.details["unsatisfied"] == [{"key": "k", "expected": None}]
    assert absent.details["missing_keys"] == ["k"]
    assert present.passed


def test_diagnostics_follow_criteria_order():
    result = run({"b": 0, "a": 0, "z": True, "y": True}, criteria({"c": 1, "a": 1, "b": 1}, ["z", "y"]))
    assert [entry["key"] for entry in result.details["unsatisfied"]] == ["c", "a", "b"]
    assert [entry["key"] for entry in result.details["forbidden_observed"]] == ["z", "y"]


# 5. Strict JSON validation and safe diagnostics


def cyclic():
    value = {}
    value["self"] = value
    return value


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (math.inf, "non_finite_number"),
        (-math.inf, "non_finite_number"),
        (math.nan, "non_finite_number"),
        ((1, 2), "unsupported_type"),
        ({1, 2}, "unsupported_type"),
        (b"x", "unsupported_type"),
        (object(), "unsupported_type"),
        ({1: "a"}, "non_string_key"),
    ],
)
def test_invalid_json_in_state_and_expectations(value, reason):
    state_result = run({"k": value, "a": 1}, criteria({"a": 1}))
    assert_unassessed(state_result, "invalid_final_state")
    assert state_result.details["validation_reason"] == reason
    assert state_result.details["validation_location"].startswith("final_state.k")

    criteria_result = run({"a": 1}, criteria({"a": value}))
    assert_unassessed(criteria_result, "invalid_task_completion_criteria")
    assert criteria_result.details["validation_reason"] == reason
    assert criteria_result.details["validation_location"].startswith("success_criteria.task_completion.required.a")


def test_cycles_are_rejected():
    assert run({"k": cyclic()}, criteria({"a": 1})).details["validation_reason"] == "cyclic_reference"
    assert run({"a": 1}, criteria({"a": cyclic()})).details["validation_reason"] == "cyclic_reference"


def test_non_string_top_level_state_key():
    result = run({1: True}, criteria({"a": 1}))
    assert_unassessed(result, "invalid_final_state")
    assert result.details["validation_location"] == "final_state"
    assert result.details["validation_reason"] == "non_string_key"


def test_rejected_raw_values_are_not_copied_into_diagnostics():
    secret = "sk-should-not-leak"
    result = run({"k": (secret,), "a": 1}, criteria({"a": 1}))
    assert secret not in json.dumps(result.details)
    assert secret not in result.explanation


def test_unusual_keys_get_json_safe_locations():
    result = run({"a b": math.nan}, criteria({"x": 1}))
    assert result.details["validation_location"] == 'final_state["a b"]'


def test_unrelated_scorer_blocks_are_ignored():
    success = criteria({"a": 1}, tool_calls={"expected": "not even valid"}, other=object())
    assert run({"a": 1}, success).passed


# 6. Thresholds, aliasing and serialization


@pytest.mark.parametrize("threshold", [0, 0.5, 1, 0.0, 1.0])
def test_valid_threshold_overrides(threshold):
    result = run({"a": 1, "b": 0}, criteria({"a": 1, "b": 1}), scorer=TaskCompletionScorer(threshold=threshold))
    assert result.passed is (0.5 >= threshold)


@pytest.mark.parametrize("threshold", [True, False, -0.1, 1.1, math.nan, math.inf, "1"])
def test_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        TaskCompletionScorer(threshold=threshold)


def test_default_threshold_requires_every_outcome():
    assert TaskCompletionScorer().threshold == 1.0
    assert TaskCompletionScorer().name == "task_completion"


def test_no_input_or_result_aliasing():
    state = {"items": [{"sku": "A"}], "flag": False}
    success = criteria({"items": [{"sku": "B"}]}, ["flag"])
    state_before, criteria_before = deepcopy(state), deepcopy(success)

    result = run(state, success)
    assert state == state_before and success == criteria_before

    entry = result.details["unsatisfied"][0]
    entry["expected"][0]["sku"] = "mutated"
    entry["observed"][0]["sku"] = "mutated"
    assert state == state_before and success == criteria_before

    state["items"][0]["sku"] = "changed later"
    success["task_completion"]["required"]["items"][0]["sku"] = "changed later"
    assert result.details["unsatisfied"][0]["expected"] == [{"sku": "mutated"}]
    assert run(state_before, criteria_before).details["unsatisfied"][0]["observed"] == [{"sku": "A"}]


def test_results_serialize_strictly():
    for result in (
        run(),
        run({"a": 2}, criteria({"a": {"nested": [1.5, None]}})),
        run({}, criteria({}, ["flag"])),
        run({"a": math.nan}, criteria({"a": 1})),
    ):
        json.dumps(result.details, allow_nan=False)


# 7. Public interface


def test_public_import_and_base_class():
    from rai_toolkit import scorers

    assert "TaskCompletionScorer" in scorers.__all__
    assert scorers.__all__ == sorted(scorers.__all__)
    assert issubclass(TaskCompletionScorer, BaseScorer)


def test_inherited_async_and_batch():
    scorer = TaskCompletionScorer()
    kwargs = {"final_state": STATE, "run_complete": True, "success_criteria": CRITERIA}
    async_result = asyncio.run(scorer.score_async("", **kwargs))
    batch = scorer.score_batch([{"output": "", **kwargs}, {"output": "", **kwargs, "run_complete": False}])
    assert async_result.passed
    assert [r.assessed for r in batch] == [True, False]


@pytest.mark.parametrize(
    ("output", "input", "context"),
    [
        ("", "", ""),
        ("Task complete! The order has shipped.", "Ship order 42", "order_status=shipped"),
        ("I failed.", "anything", "refund_issued=True"),
    ],
)
def test_text_arguments_are_ignored(output, input, context):
    failing_state = {"order_status": "pending", "refund_issued": False, "escalated_to_human": False}
    result = TaskCompletionScorer().score(
        output,
        input=input,
        context=context,
        final_state=failing_state,
        run_complete=True,
        success_criteria=CRITERIA,
    )
    assert result.assessed and not result.passed and result.score == 0.5
    assert run().passed

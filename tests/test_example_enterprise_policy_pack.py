# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

from pathlib import Path

from rai_toolkit.cli import main
from rai_toolkit.policies.engine import PolicyEngine


EXAMPLE_PACK = (
    Path(__file__).resolve().parents[1]
    / "rai_toolkit"
    / "policies"
    / "packs"
    / "example_enterprise_pack"
)


def test_example_enterprise_pack_lints_and_loads_both_policies(capsys) -> None:
    exit_code = main(["policies", "lint", str(EXAMPLE_PACK)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OK: 2 policies loaded" in captured.out
    assert "high-risk-disclaimer-required" in captured.out
    assert "no-external-data-references" in captured.out
    assert captured.err == ""

    engine = PolicyEngine.from_directory(EXAMPLE_PACK)
    assert {policy.name for policy in engine.policies} == {
        "high-risk-disclaimer-required",
        "no-external-data-references",
    }


def test_example_enterprise_pack_content_triggers() -> None:
    engine = PolicyEngine.from_directory(EXAMPLE_PACK)

    assert {
        violation.policy_name
        for violation in engine.evaluate(
            [], model_output="My recommendation is to approve this request."
        )
    } == {"high-risk-disclaimer-required"}
    assert (
        engine.evaluate(
            [], model_output="My recommendation is for informational purposes."
        )
        == []
    )

    assert {
        violation.policy_name
        for violation in engine.evaluate(
            [], model_output="Please upload to an external service."
        )
    } == {"no-external-data-references"}
    assert (
        engine.evaluate(
            [], model_output="Please upload to the approved internal portal."
        )
        == []
    )

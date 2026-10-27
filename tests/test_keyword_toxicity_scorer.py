# SPDX-FileCopyrightText: 2026 Kunyang Zhang
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Offline tests for `KeywordToxicityScorer` keyword matching.

When explicitly configured, the scorer can flag a response or block it through
`GuardedModel(block_on_scorer_fail=True)`. These tests cover false hits for text that
contains none of the configured keywords as words of its own.
"""

from __future__ import annotations

import pytest

from rai_toolkit.scorers import KeywordToxicityScorer


@pytest.fixture
def scorer() -> KeywordToxicityScorer:
    return KeywordToxicityScorer()


@pytest.mark.parametrize(
    "response",
    [
        "She has strong skills in Python and SQL.",
        "Please describe the skill set required.",
        "The system reached stability after the fix.",
        "The installer is stable and well tested.",
        "An offshoot of the original project.",
        "That sentence is an oxymoron.",
        "The paper describes a bombardier beetle.",
        "Weather forecasts are usually accurate.",
    ],
)
def test_words_that_merely_contain_a_keyword_are_not_flagged(
    scorer: KeywordToxicityScorer, response: str
) -> None:
    """Regression: "skills" contains "kill", "stability" contains "stab",
    "offshoot" contains "shoot" and "oxymoron" contains "moron"."""
    result = scorer.score(response)

    assert result.details["toxic_categories"] == {}
    assert result.details["total_matches"] == 0
    assert result.score == 1.0
    assert result.passed
    assert result.assessed


@pytest.mark.parametrize(
    ("response", "category"),
    [
        ("I will kill the process.", "violence"),
        ("KILL THE PROCESS", "violence"),
        ("That is an attack on the schema.", "violence"),
        ("You are a moron.", "harassment"),
        ("It is worthLESS.", "harassment"),
        ("They called the group subhuman.", "hate_speech"),
        ("Please end your life.", "self_harm"),
        ("Do not cut yourself.", "self_harm"),
        ("A self-harm reference was removed.", "self_harm"),
    ],
)
def test_keywords_are_still_detected_as_words(
    scorer: KeywordToxicityScorer, response: str, category: str
) -> None:
    """The boundary only removes matches inside a longer Latin word."""
    result = scorer.score(response)

    assert category in result.details["toxic_categories"]
    assert not result.passed


@pytest.mark.parametrize(
    ("response", "category"),
    [
        ("You should be murdered.", "violence"),
        ("They are killing the process.", "violence"),
        ("He was stabbing the log file.", "violence"),
        ("They were bombing the cache.", "violence"),
        ("The service was attacked.", "violence"),
        ("It destroys the index.", "violence"),
        ("Bombs were mentioned.", "violence"),
        ("They hated the output.", "hate_speech"),
        ("Stop hating on the design.", "hate_speech"),
    ],
)
def test_inflected_keywords_stay_detected(
    scorer: KeywordToxicityScorer, response: str, category: str
) -> None:
    """Regression: a plain word boundary on the right also stopped "murdered",
    "killing", "stabbing" and "bombing" from matching, so the guardrail below
    allowed them through."""
    result = scorer.score(response)

    assert category in result.details["toxic_categories"]
    assert not result.passed


def test_a_trimmed_stem_on_its_own_is_not_a_hit(scorer: KeywordToxicityScorer) -> None:
    """"hate" is trimmed to "hat" to match "-ing", but "hat" alone is headwear."""
    result = scorer.score("Wear a hat.")

    assert result.details["toxic_categories"] == {}
    assert result.passed


_DEFAULT_KEYWORD_COUNT = 21
"""Keywords configured in the default categories, which is the score denominator."""

_SINGLE_HIT_SCORE = 0.5238095238095238
"""Score of exactly one canonical hit under the default configuration.

Pinned as a literal because this change must not touch the scoring formula: a base
word and every form that maps to it have to keep returning the same score.
"""

_DERIVATIVE_FORMS = (
    ("killer", "kill", "violence"),
    ("killers", "kill", "violence"),
    ("murderer", "murder", "violence"),
    ("murderers", "murder", "violence"),
    ("shooter", "shoot", "violence"),
    ("bomber", "bomb", "violence"),
    ("attacker", "attack", "violence"),
    ("hateful", "hate", "hate_speech"),
    ("stupidity", "stupid", "harassment"),
    ("idiotic", "idiot", "harassment"),
    ("moronic", "moron", "harassment"),
    ("worthlessness", "worthless", "harassment"),
    ("inferiority", "inferior", "hate_speech"),
    ("verminous", "vermin", "hate_speech"),
    ("assaultive", "assault", "violence"),
)
"""Each derivative, the canonical keyword it must report, and that keyword's category."""


def _spelling(text: str, case: str) -> str:
    """Return ``text`` in ``"lower"``, ``"upper"``, or ``"mixed"`` case.

    Mixed case alternates and starts with an uppercase letter, so ``"kill"``
    becomes ``"KiLl"``.
    """
    if case == "lower":
        return text.lower()
    if case == "upper":
        return text.upper()
    if case == "mixed":
        return "".join(
            character.upper() if index % 2 == 0 else character.lower()
            for index, character in enumerate(text)
        )
    raise ValueError(f"unknown case {case!r}")


@pytest.mark.parametrize(
    ("form", "canonical", "category"),
    _DERIVATIVE_FORMS,
)
def test_keyword_derivatives_are_detected_as_their_canonical_keyword(
    scorer: KeywordToxicityScorer, form: str, canonical: str, category: str
) -> None:
    """Regression: each of these forms was reported as `score=1.0`, `passed=True`
    and `assessed=True` with no keyword match."""
    result = scorer.score(form)

    assert result.details["toxic_categories"] == {category: [canonical]}
    assert result.details["total_matches"] == 1
    assert result.score == _SINGLE_HIT_SCORE
    assert not result.passed
    assert result.assessed


def test_a_derivative_does_not_change_the_score_denominator(
    scorer: KeywordToxicityScorer,
) -> None:
    """The derivative is an alternative of the canonical keyword, not a new keyword,
    so the 21 configured keywords - and the score of one hit - are unaffected."""
    assert sum(len(words) for words in scorer._categories.values()) == (
        _DEFAULT_KEYWORD_COUNT
    )
    assert scorer.score("kill").score == _SINGLE_HIT_SCORE
    assert scorer.score("killer").score == scorer.score("kill").score


@pytest.mark.parametrize(
    "response",
    [
        "They killed the process and the killers escaped.",
        "The shooter and the shooters left.",
        "It was an attack by an attacker.",
    ],
)
def test_a_canonical_keyword_counts_once_however_many_forms_occur(
    scorer: KeywordToxicityScorer, response: str
) -> None:
    """One category contributes one canonical keyword at most, so repeating a family
    must not push the score down like several distinct keywords would."""
    result = scorer.score(response)

    assert result.details["total_matches"] == 1
    assert result.score == _SINGLE_HIT_SCORE


def test_derivatives_of_different_keywords_count_separately(
    scorer: KeywordToxicityScorer,
) -> None:
    result = scorer.score("A killer and a murderer.")

    assert result.details["toxic_categories"] == {"violence": ["kill", "murder"]}
    assert result.details["total_matches"] == 2


def test_a_derivative_does_not_weaken_the_benign_controls(
    scorer: KeywordToxicityScorer,
) -> None:
    """The forms added here must not turn the existing benign controls into hits."""
    for response in (
        "She has strong skills in Python.",
        "The system reached stability.",
        "An offshoot of the project.",
        "That sentence is an oxymoron.",
        "The paper describes a bombardier beetle.",
    ):
        assert scorer.score(response).passed, response


@pytest.mark.parametrize("response", ["killbed", "killring", "bombted"])
def test_doubling_must_repeat_the_keywords_own_final_consonant(
    scorer: KeywordToxicityScorer, response: str
) -> None:
    """Regression: any character in `[bdglmnprt]` used to be accepted before "-ed"
    or "-ing", so these synthetic non-inflections counted as hits.

    Kept separate from the derivative tests so a passing derivative case cannot hide
    a suffix regression.
    """
    result = scorer.score(response)

    assert result.details["toxic_categories"] == {}
    assert result.details["total_matches"] == 0
    assert result.score == 1.0
    assert result.passed


@pytest.mark.parametrize(
    ("response", "canonical"),
    [
        ("stabbed", "stab"),
        ("stabbing", "stab"),
        ("bombed", "bomb"),
        ("bombing", "bomb"),
    ],
)
def test_doubled_consonant_inflections_stay_detected(
    scorer: KeywordToxicityScorer, response: str, canonical: str
) -> None:
    """The other half of the suffix correction: tightening the doubled consonant must
    not drop the forms it exists for."""
    result = scorer.score(response)

    assert result.details["toxic_categories"] == {"violence": [canonical]}
    assert result.score == _SINGLE_HIT_SCORE
    assert not result.passed


_CUSTOM_KEYWORD_SINGLE_HIT_SCORE = 0.5454545454545454
"""Score of one hit for a scorer with the 21 default keywords plus one custom keyword.

The custom keyword adds exactly one entry to the denominator, which is what keeps the
per-keyword calibration intact.
"""

_CASED_DERIVATIVE_SCORE = 0.09090909090909094
"""Score of two hits with 22 configured keywords: ``1 - 2 / (22 * 0.1)``.

A derivative of a default keyword also hits that default keyword, so configuring
the same word again contributes a second match. ``threshold=0.5`` sits between
this and the one-match score above, which is where a case-sensitive derivative
lookup changed the verdict.
"""

_CASE_VERDICT_THRESHOLD = 0.5


@pytest.mark.parametrize(
    ("keyword", "response"),
    [
        ("nab", "nabbed"),
        ("NAB", "nabbed"),
        ("nAb", "nabbed"),
        ("nab", "NABBED"),
        ("NAB", "Nabbed"),
        ("nAb", "naBBed"),
        ("grab", "grabbing"),
        ("GRAB", "grabbing"),
        ("GrAb", "GrAbBing"),
    ],
)
def test_a_custom_keyword_is_matched_case_insensitively_in_its_own_spelling(
    keyword: str, response: str
) -> None:
    """Regression: the doubling test compared the keyword's final character against a
    lowercase set, so an uppercase keyword never got a doubled alternative at all -
    `re.IGNORECASE` cannot recover an alternative that was never built. A configured
    "NAB" missed "nabbed" entirely while "nab" matched it."""
    scorer = KeywordToxicityScorer(extra_keywords={"custom": [keyword]})

    result = scorer.score(response)

    assert result.details["toxic_categories"] == {"custom": [keyword]}
    assert result.details["total_matches"] == 1
    assert result.score == _CUSTOM_KEYWORD_SINGLE_HIT_SCORE
    assert result.assessed
    assert not result.passed


def test_case_variants_of_a_custom_keyword_score_identically() -> None:
    """The casing of the configured keyword or of the response must not move the score:
    the evidence keeps the caller's spelling and the denominator counts one keyword."""
    scores = {
        keyword: KeywordToxicityScorer(extra_keywords={"custom": [keyword]})
        .score("nabbed")
        .score
        for keyword in ("nab", "NAB", "nAb")
    }

    assert set(scores.values()) == {_CUSTOM_KEYWORD_SINGLE_HIT_SCORE}


@pytest.mark.parametrize(
    ("keyword", "response"),
    [
        ("snare", "snaring"),
        ("SNARE", "snaring"),
        ("Snare", "snaring"),
        ("SNARE", "SNARING"),
    ],
)
def test_an_e_dropping_custom_keyword_keeps_its_form_case_insensitively(
    keyword: str, response: str
) -> None:
    """Pre-existing on `main`: the e-dropping branch tested `keyword.endswith("e")`, so
    it was skipped for an uppercase keyword and "SNARE" missed "snaring" the same way."""
    scorer = KeywordToxicityScorer(extra_keywords={"custom": [keyword]})

    result = scorer.score(response)

    assert result.details["toxic_categories"] == {"custom": [keyword]}
    assert result.score == _CUSTOM_KEYWORD_SINGLE_HIT_SCORE
    assert not result.passed


@pytest.mark.parametrize("response", ["KILLBED", "KillRing", "BOMBTED"])
def test_case_variants_of_a_synthetic_doubling_are_still_not_hits(
    scorer: KeywordToxicityScorer, response: str
) -> None:
    """Normalizing the doubling check must not reopen the synthetic non-inflections."""
    result = scorer.score(response)

    assert result.details["toxic_categories"] == {}
    assert result.score == 1.0
    assert result.passed


@pytest.mark.parametrize(("form", "canonical", "category"), _DERIVATIVE_FORMS)
@pytest.mark.parametrize("keyword_case", ["lower", "upper", "mixed"])
@pytest.mark.parametrize("response_case", ["lower", "upper", "mixed"])
def test_derivative_lookup_keeps_the_same_public_result_across_casings(
    form: str, canonical: str, category: str, keyword_case: str, response_case: str
) -> None:
    """Regression: ``_KEYWORD_DERIVATIVES.get(keyword)`` used the configured spelling.

    The map keys are lowercase, so ``"KILL"`` and ``"KiLl"`` missed every derivative
    that ``"kill"`` matched. At ``threshold=0.5`` that was a different score, match
    count, and verdict. The reported keyword stays the caller's spelling.
    """
    keyword = _spelling(canonical, keyword_case)
    response = _spelling(form, response_case)
    result = KeywordToxicityScorer(
        extra_keywords={"custom": [keyword]},
        threshold=_CASE_VERDICT_THRESHOLD,
    ).score(response)

    assert result.details["toxic_categories"] == {
        category: [canonical],
        "custom": [keyword],
    }
    assert result.details["total_matches"] == 2
    assert result.score == _CASED_DERIVATIVE_SCORE
    assert result.assessed
    assert not result.passed


def test_review_reproduction_keyword_and_response_casings_score_identically() -> None:
    """The public-scorer reproduction: ``kill`` / ``KILL`` / ``KiLl`` on "killer".

    Lowercase, uppercase, and mixed-case keywords and responses must share one
    score, one match count, and one verdict, while each result reports the
    configured spelling rather than a normalized one.
    """
    seen: list[tuple[float, bool, int]] = []
    for keyword_case in ("lower", "upper", "mixed"):
        keyword = _spelling("kill", keyword_case)
        for response_case in ("lower", "upper", "mixed"):
            result = KeywordToxicityScorer(
                extra_keywords={"custom": [keyword]},
                threshold=_CASE_VERDICT_THRESHOLD,
            ).score(_spelling("killer", response_case))

            assert result.details["toxic_categories"] == {
                "violence": ["kill"],
                "custom": [keyword],
            }
            seen.append(
                (result.score, result.passed, result.details["total_matches"])
            )

    assert seen == [(_CASED_DERIVATIVE_SCORE, False, 2)] * len(seen)


async def test_guardrail_blocks_a_derivative_threat() -> None:
    """The derivative mapping has to reach an explicitly configured blocking
    guardrail, not only direct scoring."""
    from rai_toolkit.guardrails import GuardedModel
    from rai_toolkit.models import CallableModel

    guarded = GuardedModel(
        model=CallableModel(
            lambda text, context="", **kwargs: "The killer was identified."
        ),
        output_scorers=[KeywordToxicityScorer()],
        block_on_scorer_fail=True,
    )

    response = await guarded.predict("a harmless prompt")

    assert response.metadata["blocked"] is True
    assert response.metadata["blocked_by"] == "scorer:KeywordToxicityScorer"
    assert response.output == guarded.blocked_response


async def test_guardrail_blocks_an_inflected_threat() -> None:
    """An explicitly configured blocking guardrail must reject an inflected threat
    detected by its keyword scorer."""
    from rai_toolkit.guardrails import GuardedModel
    from rai_toolkit.models import CallableModel

    guarded = GuardedModel(
        model=CallableModel(lambda text, context="", **kwargs: "You should be murdered."),
        output_scorers=[KeywordToxicityScorer()],
        block_on_scorer_fail=True,
    )

    response = await guarded.predict("a harmless prompt")

    assert response.metadata["blocked"] is True
    assert response.metadata["blocked_by"] == "scorer:KeywordToxicityScorer"
    assert response.output == guarded.blocked_response


async def test_guardrail_allows_an_ordinary_answer() -> None:
    """The other half of the same guardrail: a word that merely contains a keyword
    must not block a response."""
    from rai_toolkit.guardrails import GuardedModel
    from rai_toolkit.models import CallableModel

    guarded = GuardedModel(
        model=CallableModel(
            lambda text, context="", **kwargs: "She has strong skills in Python."
        ),
        output_scorers=[KeywordToxicityScorer()],
        block_on_scorer_fail=True,
    )

    response = await guarded.predict("a harmless prompt")

    assert response.metadata["blocked"] is False
    assert response.output == "She has strong skills in Python."


def test_matched_keyword_is_reported_in_its_configured_form(
    scorer: KeywordToxicityScorer,
) -> None:
    result = scorer.score("KILL THE PROCESS")

    assert result.details["toxic_categories"] == {"violence": ["kill"]}


def test_extra_keywords_apply_to_the_instance_that_declares_them() -> None:
    scorer = KeywordToxicityScorer(extra_keywords={"violence": ["zap"]})

    result = scorer.score("The tool will zap the record.")

    assert result.details["toxic_categories"] == {"violence": ["zap"]}
    assert not result.passed


def test_extra_keywords_can_introduce_a_new_category() -> None:
    scorer = KeywordToxicityScorer(extra_keywords={"profanity": ["darn"]})

    result = scorer.score("Well, darn.")

    assert result.details["toxic_categories"] == {"profanity": ["darn"]}


def test_extra_keywords_do_not_leak_into_other_instances() -> None:
    """Regression: a shallow copy of the category mapping extended the lists held
    by the class attribute, so a custom keyword reached every later scorer."""
    KeywordToxicityScorer(extra_keywords={"violence": ["zap"]})

    assert "zap" not in KeywordToxicityScorer.TOXIC_CATEGORIES["violence"]

    default_scorer = KeywordToxicityScorer()
    assert "zap" not in default_scorer._categories["violence"]
    assert default_scorer.score("The tool will zap the record.").details[
        "toxic_categories"
    ] == {}


def test_extra_keywords_do_not_leak_between_two_configured_scorers() -> None:
    first = KeywordToxicityScorer(extra_keywords={"violence": ["zap"]})
    second = KeywordToxicityScorer(extra_keywords={"violence": ["squash"]})

    assert first.score("squash").details["toxic_categories"] == {}
    assert second.score("zap").details["toxic_categories"] == {}


def test_keywords_outside_latin_script_still_match() -> None:
    """Word boundaries built from `\\b` would never match these, because
    consecutive characters of such scripts are word characters on both sides."""
    scorer = KeywordToxicityScorer(extra_keywords={"violence": ["暴力"]})

    result = scorer.score("这是一段暴力内容。")

    assert result.details["toxic_categories"] == {"violence": ["暴力"]}


def test_response_without_any_keyword_scores_a_full_pass(
    scorer: KeywordToxicityScorer,
) -> None:
    result = scorer.score("The migration finished without incidents.")

    assert result.score == 1.0
    assert result.passed
    assert result.explanation == "No toxic keywords detected"
    assert result.category == "MIT-1.2"

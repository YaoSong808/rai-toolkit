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
"""Keywords in the shipped ``TOXIC_CATEGORIES``, which is the normalization basis."""

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
    so the 21-keyword basis - and the score of one hit - are unaffected."""
    assert sum(len(words) for words in scorer._categories.values()) == (
        _DEFAULT_KEYWORD_COUNT
    )
    assert scorer.normalization_basis() == _DEFAULT_KEYWORD_COUNT
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


_CUSTOM_KEYWORD_SINGLE_HIT_SCORE = 0.5238095238095238
"""Score of one hit from a configured custom keyword.

The custom keyword is matched and reported, but it does not enter the basis.
The score is therefore the canonical one-hit score, not ``1 - 1 / (22 * 0.1)``
(``0.5454545454545454``).
"""

_CASED_DERIVATIVE_SCORE = 0.04761904761904767
"""Score of two hits on the canonical basis of 21: ``1 - 2 / (21 * 0.1)``.

A derivative of a default keyword also hits that default keyword, so configuring
the same word again contributes a second match. The extra keyword does not
change the basis. ``threshold=0.5`` sits between this and the one-match score,
which is where a case-sensitive derivative lookup changed the verdict.
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
    """The casing of the configured keyword or of the response must not move the score.

    Evidence keeps the caller's spelling, and each spelling scores as one hit on
    the canonical basis.
    """
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


_TWO_HIT_SCORE = 0.04761904761904767
"""Exact two-hit score at basis 21: ``1 - 2 / (21 * 0.1)``.

This is the float the unchanged formula returns. The acceptance table prints
the same quantity as ``0.04761904761904745``.
"""

_THREE_HIT_SCORE = 0.0
"""Three hits clamp: ``1 - 3 / (21 * 0.1)`` is negative, so the score is 0.0."""

_HIT_RESPONSES = {
    0: "The migration finished without incidents.",
    1: "They are killing civilians.",
    2: "They are killing and murdering civilians.",
    3: "They are killing, murdering, and attacking civilians.",
}
"""Responses with exactly 0, 1, 2, or 3 canonical keyword hits."""


def _catalog_snapshot(catalog: dict[str, list[str]]) -> dict[str, list[str]]:
    return {category: list(keywords) for category, keywords in catalog.items()}


def test_default_configuration_scores_stay_exact() -> None:
    """0, 1, 2, and 3 canonical hits keep the scores of the shipped catalog."""
    zero = KeywordToxicityScorer().score(_HIT_RESPONSES[0])
    one = KeywordToxicityScorer().score(_HIT_RESPONSES[1])
    two = KeywordToxicityScorer().score(_HIT_RESPONSES[2])
    three = KeywordToxicityScorer().score(_HIT_RESPONSES[3])

    assert zero.score == 1.0
    assert zero.passed
    assert zero.details == {"toxic_categories": {}, "total_matches": 0}

    assert one.score == _SINGLE_HIT_SCORE
    assert not one.passed
    assert one.details == {
        "toxic_categories": {"violence": ["kill"]},
        "total_matches": 1,
    }

    assert two.score == _TWO_HIT_SCORE
    assert not two.passed
    assert two.details == {
        "toxic_categories": {"violence": ["kill", "murder"]},
        "total_matches": 2,
    }

    assert three.score == _THREE_HIT_SCORE
    assert not three.passed
    assert three.details == {
        "toxic_categories": {"violence": ["kill", "murder", "attack"]},
        "total_matches": 3,
    }


@pytest.mark.parametrize(
    ("threshold", "hits", "expected_score", "expected_passed"),
    [
        (0.9, 0, 1.0, True),
        (0.9, 1, _SINGLE_HIT_SCORE, False),
        (0.9, 2, _TWO_HIT_SCORE, False),
        (0.6, 0, 1.0, True),
        (0.6, 1, _SINGLE_HIT_SCORE, False),
        (0.6, 2, _TWO_HIT_SCORE, False),
        (0.5, 0, 1.0, True),
        (0.5, 1, _SINGLE_HIT_SCORE, True),
        (0.5, 2, _TWO_HIT_SCORE, False),
    ],
)
def test_threshold_still_decides_the_verdict(
    threshold: float, hits: int, expected_score: float, expected_passed: bool
) -> None:
    """Non-default thresholds keep the same scores; only ``passed`` moves."""
    result = KeywordToxicityScorer(threshold=threshold).score(_HIT_RESPONSES[hits])

    assert result.score == expected_score
    assert result.passed is expected_passed


@pytest.mark.parametrize(
    "extra_keywords",
    [
        pytest.param(
            {"violence": [f"zzz{index}" for index in range(79)]},
            id="matched-category",
        ),
        pytest.param(
            {"unused": [f"zzz{index}" for index in range(50)]},
            id="new-category",
        ),
    ],
)
def test_unmatched_keywords_leave_a_hit_result_unchanged(
    extra_keywords: dict[str, list[str]],
) -> None:
    """Adding keywords that do not occur leaves score, verdict, and evidence."""
    response = _HIT_RESPONSES[1]
    before = KeywordToxicityScorer().score(response)
    after = KeywordToxicityScorer(extra_keywords=extra_keywords).score(response)

    assert after == before
    assert after.score == _SINGLE_HIT_SCORE
    assert after.passed is False
    assert after.details == before.details
    assert after.score != 0.9


def test_unmatched_keywords_leave_a_clean_result_unchanged() -> None:
    response = _HIT_RESPONSES[0]
    before = KeywordToxicityScorer().score(response)
    after = KeywordToxicityScorer(
        extra_keywords={"unused": [f"zzz{index}" for index in range(99)]}
    ).score(response)

    assert after == before
    assert after.score == 1.0
    assert after.passed is True
    assert after.details == {"toxic_categories": {}, "total_matches": 0}


def test_a_matched_custom_keyword_scores_as_one_canonical_hit() -> None:
    """A custom hit used to score ``0.5454545454545454`` because the extra
    keyword enlarged the denominator. It now scores as one canonical hit, and
    the evidence is still the literal configured keyword.
    """
    scorer = KeywordToxicityScorer(extra_keywords={"custom": ["darn"]})

    result = scorer.score("Well, darn.")

    assert result.details == {
        "toxic_categories": {"custom": ["darn"]},
        "total_matches": 1,
    }
    assert result.score == _SINGLE_HIT_SCORE
    assert result.score != 0.5454545454545454
    assert not result.passed
    assert scorer.normalization_basis() == _DEFAULT_KEYWORD_COUNT


def test_derivative_aliases_never_enter_the_normalization_basis() -> None:
    from rai_toolkit.scorers.programmatic import _KEYWORD_DERIVATIVES

    catalog = {
        word
        for words in KeywordToxicityScorer.TOXIC_CATEGORIES.values()
        for word in words
    }
    aliases = {
        alias
        for forms in _KEYWORD_DERIVATIVES.values()
        for alias in forms
    }
    assert aliases
    assert aliases.isdisjoint(catalog)
    assert KeywordToxicityScorer.normalization_basis() == len(catalog) == (
        _DEFAULT_KEYWORD_COUNT
    )

    result = KeywordToxicityScorer().score("killer")

    assert result.details == {
        "toxic_categories": {"violence": ["kill"]},
        "total_matches": 1,
    }
    assert result.score == _SINGLE_HIT_SCORE


class _WideToxicityCatalog(KeywordToxicityScorer):
    """One hundred canonical keywords, so one hit is ``1 - 1/10``."""

    TOXIC_CATEGORIES = {
        "violence": ["kill", *[f"canon{index}" for index in range(99)]],
    }


class _NarrowToxicityCatalog(KeywordToxicityScorer):
    TOXIC_CATEGORIES = {"violence": ["kill", "murder"]}


class _InheritedToxicityCatalog(KeywordToxicityScorer):
    """No catalog override. The basis stays the base catalog's 21."""


def test_subclass_basis_follows_its_own_canonical_catalog() -> None:
    """A basis frozen on the base class would still be 21 for every override."""
    assert KeywordToxicityScorer.normalization_basis() == _DEFAULT_KEYWORD_COUNT
    assert _InheritedToxicityCatalog.normalization_basis() == _DEFAULT_KEYWORD_COUNT
    assert _WideToxicityCatalog.normalization_basis() == 100
    assert _NarrowToxicityCatalog.normalization_basis() == 2

    wide = _WideToxicityCatalog().score(_HIT_RESPONSES[1])
    # 1 - 1/10 = 0.9, which passes the default threshold. Basis 21 would fail.
    assert wide.score == 0.9
    assert wide.passed is True
    assert wide.score != _SINGLE_HIT_SCORE

    narrow = _NarrowToxicityCatalog().score(_HIT_RESPONSES[1])
    # basis 2 clamps the denominator to 1, so one hit scores 0.0.
    assert narrow.score == 0.0
    assert narrow.passed is False
    assert narrow.details == {
        "toxic_categories": {"violence": ["kill"]},
        "total_matches": 1,
    }


def test_extra_keywords_cannot_mutate_catalog_or_basis() -> None:
    base_catalog = _catalog_snapshot(KeywordToxicityScorer.TOXIC_CATEGORIES)
    wide_catalog = _catalog_snapshot(_WideToxicityCatalog.TOXIC_CATEGORIES)
    extras = {
        "violence": [f"zzz{index}" for index in range(79)],
        "unused": [f"qqq{index}" for index in range(50)],
    }

    base = KeywordToxicityScorer(extra_keywords=extras)
    wide = _WideToxicityCatalog(extra_keywords=extras)

    assert _catalog_snapshot(KeywordToxicityScorer.TOXIC_CATEGORIES) == base_catalog
    assert _catalog_snapshot(_WideToxicityCatalog.TOXIC_CATEGORIES) == wide_catalog
    assert base.normalization_basis() == _DEFAULT_KEYWORD_COUNT
    assert KeywordToxicityScorer.normalization_basis() == _DEFAULT_KEYWORD_COUNT
    assert wide.normalization_basis() == 100
    assert _WideToxicityCatalog.normalization_basis() == 100

    response = _HIT_RESPONSES[1]
    assert base.score(response) == KeywordToxicityScorer().score(response)
    assert wide.score(response) == _WideToxicityCatalog().score(response)
    assert wide.score(response).score == 0.9


async def test_guardrail_unmatched_keywords_cannot_dilute_a_block() -> None:
    """The dilution reproduction: 79 keywords that never occur used to raise one
    ``kill`` hit from a failing score to ``0.9`` (pass), so a blocking
    guardrail let the response through. The block has to stand.
    """
    from rai_toolkit.guardrails import GuardedModel
    from rai_toolkit.models import CallableModel

    response_text = "They are killing civilians."
    extra = {"unused": [f"zzz{index}" for index in range(79)]}

    def _guarded(scorer: KeywordToxicityScorer) -> GuardedModel:
        return GuardedModel(
            model=CallableModel(lambda text, context="", **kwargs: response_text),
            output_scorers=[scorer],
            block_on_scorer_fail=True,
        )

    plain_guard = _guarded(KeywordToxicityScorer())
    diluted_guard = _guarded(KeywordToxicityScorer(extra_keywords=extra))
    plain = await plain_guard.predict("a harmless prompt")
    diluted = await diluted_guard.predict("a harmless prompt")

    assert plain.metadata["blocked"] is True
    assert plain.output == plain_guard.blocked_response
    assert diluted.metadata["blocked"] is True
    assert diluted.metadata["blocked_by"] == "scorer:KeywordToxicityScorer"
    assert diluted.output == diluted_guard.blocked_response
    assert diluted.metadata["scorer_results"]["KeywordToxicityScorer"] == {
        "score": _SINGLE_HIT_SCORE,
        "passed": False,
    }

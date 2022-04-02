import copy
import re

from hypothesis import given
from hypothesis.strategies import booleans, text

from seq2rel.common import util


@given(text=text(), lowercase=booleans())
def test_sanitize_text(text: str, lowercase: bool) -> None:
    sanitized = util.sanitize_text(text, lowercase=lowercase)

    # There should be no cases of multiple spaces or tabs
    assert re.search(r"[ ]{2,}", sanitized) is None
    assert "\t" not in sanitized
    # The beginning and end of the string should be stripped of whitespace
    assert not sanitized.startswith(("\n", " "))
    assert not sanitized.endswith(("\n", " "))
    # Sometimes, hypothesis generates text that cannot be lowercased (like latin characters).
    # We don't particularly care about this, and it breaks this check.
    # Only run if the generated text can be lowercased.
    if lowercase and text.lower().islower():
        assert all(not char.isupper() for char in sanitized)


def test_extract_relations() -> None:
    linearizations = [
        # Empty string
        "",
        # Non-empty string with no relation
        "I don't contain anything of interest!",
        # Non-empty string with no relation
        "fenoprofen @DRUG@ @ADE@",
        # A valid string with one relation
        "fenoprofen @DRUG@ pure red cell aplasia @EFFECT@ @ADE@",
        # A valid string with multiple relations
        (
            "bimatoprost @DRUG@ cystoid macula edema @EFFECT@ @ADE@"
            # A duplicate relation that should not be included in the deserialized annotation
            " bimatoprost @DRUG@ cystoid macula edema @EFFECT@ @ADE@"
            " latanoprost @DRUG@ cystoid macula edema @EFFECT@ @ADE@"
            # A duplicate mention that should not be included in the deserialized annotation
            " methamphetamine ; meth ; meth @CHEMICAL@ psychosis ; psychotic disorders @DISEASE@ @CID@"
        ),
        # A valid string with multiple relations and non-alpha-numeric characters in the special tokens
        (
            "pasay city @LOC@ metro manila @LOC@ @LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY@"
            # A duplicate entity that should only be retained if remove_duplicate_ents is False
            " pasay city @LOC@ pasay city @LOC@ @LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY@"
        ),
    ]

    # Check that we can call the function on a list of strings
    expected = [
        {},
        {},
        {},
        {"ADE": [((("fenoprofen",), "DRUG"), (("pure red cell aplasia",), "EFFECT"))]},
        {
            "ADE": [
                ((("bimatoprost",), "DRUG"), (("cystoid macula edema",), "EFFECT")),
                ((("latanoprost",), "DRUG"), (("cystoid macula edema",), "EFFECT")),
            ],
            "CID": [
                (
                    (("methamphetamine", "meth"), "CHEMICAL"),
                    (("psychotic disorders", "psychosis"), "DISEASE"),
                )
            ],
        },
        {
            "LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY": [
                ((("pasay city",), "LOC"), (("metro manila",), "LOC")),
                ((("pasay city",), "LOC"), (("pasay city",), "LOC")),
            ],
        },
    ]
    # Set `ordered_ents=True` so that mentions aren't sorted (easier to write test cases).
    actual = util.extract_relations(linearizations, ordered_ents=True)
    assert expected == actual

    # Check that a relation with duplicate entities is removed when `remove_duplicate_ents` is True.
    deduplicated_expected = copy.deepcopy(expected)
    del deduplicated_expected[-1]["LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY"][-1]  # type: ignore
    actual = util.extract_relations(linearizations, ordered_ents=True, remove_duplicate_ents=True)
    assert deduplicated_expected == actual

    # Check that a relation provided via `filtered_relations` are removed from the output.
    filtered_expected = copy.deepcopy(expected)
    filtered_relations = [""] * len(linearizations)
    filtered_relations[3] = "fenoprofen @DRUG@ pure red cell aplasia @EFFECT@ @ADE@"
    del filtered_expected[3]["ADE"][-1]  # type: ignore
    actual = util.extract_relations(
        linearizations, ordered_ents=True, filtered_relations=filtered_relations
    )
    assert filtered_expected == actual


def test_extract_entities() -> None:
    linearization = (
        # Duplicate coreferent mentions + case insensitivity
        "methamphetamine ; Meth ; meth @CHEMICAL@"
        # Duplicate entity + case insensitivity + order insensitivity
        " psychosis ; Psychotic disorders @DISEASE@"
        " psychotic disorders ; psychosis @DISEASE@"
        " @CID@"
    )
    actual = util.extract_entities(linearization, remove_duplicate_ents=False)
    expected: util.EntityAnnotation = (
        (("methamphetamine", "meth"), "CHEMICAL"),
        (("psychotic disorders", "psychosis"), "DISEASE"),
        # The duplicate entity is kept because remove_duplicate_ents is False.
        (("psychotic disorders", "psychosis"), "DISEASE"),
    )
    assert actual == expected

    actual = util.extract_entities(linearization, remove_duplicate_ents=True)
    expected = (
        (("methamphetamine", "meth"), "CHEMICAL"),
        # The duplicate entity is removed because remove_duplicate_ents is True.
        (("psychotic disorders", "psychosis"), "DISEASE"),
    )
    assert actual == expected

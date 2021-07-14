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


def test_deserialize_annotation() -> None:
    # Test:
    # - the empty string
    # - non-empty string with no relation
    # - non-empty string with an invalid relation
    # - a single relation string
    # - a multiple relation string
    # - a more complicated relation type with non-alpha-numeric characters in the special tokens
    # - duplicate entity mentions and duplicate coreferent mentions
    serialized_annotations = [
        "",
        "I don't contain anything of interest!",
        "@ADE@ fenoprofen @DRUG@ @EOR@",
        "@ADE@ fenoprofen @DRUG@ pure red cell aplasia @EFFECT@ @EOR@",
        (
            "@ADE@ bimatoprost @DRUG@ cystoid macula edema @EFFECT@ @EOR@"
            # A duplicate relation that should not be included in the deserialized annotation
            " @ADE@ bimatoprost @DRUG@ cystoid macula edema @EFFECT@ @EOR@"
            " @ADE@ latanoprost @DRUG@ cystoid macula edema @EFFECT@ @EOR@"
            # A duplicate mention that should not be included in the deserialized annotation
            " @CID@ methamphetamine ; meth ; meth @CHEMICAL@ psychosis ; psychotic disorders @DISEASE@ @EOR@"
        ),
        "@LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY@ pasay city @LOC@ metro manila @LOC@ @EOR@",
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
                ((("pasay city",), "LOC"), (("metro manila",), "LOC"))
            ],
        },
    ]
    actual = util.deserialize_annotations(serialized_annotations)
    assert expected == actual

    # Check that we can call the function on a single string
    actual = util.deserialize_annotations(serialized_annotations[-1])
    assert [expected[-1]] == actual


def test_normalize_clusters() -> None:
    clusters = (
        # Duplicate coreferent mentions + case insensitivity
        ("methamphetamine ; Meth ; meth", "CHEMICAL"),
        # Duplicate clusters + case insensitivity + order insensitivity
        ("psychosis ; Psychotic disorders", "DISEASE"),
        ("psychotic disorders ; psychosis", "DISEASE"),
    )
    actual = util._normalize_clusters(clusters)
    expected = (
        (("methamphetamine", "meth"), "CHEMICAL"),
        (("psychotic disorders", "psychosis"), "DISEASE"),
        # The duplicate cluster is kept because some relations do contain repeated
        # clusters (like homodimers in protein-protein interactions).
        (("psychotic disorders", "psychosis"), "DISEASE"),
    )
    assert actual == expected

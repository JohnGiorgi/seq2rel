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
    # - a single relation string
    # - a multiple relation string
    serialized_annotations = [
        "",
        "I don't contain anything of interest!",
        "@ADE@ fenoprofen @DRUG@ pure red cell aplasia @EFFECT@ @EOR@",
        (
            "@ADE@ bimatoprost @DRUG@ cystoid macula edema @EFFECT@ @EOR@"
            " @ADE@ latanoprost @DRUG@ cystoid macula edema @EFFECT@ @EOR@"
            " @CID@ methamphetamine; meth @CHEMICAL@ psychosis; psychotic disorders @DISEASE@ @EOR@"
        ),
    ]

    # Check that we can call the function of a list of strings
    expected = [
        {},
        {},
        {"ADE": [(("fenoprofen", "DRUG"), ("pure red cell aplasia", "EFFECT"))]},
        {
            "ADE": [
                (("bimatoprost", "DRUG"), ("cystoid macula edema", "EFFECT")),
                (("latanoprost", "DRUG"), ("cystoid macula edema", "EFFECT")),
            ],
            "CID": [
                (
                    ("methamphetamine; meth", "CHEMICAL"),
                    ("psychosis; psychotic disorders", "DISEASE"),
                )
            ],
        },
    ]
    actual = util.deserialize_annotations(serialized_annotations)
    assert expected == actual

    # Check that we can call the function on a single string
    actual = util.deserialize_annotations(serialized_annotations[-1])
    assert [expected[-1]] == actual

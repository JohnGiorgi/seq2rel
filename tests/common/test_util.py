import re

from hypothesis import given
from hypothesis.strategies import booleans, text

from seq2rel.common.util import sanitize_text


class TestUtil:
    @given(text=text(), lowercase=booleans())
    def test_sanitize_text(self, text: str, lowercase: bool) -> None:
        sanitized = sanitize_text(text, lowercase=lowercase)

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

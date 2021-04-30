import copy

from seq2rel.common.util import END_OF_REL_SYMBOL, SPECIAL_TARGET_TOKENS
from seq2rel.data.tokenizers.pretrained_transformer_tokenizer_seq2rel import (
    PretrainedTransformerTokenizerSeq2Rel,
)


class TestPretrainedTransformerTokenizerSeq2Rel:
    def test_special_tokens(self) -> None:
        """This tests asserts that none of our special tokens are broken up."""
        tokenizer = PretrainedTransformerTokenizerSeq2Rel(
            model_name="bert-base-cased", add_special_tokens=False
        )
        expected_tokens = copy.deepcopy(SPECIAL_TARGET_TOKENS)
        sentence = " ".join(expected_tokens)
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_additional_special_tokens(self) -> None:
        """This tests asserts that we can add additional special tokens that aren't broken up."""
        test_special_token = "@SPECIAL@"
        tokenizer_kwargs = {
            "special_tokens": [test_special_token],
            "additional_special_tokens": [test_special_token],
        }
        tokenizer = PretrainedTransformerTokenizerSeq2Rel(
            model_name="bert-base-cased", tokenizer_kwargs=tokenizer_kwargs
        )

        sentence = (
            f"I contain a default special token {END_OF_REL_SYMBOL}"
            f" and an additional one {test_special_token}."
        )
        expected_tokens = [
            "[CLS]",
            "I",
            "contain",
            "a",
            "default",
            "special",
            "token",
            f"{END_OF_REL_SYMBOL}",
            "and",
            "an",
            "additional",
            "one",
            f"{test_special_token}",
            ".",
            "[SEP]",
        ]
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

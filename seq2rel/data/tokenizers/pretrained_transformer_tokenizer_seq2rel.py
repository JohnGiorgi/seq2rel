import copy
import logging
from typing import Any, Dict, Optional

from allennlp.data.tokenizers import PretrainedTransformerTokenizer, Tokenizer
from seq2rel.common.util import SPECIAL_TARGET_TOKENS

logger = logging.getLogger(__name__)


@Tokenizer.register("pretrained_transformer_seq2rel")
class PretrainedTransformerTokenizerSeq2Rel(PretrainedTransformerTokenizer):
    """
    This is a thin wrapper around `PretrainedTransformerTokenizer` to provide any of the
    modifications necessary to use the model for information extraction. The arguments are
    identical to `PretrainedTransformerTokenizer`. For details, please see:
    [`PretrainedTransformerTokenizer`](https://github.com/allenai/allennlp/blob/main/allennlp/data/tokenizers/pretrained_transformer_tokenizer.py),

    Registered as a `Tokenizer` with name "pretrained_transformer_seq2rel".
    """

    def __init__(self, tokenizer_kwargs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        # Here, we add any special tokens used during decoding. The purpose is to prevent the
        # target decoder from splitting these tokens up into multiple word pieces.
        # This is done here, to prevent the user from having to provide these in the config.
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        else:
            tokenizer_kwargs = tokenizer_kwargs.copy()
        special_tokens = copy.deepcopy(SPECIAL_TARGET_TOKENS)
        # Careful not to overwrite special tokens provided on instantiation
        special_tokens.extend(tokenizer_kwargs.get("special_tokens", []))
        special_tokens.extend(tokenizer_kwargs.get("additional_special_tokens", []))
        # HF tokenizers name this parameter one of two things, including both here.
        tokenizer_kwargs = {
            "special_tokens": special_tokens,
            "additional_special_tokens": special_tokens,
        }
        super().__init__(tokenizer_kwargs=tokenizer_kwargs, **kwargs)

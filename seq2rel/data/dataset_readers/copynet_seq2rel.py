import logging
from typing import Dict

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp_models.generation.dataset_readers import CopyNetDatasetReader
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("copynet_seq2rel")
class CopyNetSeq2RelDatasetReader(CopyNetDatasetReader):
    """
    This is a thin wrapper around `CopyNetDatasetReader` to provide any of the modifications
    necessary to use the dataset reader for information extraction. The arguments are identical to
    `CopyNetSeq2Seq`. For details, please see:
    [`CopyNetSeq2Seq`](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/generation/dataset_readers/copynet_seq2seq.py),

    # Parameters

    target_namespace : `str`, required
        The vocab namespace for the targets. This needs to be passed to the dataset reader
        in order to construct the NamespaceSwappingField.
    source_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to `SpacyTokenizer()`.
    target_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to `source_tokenizer`.
    source_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define input (source side) token representations. Defaults to
        `{"tokens": SingleIdTokenIndexer()}`.
    """

    def __init__(
        self,
        target_namespace: str,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            target_namespace, source_tokenizer, target_tokenizer, source_token_indexers, **kwargs
        )

    @overrides
    def text_to_instance(
        self, source_string: str, target_string: str = None
    ) -> Instance:  # type: ignore
        # We have to add a space in front of the source/target strings in order to achieve
        # consistant tokenization with certain tokenizers, like GPT-2. Enforce this behavior here.
        # See: https://github.com/huggingface/transformers/issues/1196
        source_string = " " + source_string.lstrip()
        if target_string:
            target_string = " " + target_string.lstrip()
        return super().text_to_instance(source_string, target_string)

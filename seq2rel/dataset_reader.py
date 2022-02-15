import logging
from math import floor
from typing import Optional

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp_models.generation.dataset_readers import CopyNetDatasetReader

logger = logging.getLogger(__name__)


@DatasetReader.register("seq2rel")
class Seq2RelDatasetReader(CopyNetDatasetReader):
    """
    This is a thin wrapper around `CopyNetDatasetReader` to provide any of the modifications
    necessary to use the dataset reader for information extraction. For details, please see:
    [`CopyNetSeq2Seq`](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/generation/dataset_readers/copynet_seq2seq.py).

    # Parameters

    max_length : `int`, optional (default = None)
        The maximum length of the source sequences. If not None, the source sequences will be
        truncated to this length by concatenating the first 75% and last 25% `max_length` tokens.
        This is known as head+tail truncation. See https://arxiv.org/abs/1905.05583 for details.
        Note: This only works correctly if the source tokenizer is a `PretrainedTransformerTokenizer`.
    """

    def __init__(self, max_length: Optional[int] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_length = max_length

    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in self.shard_iterable(enumerate(data_file)):
                line = line.strip("\n")
                if not line:
                    continue
                line_parts = line.split("\t")
                if len(line_parts) == 2:
                    source_sequence, target_sequence, filtered_relations = *line_parts, None  # type: ignore
                elif len(line_parts) == 3:
                    source_sequence, target_sequence, filtered_relations = line_parts
                else:
                    raise RuntimeError(
                        "Invalid line format: %s (line number %d)" % (line, line_num + 1)
                    )
                if not source_sequence:
                    continue
                yield self.text_to_instance(source_sequence, target_sequence, filtered_relations)

    def text_to_instance(
        self,
        source_string: str,
        target_string: str = None,
        filtered_relations: str = None,
        weight: float = None,
    ) -> Instance:  # type: ignore

        if self._max_length is not None:
            if not isinstance(self._source_tokenizer, PretrainedTransformerTokenizer):
                raise ValueError(
                    "max_length was provided to Seq2RelDatasetReader. source_tokenizer must be a"
                    " PretrainedTransformerTokenizer."
                )
            source_string = self._head_tail_truncation(source_string)
        # We have to add a space in front of the source/target strings in order to achieve
        # consistant tokenization with certain tokenizers, like GPT. Enforce this behavior here.
        # See: https://github.com/huggingface/transformers/issues/1196
        source_string = " " + source_string.lstrip()
        if target_string:
            target_string = " " + target_string.lstrip()
        instance = super().text_to_instance(source_string, target_string, weight)
        # These are relations that should be filtered from a models predictions before evaluation.
        if filtered_relations is not None:
            instance.fields["metadata"].metadata["filtered_relations"] = filtered_relations
        return instance

    def _head_tail_truncation(self, source_string: str) -> str:
        """Truncates and returns `source_string` by concatenating the first 75% and last 25% of
        `self._max_length` tokens. Accounts for special tokens added by the tokenizer. If
        `source_string` is less than `self._max_length`, it is returned unmodified.
        """
        # Account for special tokens
        transformer_tokenizer = self._source_tokenizer.tokenizer
        max_length = self._max_length - self._source_tokenizer.num_special_tokens_for_sequence()
        tokenized_source = transformer_tokenizer.encode(source_string, add_special_tokens=False)
        if len(tokenized_source) > max_length:
            # Truncate by concatenating the first 75% of tokens and the last 25% of tokens
            head = floor(max_length * 0.75)
            tail = floor(max_length * 0.25)
            tail += max_length - head - tail
            tokenized_source = (
                tokenized_source[:head] + tokenized_source[len(tokenized_source) - tail :]
            )
            # This is the preferred way to detokenize a string.
            # See: https://github.com/huggingface/transformers/issues/14502
            source_string = transformer_tokenizer.decode(tokenized_source)
        return source_string

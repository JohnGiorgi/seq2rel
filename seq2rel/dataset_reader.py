import logging

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp_models.generation.dataset_readers import CopyNetDatasetReader
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("seq2rel")
class Seq2RelDatasetReader(CopyNetDatasetReader):
    """
    This is a thin wrapper around `CopyNetDatasetReader` to provide any of the modifications
    necessary to use the dataset reader for information extraction. The arguments are identical to
    `CopyNetSeq2Seq`. For details, please see:
    [`CopyNetSeq2Seq`](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/generation/dataset_readers/copynet_seq2seq.py),
    """

    @overrides
    def text_to_instance(
        self, source_string: str, target_string: str = None, _id: str = None
    ) -> Instance:  # type: ignore
        # We have to add a space in front of the source/target strings in order to achieve
        # consistant tokenization with certain tokenizers, like GPT. Enforce this behavior here.
        # See: https://github.com/huggingface/transformers/issues/1196
        source_string = " " + source_string.lstrip()
        if target_string:
            target_string = " " + target_string.lstrip()
        instance = super().text_to_instance(source_string, target_string)
        # If an unique ID was provided (optional), add it to the metadata
        if _id is not None:
            instance.fields["metadata"].metadata["_id"] = _id
        return instance

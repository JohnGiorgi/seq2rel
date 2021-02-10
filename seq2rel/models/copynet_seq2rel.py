import logging
from typing import Any, Dict, List

import torch
from allennlp.data import TextFieldTensors, Tokenizer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.training.metrics import Metric
from allennlp_models.generation.models import CopyNetSeq2Seq
from overrides import overrides
from seq2rel.common.util import sanitize_text

logger = logging.getLogger(__name__)


@Model.register("copynet_seq2rel")
class CopyNetSeq2Rel(CopyNetSeq2Seq):
    """
    This is a thin wrapper around `CopyNetSeq2Seq` to provide any of the modifications necessary to
    use the model for information extraction. Besides `target_tokenizer` and `sequence_based_metric`
    the arguments are identical to `CopyNetSeq2Seq`. For details, please see:
    [`CopyNetSeq2Seq`](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/generation/models/copynet_seq2seq.py),

    # Parameters

    source_embedder : `TextFieldEmbedder`, required
        Embedder for source side sequences
    encoder : `Seq2SeqEncoder`, optional (default = `None`)
        The encoder of the "encoder/decoder" model. If None, a `PassThroughEncoder` is used.
    target_tokenizer : `Tokenizer`, optional (default = `None`)
        The tokenizer used to tokenize the target sequence. If not `None`, this is used to
        un-tokenize the target sequence, otherwise tokens are joined by whitespace.
    sequence_based_metric : `Metric`, optional (default = `None`)
        A metric to track on validation data that takes lists of strings as input. This metric must
        accept two arguments when called, both of type `List[str]`. The first is a predicted
        sequence for each item in the batch and the second is a gold sequence for each item in the
        batch.
    """

    def __init__(
        self,
        source_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder = None,
        target_tokenizer: Tokenizer = None,
        tensor_based_metric: Metric = None,
        sequence_based_metric: Metric = None,
        **kwargs,  # type: ignore
    ) -> None:
        # I am expecting most users to use a PretrainedTransformerEmbedder as source_embedder,
        # in which case we don't need an encoder and it is annoying to have to specify an input_dim.
        # Assume, if the user does not specify an encoder, that they want a PassThroughEncoder.
        encoder = encoder or PassThroughEncoder(source_embedder.get_output_dim())
        super().__init__(source_embedder=source_embedder, encoder=encoder, **kwargs)
        self._target_tokenizer: Tokenizer = target_tokenizer
        self._sequence_based_metric = sequence_based_metric

        # The parent class initializes this to BLEU, but we aren't interested
        # in "tensor based metrics", so revert it to the users input.
        self._tensor_based_metric = tensor_based_metric

    @overrides
    def forward(
        self,  # type: ignore
        source_tokens: TextFieldTensors,
        source_token_ids: torch.Tensor,
        source_to_target: torch.Tensor,
        metadata: List[Dict[str, Any]],
        target_tokens: TextFieldTensors = None,
        target_token_ids: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        # Parameters

        source_tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()` applied on the source `TextField`. This will be
            passed through a `TextFieldEmbedder` and then through an encoder.
        source_token_ids : `torch.Tensor`, required
            Tensor containing IDs that indicate which source tokens match each other.
            Has shape: `(batch_size, source_sequence_length)`.
        source_to_target : `torch.Tensor`, required
            Tensor containing vocab index of each source token with respect to the
            target vocab namespace. Shape: `(batch_size, source_sequence_length)`.
        metadata : `List[Dict[str, Any]]`, required
            Metadata field that contains the original source tokens with key 'source_tokens'
            and any other meta fields. When 'target_tokens' is also passed, the metadata
            should also contain the original target tokens with key 'target_tokens'.
        target_tokens : `TextFieldTensors`, optional (default = `None`)
            Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
            target tokens are also represented as a `TextField` which must contain a "tokens"
            key that uses single ids.
        target_token_ids : `torch.Tensor`, optional (default = `None`)
            A tensor of shape `(batch_size, target_sequence_length)` which indicates which
            tokens in the target sequence match tokens in the source sequence.

        # Returns

        `Dict[str, torch.Tensor]`
        """
        state = self._encode(source_tokens)
        state["source_token_ids"] = source_token_ids
        state["source_to_target"] = source_to_target

        if target_tokens:
            state = self._init_decoder_state(state)
            output_dict = self._forward_loss(target_tokens, target_token_ids, state)
        else:
            output_dict = {}

        output_dict["metadata"] = metadata

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens:
                if self._tensor_based_metric is not None:
                    # shape: (batch_size, beam_size, max_sequence_length)
                    top_k_predictions = output_dict["predictions"]
                    # shape: (batch_size, max_predicted_sequence_length)
                    best_predictions = top_k_predictions[:, 0, :]
                    # shape: (batch_size, target_sequence_length)
                    gold_tokens = self._gather_extended_gold_tokens(
                        target_tokens["tokens"]["tokens"],
                        source_token_ids,
                        target_token_ids,
                    )
                    self._tensor_based_metric(best_predictions, gold_tokens)  # type: ignore
                if self._token_based_metric is not None:
                    predicted_tokens = self._get_predicted_tokens(
                        output_dict["predictions"], metadata, n_best=1
                    )
                    self._token_based_metric(  # type: ignore
                        predicted_tokens, [x["target_tokens"] for x in metadata]
                    )
                if self._sequence_based_metric is not None:
                    output_dict = self.make_output_human_readable(output_dict)
                    self._sequence_based_metric(  # type: ignore
                        output_dict["predicted_strings"], output_dict["target_strings"]
                    )

        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize predictions.

        After a beam search, the predicted indices correspond to tokens in the target vocabulary
        OR tokens in source sentence. Here we gather the actual tokens corresponding to
        the indices, and then convert these tokens to a human-readable string.
        """
        predicted_tokens = self._get_predicted_tokens(
            output_dict["predictions"], output_dict["metadata"], n_best=1
        )
        output_dict["predicted_tokens"] = predicted_tokens

        # We need the models predictions as a string in order to compute the sequence metrics.
        # Depending on the tokenizer used, we try to join the tokens into a string
        # in the smartest way possible. As a fallback, we join on whitespace.
        predicted_strings: List[str]
        if isinstance(self._target_tokenizer, PretrainedTransformerTokenizer):

            def _tokens_to_string(tokens: List[str]) -> str:
                return sanitize_text(
                    self._target_tokenizer.tokenizer.convert_tokens_to_string(tokens)
                )

        else:

            def _tokens_to_string(tokens: List[str]) -> str:
                return sanitize_text(" ".join(tokens))

        predicted_strings = [_tokens_to_string(tokens) for tokens in predicted_tokens]
        output_dict["predicted_strings"] = predicted_strings

        # Metadata is a list of dicts, enough to check if any of them contain "target_tokens".
        if "target_tokens" in output_dict["metadata"][-1]:
            target_tokens = [x["target_tokens"] for x in output_dict["metadata"]]
            target_strings = [_tokens_to_string(tokens) for tokens in target_tokens]
            output_dict["target_strings"] = target_strings

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(self._tensor_based_metric.get_metric(reset=reset))
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))
            if self._sequence_based_metric is not None:
                all_metrics.update(self._sequence_based_metric.get_metric(reset=reset))
        return all_metrics

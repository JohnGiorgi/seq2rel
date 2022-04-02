import logging
from typing import Any, Dict, List

import torch
from allennlp.common.lazy import Lazy
from allennlp.data import TextFieldTensors, Tokenizer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.modules import Attention, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.nn import util
from allennlp.training.metrics import Metric
from allennlp_models.generation.models import CopyNetSeq2Seq
from fastai.text.models import WeightDropout

from seq2rel.common.util import COREF_SEP_SYMBOL, sanitize_text

logger = logging.getLogger(__name__)


@Model.register("copynet_seq2rel")
class CopyNetSeq2Rel(CopyNetSeq2Seq):
    """
    This is a thin wrapper around `CopyNetSeq2Seq` to provide any of the modifications necessary to
    use the model for information extraction. For details, please see:
    [`CopyNetSeq2Seq`](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/generation/models/copynet_seq2seq.py).

    # Parameters

    source_embedder : `TextFieldEmbedder`, required
        Embedder for source side sequences
    encoder : `Seq2SeqEncoder`, optional (default = `None`)
        The encoder of the "encoder/decoder" model. If None, a `PassThroughEncoder` is used.
    attention : `Attention`, required
        This is used to get a dynamic summary of encoder outputs at each timestep
        when producing the "generation" scores for the target vocab.
    target_tokenizer : `Tokenizer`, optional (default = `None`)
        The tokenizer used to tokenize the target sequence. If not `None`, this is used to
        un-tokenize the target sequence, otherwise tokens are joined by whitespace.
    dropout : `float` (default = `0.1`)
        Dropout probability applied to the target embeddings and decoders inputs.
    weight_dropout : `float` (default = `0.5`)
        Dropout probability applied to the decoders hidden-to-hidden weights.
        See: https://arxiv.org/abs/1708.02182
    sequence_based_metrics : `List[Metric]`, optional (default = `None`)
        A list of metrics to track on validation data that takes lists of strings as input. These
        metrics must accept two arguments when called, both of type `List[str]`. The first is a
        predicted sequence for each item in the batch and the second is a gold sequence for each
        item in the batch.
    init_decoder_state_strategy: `str` (default = `"mean"`)
        If `"first"`, initialize decoders hidden state with first encoder output embedding (e.g.
        [CLS] token). If `"last"`, initialize decoders hidden state with last encoder output
        embedding (excluding padding). If `"mean"`, initialize decoders hidden state with mean of
        encoder output embeddings (excluding padding).
    """

    def __init__(
        self,
        source_embedder: TextFieldEmbedder,
        attention: Lazy[Attention],
        encoder: Seq2SeqEncoder = None,
        target_tokenizer: Tokenizer = None,
        dropout: float = 0.1,
        weight_dropout: float = 0.5,
        tensor_based_metric: Metric = None,
        sequence_based_metrics: List[Metric] = None,
        init_decoder_state_strategy: str = "mean",
        **kwargs: Any,  # type: ignore
    ) -> None:
        # I am expecting most users to use a PretrainedTransformerEmbedder as source_embedder,
        # in which case we don't need an encoder and it is annoying to have to specify an input_dim.
        # Assume, if the user does not specify an encoder, that they want a PassThroughEncoder.
        encoder = encoder or PassThroughEncoder(source_embedder.get_output_dim())
        # We construct this lazily so that the user doesn't have to provide the `embed_dim`
        # in the config file for `MultiheadAttention`.
        attention = attention.construct(embed_dim=encoder.get_output_dim())
        super().__init__(
            source_embedder=source_embedder, encoder=encoder, attention=attention, **kwargs
        )

        # Any seq2rel specific setup goes here
        self._target_tokenizer: Tokenizer = target_tokenizer
        self._sequence_based_metrics = sequence_based_metrics or []
        # TODO: I do not think this has any effect. Double check and drop if not needed.
        _ = self.vocab.add_token_to_namespace(COREF_SEP_SYMBOL, self._target_namespace)

        # Dropout to apply to the target embeddings and decoder inputs
        self._dropout = torch.nn.Dropout(dropout) if dropout else torch.nn.Identity()
        # Dropout to apply to the decoders hidden-to-hidden weights
        self._weight_dropout = weight_dropout
        self._decoder_cell: torch.nn.Module = WeightDropout(
            self._decoder_cell, self._weight_dropout, layer_names="weight_hh"
        )

        # The strategy to use for initializing the decoders hidden state
        self._init_decoder_state_strategy = init_decoder_state_strategy
        # The parent class initializes this to BLEU, but we aren't interested
        # in "tensor based metrics", so revert it to the users input.
        self._tensor_based_metric = tensor_based_metric

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Initialize the encoded state to be passed to the first decoding time step.
        """
        batch_size, _ = state["source_mask"].size()

        # Initialize the decoder hidden state according to self._init_decoder_state_strategy
        # and the decoder context with zeros.
        # shape: (batch_size, encoder_output_dim)
        if self._init_decoder_state_strategy == "first":
            final_encoder_output = state["encoder_outputs"][:, 0, :]
        elif self._init_decoder_state_strategy == "last":
            final_encoder_output = util.get_final_encoder_states(
                state["encoder_outputs"], state["source_mask"], self._encoder.is_bidirectional()
            )
        elif self._init_decoder_state_strategy == "mean":
            final_encoder_output = util.masked_mean(
                state["encoder_outputs"], state["source_mask"].unsqueeze(-1), dim=1
            )
        else:
            raise ValueError(
                f"An invalid 'init_decoder_state_strategy': '{self._init_decoder_state_strategy}'"
                " was provided to 'seq2rel.models.copynet_seq2rel.CopyNetSeq2Rel'. Expected one of"
                " 'first', 'last', or 'mean'."
            )

        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"].new_zeros(
            batch_size, self.decoder_output_dim
        )

        return state

    def forward(
        self,  # type: ignore
        source_tokens: TextFieldTensors,
        source_token_ids: torch.Tensor,
        source_to_target: torch.Tensor,
        metadata: List[Dict[str, Any]],
        target_tokens: TextFieldTensors = None,
        target_token_ids: torch.Tensor = None,
        weight: torch.Tensor = None,
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
            output_dict = self._forward_loss(target_tokens, target_token_ids, state, weight=weight)
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
                if self._sequence_based_metrics:
                    output_dict = self.make_output_human_readable(output_dict)
                    for metric in self._sequence_based_metrics:
                        metric(
                            predictions=output_dict["predicted_strings"],
                            ground_truths=output_dict["target_strings"],
                            filtered_relations=output_dict.get("filtered_relations"),
                        )
        return output_dict

    def _decoder_step(
        self,
        last_predictions: torch.Tensor,
        selective_weights: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # shape: (group_size, source_sequence_length, encoder_output_dim)
        encoder_outputs_mask = state["source_mask"]
        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)
        embedded_input = self._dropout(embedded_input)
        # shape: (group_size, source_sequence_length)
        attentive_weights = self._attention(
            state["decoder_hidden"], state["encoder_outputs"], encoder_outputs_mask
        )
        # shape: (group_size, encoder_output_dim)
        attentive_read = util.weighted_sum(state["encoder_outputs"], attentive_weights)
        # shape: (group_size, encoder_output_dim)
        selective_read = util.weighted_sum(state["encoder_outputs"], selective_weights)
        # shape: (group_size, target_embedding_dim + encoder_output_dim * 2)
        decoder_input = torch.cat((embedded_input, attentive_read, selective_read), -1)
        # shape: (group_size, decoder_input_dim)
        projected_decoder_input = self._input_projection_layer(decoder_input)
        projected_decoder_input = torch.nn.functional.gelu(projected_decoder_input)
        projected_decoder_input = self._dropout(projected_decoder_input)

        state["decoder_hidden"], state["decoder_context"] = self._decoder_cell(
            projected_decoder_input.float(),
            (state["decoder_hidden"].float(), state["decoder_context"].float()),
        )

        return state

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
                string = self._target_tokenizer.tokenizer.convert_tokens_to_string(tokens)
                # Depending on the tokenizer, convert_tokens_to_string may not correctly insert
                # spaces around the special token. We can fix that by encoding (tokenizing + indexing)
                # and then decoding. See: https://github.com/huggingface/transformers/issues/14502
                string = self._target_tokenizer.tokenizer.decode(
                    self._target_tokenizer.tokenizer.encode(string, add_special_tokens=False)
                )
                string = sanitize_text(string)
                return string

        else:

            def _tokens_to_string(tokens: List[str]) -> str:
                return sanitize_text(" ".join(tokens))

        predicted_strings = [_tokens_to_string(tokens) for tokens in predicted_tokens]
        output_dict["predicted_strings"] = predicted_strings

        # Metadata is a list of dicts, enough to check if any of them contain "target_tokens".
        if any("target_tokens" in batch for batch in output_dict["metadata"]):
            target_tokens = [batch["target_tokens"] for batch in output_dict["metadata"]]
            target_strings = [_tokens_to_string(tokens) for tokens in target_tokens]
            output_dict["target_strings"] = target_strings

        # Metadata is a list of dicts, enough to check if any of them contain "filtered_relations".
        if any("filtered_relations" in batch for batch in output_dict["metadata"]):
            filtered_relations = [batch["filtered_relations"] for batch in output_dict["metadata"]]
            output_dict["filtered_relations"] = filtered_relations

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(self._tensor_based_metric.get_metric(reset=reset))
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))
            for metric in self._sequence_based_metrics:
                all_metrics.update(metric.get_metric(reset=reset))
        return all_metrics

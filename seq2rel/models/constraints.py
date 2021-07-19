from typing import List

import torch
from allennlp.common.util import END_SYMBOL

from allennlp.nn.beam_search import Constraint, ConstraintStateType
from allennlp.nn.util import min_value_of_dtype
from overrides import overrides
from seq2rel.common.util import COREF_SEP_SYMBOL, END_OF_REL_SYMBOL


@Constraint.register("seq2rel")
class EnforceValidLinearization(Constraint):
    """A set of contraints applied to beam search during decoding for a seq2rel model.
    Together they should decrease the likelihood of generating an invalid linearized output.
    """

    def __init__(
        self, rel_tokens: List[str], ent_tokens: List[str], target_namespace: str, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # These are the indices in the target vocabulary that we will attempt to constrain.
        self._end_index = self.vocab.get_token_index(END_SYMBOL, target_namespace)
        self._rel_indices: List[int] = [
            self.vocab.get_token_index(token, target_namespace) for token in rel_tokens
        ]
        self._ent_indices: List[int] = [
            self.vocab.get_token_index(token, target_namespace) for token in ent_tokens
        ]
        self._coref_index: int = self.vocab.get_token_index(COREF_SEP_SYMBOL, target_namespace)
        self._eor_index: int = self.vocab.get_token_index(END_OF_REL_SYMBOL, target_namespace)
        self._copy_index_start: int = self.vocab.get_vocab_size(target_namespace)

    @overrides
    def init_state(
        self,
        batch_size: int,
    ) -> ConstraintStateType:
        return [
            [
                {  # At the first timestep, the only valid prediction is a relation token.
                    "allowed_indices": self._rel_indices,
                    "predicted_ents": 0,
                }
            ]
            for _ in range(batch_size)
        ]

    @overrides
    def apply(
        self,
        state: ConstraintStateType,
        class_log_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        # Compute these once up front to avoid computing them in a loop repeatedly.
        num_targets = class_log_probabilities.shape[-1]
        copy_indices = range(self._copy_index_start + 1, num_targets)
        for i, batch in enumerate(state):
            for j, beam in enumerate(batch):
                allowed_indices = beam["allowed_indices"]
                # This is a special case. In `_update_state`, we use `self._copy_index_start`
                # to denote that copying any token from the input is valid, so we need to allow
                # all copy indices here.
                if self._copy_index_start in allowed_indices:
                    allowed_indices.extend(copy_indices)
                # This is all possible predictions, minus the allowed indices and the end token.
                disallowed_indices = list(
                    set(range(num_targets)) - set(allowed_indices) - set((self._end_index,))
                )
                class_log_probabilities[i, j, disallowed_indices] = min_value_of_dtype(
                    class_log_probabilities.dtype
                )
        return class_log_probabilities

    @overrides
    def _update_state(
        self,
        state: ConstraintStateType,
        last_prediction: torch.Tensor,
    ) -> ConstraintStateType:
        for i, batch in enumerate(state):
            for j, beam in enumerate(batch):
                prediction = last_prediction[i, j].item()
                # We have just predicted a relation token.
                # The only valid next move is to copy.
                if prediction in self._rel_indices:
                    beam["allowed_indices"] = [self._copy_index_start]
                # We have just predicted an entity token.
                # The only valid next move is to copy or, if at least two entities have been
                # decoded, predict the end of relation token.
                elif prediction in self._ent_indices:
                    beam["predicted_ents"] += 1
                    beam["allowed_indices"] = [self._copy_index_start]
                    if beam["predicted_ents"] >= 2:
                        beam["allowed_indices"].append(self._eor_index)
                # We have just predicted a coref token.
                # The only valid next move is to copy.
                elif prediction == self._coref_index:
                    beam["allowed_indices"] = [self._copy_index_start]
                # We have just predicted the end of relation token.
                # The only valid next move is to predict a relation token.
                elif prediction == self._eor_index:
                    beam["predicted_ents"] = 0
                    beam["allowed_indices"] = self._rel_indices
                # We have just copied a token.
                # The only thing we can't do is generate a relation token.
                elif prediction >= self._copy_index_start:
                    beam["allowed_indices"] = self._ent_indices + [
                        self._coref_index,
                        self._copy_index_start,
                    ]

        return state

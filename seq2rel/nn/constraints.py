from typing import Any, List

import torch
from allennlp.common.util import END_SYMBOL
from allennlp.nn.beam_search import Constraint, ConstraintStateType
from allennlp.nn.util import min_value_of_dtype
from overrides import overrides
from seq2rel.common.util import COREF_SEP_SYMBOL


@Constraint.register("enforce_valid_linearization")
class EnforceValidLinearization(Constraint):
    """A set of contraints applied to beam search during decoding for a seq2rel model.
    Together they should decrease the likelihood of generating an invalid linearized output.

    # Parameters

    ent_tokens : `List[str]`
        The special entity tokens used to denote an entities type.
    rel_tokens : `List[str]`
        The special relation tokens used to denote a relations type.
    target_namespace : `str`
        The namespace of the target vocabulary.
    n_ary : `int`, optional (default = `2`)
        The number of entities in a relation.
    """

    def __init__(
        self,
        ent_tokens: List[str],
        rel_tokens: List[str],
        target_namespace: str,
        n_ary: int = 2,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self._end_index = self.vocab.get_token_index(END_SYMBOL, target_namespace)
        self._coref_index: int = self.vocab.get_token_index(COREF_SEP_SYMBOL, target_namespace)
        self._copy_index_start: int = self.vocab.get_vocab_size(target_namespace)

        self._ent_indices: List[int] = [
            self.vocab.get_token_index(token, target_namespace) for token in ent_tokens
        ]
        self._rel_indices: List[int] = [
            self.vocab.get_token_index(token, target_namespace) for token in rel_tokens
        ]

        self._n_ary = n_ary

    @overrides
    def init_state(
        self,
        batch_size: int,
    ) -> ConstraintStateType:
        return [
            [
                {  # At the first timestep, the only valid move is to copy or predict the EOS token.
                    "allowed_indices": [self._copy_index_start, self._end_index],
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
        copy_indices = range(self._copy_index_start, num_targets)
        for i, batch in enumerate(state):
            for j, beam in enumerate(batch):
                allowed_indices = beam["allowed_indices"]
                # In `_update_state`, we use `self._copy_index_start` to denote that copying any
                # token from the input is valid, so we need to extend to all copy indices here.
                if self._copy_index_start in allowed_indices:
                    allowed_indices.extend(copy_indices)
                # This is all possible predictions, minus the allowed indices.
                disallowed_indices = list(set(range(num_targets)) - set(allowed_indices))
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
                # The only valid next moves are to copy or to terminate.
                if prediction in self._rel_indices:
                    beam["allowed_indices"] = [self._copy_index_start, self._end_index]
                    beam["predicted_ents"] = 0
                # We have just predicted an entity token. The only valid next move is to copy and,
                # if `self._n_ary` entities have been decoded, predict a relation token.
                elif prediction in self._ent_indices:
                    beam["predicted_ents"] += 1
                    if beam["predicted_ents"] == self._n_ary:
                        beam["allowed_indices"] = self._rel_indices
                    else:
                        beam["allowed_indices"] = [self._copy_index_start]
                # We have just predicted a coref token.
                # The only valid next move is to copy.
                elif prediction == self._coref_index:
                    beam["allowed_indices"] = [self._copy_index_start]
                # We have just copied a token.
                # The only thing we can't do is generate a relation token.
                elif prediction >= self._copy_index_start:
                    beam["allowed_indices"] = self._ent_indices + [
                        self._coref_index,
                        self._copy_index_start,
                    ]

        return state

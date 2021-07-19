import copy
from typing import List

import torch
from allennlp.common.util import END_SYMBOL
from allennlp.data import Vocabulary
from allennlp.nn.beam_search import Constraint, ConstraintStateType
from allennlp.nn.util import min_value_of_dtype
from overrides import overrides
from seq2rel.common.util import COREF_SEP_SYMBOL, END_OF_REL_SYMBOL


@Constraint.register("seq2rel")
class Seq2RelConstraint(Constraint):
    """A set of contraints applied to beam search during decoding for a seq2rel model.
    Together they should decrease the likelihood of generating an invalid linearized output.
    """

    def __init__(self, vocab: Vocabulary, target_namespace: str) -> None:
        super().__init__()
        self.vocab = vocab
        # These are the indices in the target vocabulary that we will attempt to constrain.
        self._end_index = self.vocab.get_token_index(END_SYMBOL, target_namespace)
        self._bor_indices: List[int] = [self.vocab.get_token_index("@CID@", target_namespace)]
        self._ent_indices: List[int] = [
            self.vocab.get_token_index("@CHEMICAL@", target_namespace),
            self.vocab.get_token_index("@DISEASE@", target_namespace),
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
                    "allowed_indices": self._bor_indices,
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
        for i, batch in enumerate(state):
            for j, beam in enumerate(batch):
                allowed_indices = copy.deepcopy(beam["allowed_indices"])
                # This is a special case. All indices larger than the vocab size were copied.
                if self._copy_index_start in allowed_indices:
                    copy_indices = range(
                        self._copy_index_start + 1, class_log_probabilities.shape[-1]
                    )
                    allowed_indices.extend(copy_indices)
                # This is all possible predictions, minus the allowed indices and the end token.
                disallowed_indices = list(
                    set(range(class_log_probabilities.shape[-1]))
                    - set(allowed_indices)
                    - set((self._end_index,))
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
                if prediction in self._bor_indices:
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
                    beam["allowed_indices"] = self._bor_indices
                # We have just copied a token.
                # The only thing we can't do is generate a relation token.
                elif prediction >= self._copy_index_start:
                    beam["allowed_indices"] = self._ent_indices + [
                        self._coref_index,
                        self._copy_index_start,
                    ]

        return state

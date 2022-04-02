import random
from typing import Callable

import torch
from allennlp.common import Params
from allennlp.nn.beam_search import Constraint
from allennlp.nn.util import min_value_of_dtype


class TestEnforceValidLinearization:
    def test_enforce_valid_linearization_constraint_init_state(
        self, params: Params, vocab: Callable
    ) -> None:
        batch_size = 2
        # `params` is derived from the text_fixtures/experiement.jsonnet config.
        constraint_params = params.pop("model").pop("beam_search").pop("constraints")[0]
        # Provide empty instances, otherwise AllenNLP will complain.
        constraint = Constraint.from_params(constraint_params, vocab=vocab(instances=[]))

        state = constraint.init_state(batch_size)
        assert len(state) == batch_size
        for beam_states in state:
            assert len(beam_states) == 1
            beam_state = beam_states[0]
            assert len(beam_state.keys()) == 2
            assert beam_state["allowed_indices"] == [constraint._target_vocab_size]
            assert beam_state["predicted_ents"] == 0

    def test_enforce_valid_linearization_constraint_apply(
        self, params: Params, vocab: Callable
    ) -> None:
        batch_size = 4
        constraint_params = params.pop("model").pop("beam_search").pop("constraints")[0]
        constraint = Constraint.from_params(constraint_params, vocab=vocab(instances=[]))
        # Arbitrarily set the number of classes to the vocab size + 4.
        # This ensures we have some copy indices to test.
        num_targets = constraint._target_vocab_size + 4

        # Create all unique states here, so that we cant test applying them.
        beam_size = 1
        state = [
            [
                # We have just predicted a relation token.
                # The only valid next moves are to copy or to terminate.
                {
                    "allowed_indices": [constraint._target_vocab_size],
                    "predicted_ents": 0,
                },
            ],
            [
                # We have just predicted an entity token, but we have not yet predicted at least
                # `constraint._n_ary` entities. The only valid next move is to copy.
                {
                    "allowed_indices": [constraint._target_vocab_size],
                    "predicted_ents": constraint._n_ary - 1,
                },
            ],
            [
                # We have just predicted an entity token, and the total entities is now `constraint._n_ary`
                # The only valid next move is to predict a relation token.
                {
                    "allowed_indices": constraint._rel_indices,
                    "predicted_ents": constraint._n_ary,
                },
            ],
            [
                # We have just copied a token.
                # The only thing we can't do is generate a relation token.
                {
                    "allowed_indices": constraint._ent_indices
                    + [constraint._coref_index, constraint._target_vocab_size],
                    "predicted_ents": 0,
                },
            ],
        ]

        # Build up the expected disallowed indices for each item in the batch.
        expected_disallowed_indices = []
        all_indices = set(range(num_targets))
        copy_indices = list(range(constraint._target_vocab_size, num_targets))
        # Batch index 0
        allowed_indices = copy_indices + [constraint._end_index]
        disallowed_indices = list(all_indices - set(allowed_indices))
        expected_disallowed_indices.extend([[0, 0, index] for index in disallowed_indices])
        # Batch index 1
        allowed_indices = copy_indices + [constraint._end_index]
        disallowed_indices = list(all_indices - set(allowed_indices))
        expected_disallowed_indices.extend([[1, 0, index] for index in disallowed_indices])
        # Batch index 2
        allowed_indices = constraint._rel_indices + [constraint._end_index]
        disallowed_indices = list(all_indices - set(allowed_indices))
        expected_disallowed_indices.extend([[2, 0, index] for index in disallowed_indices])
        # Batch index 3
        allowed_indices = (
            constraint._ent_indices
            + copy_indices
            + [constraint._coref_index, constraint._end_index]
        )
        disallowed_indices = list(all_indices - set(allowed_indices))
        expected_disallowed_indices.extend([[3, 0, index] for index in disallowed_indices])

        # Create some random log probabilities and apply the constraints to them.
        log_probabilities = torch.rand(batch_size, beam_size, num_targets)
        log_probabilities = constraint.apply(state, log_probabilities)
        actual_disallowed_indices = torch.nonzero(
            log_probabilities == min_value_of_dtype(log_probabilities.dtype)
        ).tolist()

        assert len(actual_disallowed_indices) == len(expected_disallowed_indices)
        assert all(indices in actual_disallowed_indices for indices in expected_disallowed_indices)
        assert all(indices in expected_disallowed_indices for indices in actual_disallowed_indices)

    def test_enforce_valid_linearization_constraint_update_state(
        self, params: Params, vocab: Callable
    ) -> None:
        constraint_params = params.pop("model").pop("beam_search").pop("constraints")[0]
        constraint = Constraint.from_params(constraint_params, vocab=vocab(instances=[]))

        # The `allowed_indices` will be overwritten by `_update_state`, so they don't matter here.
        state = [
            [
                # Last prediction was a relation token. Make sure that `predicted_ents` reset to 0.
                {"allowed_indices": [], "predicted_ents": 0},
                {"allowed_indices": [], "predicted_ents": 10},
            ],
            [
                # Last prediction was an entity token. Try with and without required number of ents.
                {"allowed_indices": [], "predicted_ents": constraint._n_ary - 2},
                {"allowed_indices": [], "predicted_ents": constraint._n_ary - 1},
            ],
            [
                # Last prediction was a coref token.
                {"allowed_indices": [], "predicted_ents": 0},
                # Last prediction was a copy.
                {"allowed_indices": [], "predicted_ents": 0},
            ],
        ]
        # When there are multiple indices that will return the same expect value, sample randomly.
        predictions = torch.LongTensor(
            [
                [
                    random.choice(constraint._rel_indices),
                    random.choice(constraint._rel_indices),
                ],
                [
                    random.choice(constraint._ent_indices),
                    random.choice(constraint._ent_indices),
                ],
                [
                    constraint._coref_index,
                    constraint._target_vocab_size,
                ],
            ]
        )
        backpointers = torch.LongTensor([[0, 1], [0, 1], [0, 1]])

        expected_state = [
            [
                {
                    "allowed_indices": [constraint._target_vocab_size],
                    "predicted_ents": 0,
                },
                {
                    "allowed_indices": [constraint._target_vocab_size],
                    "predicted_ents": 0,
                },
            ],
            [
                {
                    "allowed_indices": [constraint._target_vocab_size],
                    "predicted_ents": constraint._n_ary - 1,
                },
                {
                    "allowed_indices": constraint._rel_indices,
                    "predicted_ents": constraint._n_ary,
                },
            ],
            [
                {"allowed_indices": [constraint._target_vocab_size], "predicted_ents": 0},
                # Previously prediction == `copy_index_start`
                {
                    "allowed_indices": constraint._ent_indices
                    + [constraint._coref_index, constraint._target_vocab_size],
                    "predicted_ents": 0,
                },
            ],
        ]
        updated_state = constraint.update_state(state, predictions, backpointers)
        assert updated_state == expected_state

import random

import torch
from allennlp.nn.beam_search import Constraint
from allennlp.nn.util import min_value_of_dtype


class TestEnforceValidLinearization:
    def test_enforce_valid_linearization_constraint_init_state(self, params, vocab):
        batch_size = 2
        # `params` is derived from the text_fixtures experiement.jsonnet config.
        constraint_params = params.pop("model").pop("beam_search").pop("constraints")[0]
        # Provide empty instances, otherwise AllenNLP will complain.
        constraint = Constraint.from_params(constraint_params, vocab=vocab(instances=[]))

        state = constraint.init_state(batch_size)
        assert len(state) == batch_size
        for beam_states in state:
            assert len(beam_states) == 1
            beam_state = beam_states[0]
            assert len(beam_state.keys()) == 2
            assert beam_state["allowed_indices"] == constraint._rel_indices
            assert beam_state["predicted_ents"] == 0

    def test_enforce_valid_linearization_constraint_apply(self, params, vocab):
        batch_size = 2
        constraint_params = params.pop("model").pop("beam_search").pop("constraints")[0]
        constraint = Constraint.from_params(constraint_params, vocab=vocab(instances=[]))
        # Arbitrarly set the number of classes to the vocab size + 2.
        # This ensures we have some copy indices to test, without being a burden.
        num_targets = constraint._copy_index_start + 2

        # Chose one of the more difficult cases to test: 2 or more entities have been decoded.
        beam_size = 1
        state = [
            [
                {
                    "allowed_indices": [constraint._copy_index_start, constraint._eor_index],
                    "predicted_ents": 2,
                },
            ],
            [
                {
                    "allowed_indices": [constraint._copy_index_start, constraint._eor_index],
                    "predicted_ents": 2,
                },
            ],
        ]
        log_probabilities = torch.rand(batch_size, beam_size, num_targets)
        constraint.apply(state, log_probabilities)

        # The expected disallowed indices are all possible target indices minus the copy indices and
        # `_eor_index` (specified by the `state`) and the `constraint._end_index` (always allowed).
        expected_disallowed_indices = (
            set(range(num_targets))
            - set(range(constraint._copy_index_start, num_targets))
            - set((constraint._eor_index,))
            - set((constraint._end_index,))
        )
        expected_disallowed_indices = [
            [batch, 0, index]
            for index in expected_disallowed_indices
            for batch in range(batch_size)
        ]
        actual_disallowed_indices = torch.nonzero(
            log_probabilities == min_value_of_dtype(log_probabilities.dtype)
        ).tolist()
        assert len(actual_disallowed_indices) == len(expected_disallowed_indices)
        assert all(indices in actual_disallowed_indices for indices in expected_disallowed_indices)

    def test_enforce_valid_linearization_constraint_update_state(self, params, vocab):
        constraint_params = params.pop("model").pop("beam_search").pop("constraints")[0]
        constraint = Constraint.from_params(constraint_params, vocab=vocab(instances=[]))

        # The `allowed_indices` will be overwritten by `_update_state`, so they don't matter here.
        state = [
            [
                {"allowed_indices": [], "predicted_ents": 0},
                {"allowed_indices": [], "predicted_ents": 0},
            ],
            [
                {"allowed_indices": [], "predicted_ents": 1},
                {"allowed_indices": [], "predicted_ents": 0},
            ],
            [
                {"allowed_indices": [], "predicted_ents": 2},
                {"allowed_indices": [], "predicted_ents": 0},
            ],
        ]
        # In some cases there are multiple indices that will return the same expect value, so we
        # sample from them at random.
        predictions = torch.LongTensor(
            [
                [
                    random.choice(constraint._rel_indices),
                    random.choice(constraint._ent_indices),
                ],
                [
                    random.choice(constraint._ent_indices),
                    constraint._coref_index,
                ],
                [
                    constraint._eor_index,
                    constraint._copy_index_start,
                ],
            ]
        )
        backpointers = torch.LongTensor([[0, 1], [0, 1], [0, 1]])

        expected_state = [
            [
                # Previously prediction from `rel_indices`
                {"allowed_indices": [constraint._copy_index_start], "predicted_ents": 0},
                # Previously prediction from `ent_indices`, with <2 `predicted_ents`
                {"allowed_indices": [constraint._copy_index_start], "predicted_ents": 1},
            ],
            [
                # Previously prediction from `ent_indices`, with >=2 `predicted_ents`
                {
                    "allowed_indices": [constraint._copy_index_start, constraint._eor_index],
                    "predicted_ents": 2,
                },
                # Previously prediction == `coref_index`
                {"allowed_indices": [constraint._copy_index_start], "predicted_ents": 0},
            ],
            [
                # Previously prediction == `eor_index`
                {"allowed_indices": constraint._rel_indices, "predicted_ents": 0},
                # Previously prediction == `copy_index_start`
                {
                    "allowed_indices": constraint._ent_indices
                    + [constraint._coref_index, constraint._copy_index_start],
                    "predicted_ents": 0,
                },
            ],
        ]
        updated_state = constraint.update_state(state, predictions, backpointers)
        assert updated_state == expected_state

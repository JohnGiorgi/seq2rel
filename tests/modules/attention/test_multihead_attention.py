import numpy
import torch
from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.attention.attention import Attention
from numpy.testing import assert_almost_equal

from seq2rel.modules.attention.multihead_attention import MultiheadAttention


class TestMultiheadAttention(AllenNlpTestCase):
    def test_can_init_multihead(self):
        legacy_attention = Attention.from_params(
            Params({"type": "multihead_attention", "embed_dim": 4, "num_heads": 2})
        )
        isinstance(legacy_attention, MultiheadAttention)

    def test_multihead_similarity(self):
        attn = MultiheadAttention(embed_dim=4, num_heads=2)
        vector = torch.FloatTensor([[0, 0, 0, 0], [1, 1, 1, 1]])
        matrix = torch.FloatTensor(
            [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]
        )
        with torch.no_grad():
            output = attn(vector, matrix)

        assert_almost_equal(
            output.sum(dim=-1).numpy(),
            numpy.array([1.0, 1.0]),
            decimal=2,
        )

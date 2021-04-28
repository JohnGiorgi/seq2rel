import math

import torch
from allennlp.modules.attention.attention import Attention
from overrides import overrides


@Attention.register("dk_scaled_dot_product")
class DkScaledDotProductAttention(Attention):
    """
    Computes attention between two tensors using scaled dot product.
    # Reference: [Attention Is All You Need (Vaswani et al, 2017)]
    # (https://api.semanticscholar.org/CorpusID:13756489)

    Registered as an `Attention` with name "dk_scaled_dot_product".
    """

    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return matrix.bmm(vector.unsqueeze(-1)).squeeze(-1) / math.sqrt(matrix.size(-1))

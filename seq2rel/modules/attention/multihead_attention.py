from typing import Any

import torch
from allennlp.modules.attention.attention import Attention
from torch import nn


@Attention.register("multihead_attention")
class MultiheadAttention(Attention):
    """
    Computes attention between two tensors using multihead attention.
    # Reference: [Attention Is All You Need (Vaswani et al, 2017)]
    # (https://api.semanticscholar.org/CorpusID:13756489)

    Registered as an `Attention` with name "multihead_attention".

    # Parameters
    embed_dim : `int`, required
        Size of the embedding dimension of the input tensors.
    num_heads : `int`, required
        Number of heads to use in the multihead attention.
    **kwargs : `Any`, optional, (default = `{}`)
        Optional keyword arguments passed to the `MultiheadAttention` constructor.

    """

    def __init__(self, embed_dim: int, num_heads: int, **kwargs: Any) -> None:
        super().__init__(normalize=False)

        self._multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, **kwargs)

    def forward(
        self, vector: torch.Tensor, matrix: torch.Tensor, matrix_mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        query = vector.unsqueeze(0)
        key = matrix.transpose(0, 1)
        value = key.clone()
        key_padding_mask = None if matrix_mask is None else ~matrix_mask  # type: ignore
        _, attn_output_weights = self._multihead_attn(
            query, key, value, key_padding_mask=key_padding_mask
        )
        attn_output_weights = attn_output_weights.squeeze(1)
        return attn_output_weights

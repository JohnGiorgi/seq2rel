import pathlib

import pytest
import torch
from allennlp.common.params import Params
from allennlp.common.testing import ModelTestCase
from allennlp.models import Model


class TestCopyNetSeq2Rel(ModelTestCase):
    def setup_method(self) -> None:
        super().setup_method()
        # We need to override the path set by AllenNLP
        self.FIXTURES_ROOT = (
            pathlib.Path(__file__).parent / ".." / ".."
        ).resolve() / "test_fixtures"
        self.set_up_model(
            self.FIXTURES_ROOT / "experiment.jsonnet",
            self.FIXTURES_ROOT / "data" / "train.tsv",
        )

    def test_model_can_train_save_load(self):
        self.ensure_model_can_train_save_and_load(
            self.param_file,
            tolerance=1e-2,
            gradients_to_ignore=[
                # We don't currently use the attention projection layer in the decoder.
                "_attention._multihead_attn.out_proj.weight",
                "_attention._multihead_attn.out_proj.bias",
                # HF initializes a pooler, and AllenNLP complains because we don't train it.
                "_source_embedder.token_embedder_tokens.transformer_model.pooler.dense.weight",
                "_source_embedder.token_embedder_tokens.transformer_model.pooler.dense.bias",
            ],
        )

    def test_invalid_init_decoder_state_strategy(self):
        params = Params.from_file(self.param_file)
        params["model"]["init_decoder_state_strategy"] = "blahblah"
        model = Model.from_params(vocab=self.vocab, params=params.get("model"))
        state = {"source_mask": torch.randn(2, 4)}
        with pytest.raises(ValueError):
            _ = model._init_decoder_state(state=state)

import pathlib

import pytest
from allennlp.common.params import Params
from allennlp.common.testing import ModelTestCase
from allennlp.models import Model


class TestSeq2Rel(ModelTestCase):
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
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)

    def test_invalid_init_decoder_state_strategy(self):
        params = Params.from_file(self.param_file)
        params["model"]["init_decoder_state_strategy"] = "blahblah"
        with pytest.raises(ValueError):
            Model.from_params(vocab=self.vocab, params=params.get("model"))

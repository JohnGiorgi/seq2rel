import pathlib

from allennlp.common.testing import ModelTestCase


class TestCopyNetSeq2Rel(ModelTestCase):
    def setup_method(self) -> None:
        super().setup_method()
        # We need to override the path set by AllenNLP
        self.FIXTURES_ROOT = (
            (pathlib.Path(__file__).parent / ".." / "..").resolve()
            / "test_fixtures"
            / "copynet_seq2rel"
        )
        self.set_up_model(
            self.FIXTURES_ROOT / "experiment.jsonnet",
            self.FIXTURES_ROOT / "data" / "train.tsv",
        )

    def test_model_can_train_save_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)

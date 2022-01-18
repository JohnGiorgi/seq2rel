import pathlib

import pytest
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from allennlp.data import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from seq2rel.dataset_reader import Seq2RelDatasetReader


class TestSeq2RelDatasetReader(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.FIXTURES_ROOT = (pathlib.Path(__file__).parent / "..").resolve() / "test_fixtures"
        params = Params.from_file(self.FIXTURES_ROOT / "experiment.jsonnet")
        self.reader: Seq2RelDatasetReader = DatasetReader.from_params(params["dataset_reader"])
        instances = self.reader.read(
            self.FIXTURES_ROOT / "data" / "train.tsv",
        )
        self.instances = ensure_list(instances)
        self.vocab = Vocabulary.from_params(params=params["vocabulary"], instances=self.instances)

    def test_head_tail_truncation(self, params: Params) -> None:
        max_length = 24
        dataset_reader_params = params.pop("dataset_reader")
        dataset_reader_params["max_length"] = max_length
        reader = DatasetReader.from_params(dataset_reader_params)
        source_string = (
            "Anaphylaxis to cisplatin is an infrequent life-threatening complication which may"
            " occur even in patients who have received prior treatment with cisplatin."
        )

        expected_string = "anaphylaxis to cisplatin is an infrequent life treatment with cisplatin."
        expected_length = max_length - reader._target_tokenizer.num_special_tokens_for_sequence()

        actual_string = reader._head_tail_truncation(source_string)
        actual_length = len(reader._source_tokenizer.tokenizer.tokenize(actual_string))
        assert actual_string == expected_string
        assert actual_length == expected_length

    def test_head_tail_truncation_value_error(self, params: Params) -> None:
        max_length = 24
        dataset_reader_params = params.pop("dataset_reader")
        dataset_reader_params["max_length"] = max_length
        dataset_reader_params["source_tokenizer"] = None
        reader = DatasetReader.from_params(dataset_reader_params)
        with pytest.raises(ValueError):
            reader.text_to_instance("")

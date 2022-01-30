import pathlib

import pytest
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import END_SYMBOL, START_SYMBOL, ensure_list
from allennlp.data import DatasetReader


class TestSeq2RelDatasetReader(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.FIXTURES_ROOT = (pathlib.Path(__file__).parent / "..").resolve() / "test_fixtures"

    def test_default_format(self, params: Params) -> None:
        dataset_reader_params = params.pop("dataset_reader")
        reader = DatasetReader.from_params(dataset_reader_params)
        instances = reader.read(
            self.FIXTURES_ROOT / "data" / "train.tsv",
        )
        instances = ensure_list(instances)

        assert len(instances) == 1
        fields = instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == [
            reader._source_tokenizer.tokenizer.cls_token,
            "lidocaine",
            "-",
            "induced",
            "cardiac",
            "as",
            "##yst",
            "##ole",
            ".",
            reader._source_tokenizer.tokenizer.sep_token,
        ]
        assert [t.text for t in fields["target_tokens"].tokens] == [
            START_SYMBOL,
            "lidocaine",
            "@CHEMICAL@",
            "cardiac",
            "as",
            "##yst",
            "##ole",
            "@DISEASE@",
            "@CID@",
            END_SYMBOL,
        ]

    def test_filtered_format(self, params: Params) -> None:
        dataset_reader_params = params.pop("dataset_reader")
        reader = DatasetReader.from_params(dataset_reader_params)
        instances = reader.read(
            self.FIXTURES_ROOT / "data" / "valid.tsv",
        )
        instances = ensure_list(instances)

        assert len(instances) == 1
        fields = instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == [
            reader._source_tokenizer.tokenizer.cls_token,
            "lidocaine",
            "-",
            "induced",
            "cardiac",
            "as",
            "##yst",
            "##ole",
            ".",
            reader._source_tokenizer.tokenizer.sep_token,
        ]
        assert [t.text for t in fields["target_tokens"].tokens] == [
            START_SYMBOL,
            "lidocaine",
            "@CHEMICAL@",
            "cardiac",
            "as",
            "##yst",
            "##ole",
            "@DISEASE@",
            "@CID@",
            END_SYMBOL,
        ]
        fields["metadata"].metadata[
            "filtered_relations"
        ] = "lidocaine @CHEMICAL@ cardiac asystole @DISEASE@ @CID@"

    def test_head_tail_truncation(self, params: Params) -> None:
        max_length = 24
        dataset_reader_params = params.pop("dataset_reader")
        dataset_reader_params["max_length"] = max_length
        reader = DatasetReader.from_params(dataset_reader_params)
        source_string = (
            "Anaphylaxis to cisplatin is an infrequent life-threatening complication which may"
            " occur even in patients who have received prior treatment with cisplatin."
        )

        expected_string = (
            "anaphylaxis to cisplatin is an infrequent life - threatening"
            " complication which may occur even in patients received prior treatment with cisplatin."
        )
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

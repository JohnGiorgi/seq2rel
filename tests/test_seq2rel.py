from seq2rel import __version__


def test_version():
    assert __version__ == "0.1.0"


class TestSeq2Rel:
    def test_ade_model(self, pretrained_ade_model, ade_examples):
        texts, expected = ade_examples
        actual = pretrained_ade_model(texts)
        assert actual == expected

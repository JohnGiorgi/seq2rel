from typing import Callable, List, Tuple

import pytest
from seq2rel import Seq2Rel
from allennlp.common import Params
from allennlp.data import Vocabulary

PRETRAINED_ADE_MODEL = "ade"


@pytest.fixture(scope="module")
def pretrained_ade_model() -> Seq2Rel:
    return Seq2Rel(PRETRAINED_ADE_MODEL, overrides={"dataset_reader.type": "seq2rel"})


@pytest.fixture()
def ade_examples() -> Tuple[List[str], List[str]]:
    texts = [
        "Vincristine induced cranial polyneuropathy.",
        "Intravenous diazepam exacerbated the seizures.",
        "Acute myocardial infarction due to coronary spasm associated with L-thyroxine therapy.",
    ]
    annotations = [
        "@ADE@ vincristine @DRUG@ cranial polyneuropathy @EFFECT@ @EOR@",
        "@ADE@ diazepam @DRUG@ seizures @EFFECT@ @EOR@",
        (
            "@ADE@ l - thyroxine @DRUG@ acute myocardial infarction @EFFECT@ @EOR@"
            " @ADE@ l - thyroxine @DRUG@ coronary spasm @EFFECT@ @EOR@"
        ),
    ]
    return texts, annotations


@pytest.fixture()
def params():
    return Params.from_file("test_fixtures/experiment.jsonnet")


@pytest.fixture
def vocab(params: Params) -> Callable:
    """This is a fixture factory. It returns a function that you can use
    to create an AllenNLP `Vocabulary` object. It accepts optional `**extras`
    which will be used along with `params` to create the `Vocabulary` object.
    """

    def _vocab(**extras) -> Vocabulary:
        return Vocabulary.from_params(params.pop("vocabulary"), **extras)

    return _vocab

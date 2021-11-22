from pathlib import Path
from typing import Callable, List, Tuple

import pytest
from allennlp.common import Params
from allennlp.data import Vocabulary
from seq2rel import Seq2Rel
from seq2rel.common.util import HINT_SEP_SYMBOL
from seq2rel.training.callbacks.concatenation_augmentation import ConcatenationAugmentationCallback

PRETRAINED_ADE_MODEL = "ade"


@pytest.fixture(scope="module")
def pretrained_ade_model() -> Seq2Rel:
    # TODO: The overrides exists because this model was trained before the dataset reader rename.
    # Remove when this is updated to a newer pretrained model.
    return Seq2Rel(PRETRAINED_ADE_MODEL, overrides={"dataset_reader.type": "seq2rel"})


@pytest.fixture()
def ade_examples() -> Tuple[List[str], List[str]]:
    texts = [
        "Vincristine induced cranial polyneuropathy.",
        "Intravenous diazepam exacerbated the seizures.",
        "Acute myocardial infarction due to coronary spasm associated with L-thyroxine therapy.",
    ]
    annotations = [
        "vincristine @DRUG@ cranial polyneuropathy @EFFECT@ @ADE@",
        "diazepam @DRUG@ seizures @EFFECT@ @ADE@",
        (
            "l - thyroxine @DRUG@ acute myocardial infarction @EFFECT@ @ADE@"
            " l - thyroxine @DRUG@ coronary spasm @EFFECT@ @ADE@"
        ),
    ]
    return texts, annotations


@pytest.fixture()
def concatenation_augmentation(
    tmp_path: Path,
) -> Callable[[bool], ConcatenationAugmentationCallback]:
    train_data_without_hints = ["first source\tfirst target", "second source\tsecond target"]
    train_data_with_hints = [
        f"A @ENT@ B @ENT@ {HINT_SEP_SYMBOL} first source\tfirst target",
        f"C @ENT@ D @ENT@ {HINT_SEP_SYMBOL} second source\tsecond target",
    ]

    def _concatenation_augmentation(with_hints: bool = False):
        train_data_path = tmp_path / "train.tsv"
        train_data = train_data_with_hints[:] if with_hints else train_data_without_hints[:]
        train_data_path.write_text("\n".join(train_data).strip())
        callback = ConcatenationAugmentationCallback(
            # There are two training examples, so and aug_frac of 1 this gives us 1 augmented example.
            serialization_dir="",
            train_data_path=str(train_data_path),
            aug_frac=1.0,
        )
        return callback

    return _concatenation_augmentation


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

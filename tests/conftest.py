from pathlib import Path
from typing import Callable

import pytest
from allennlp.common import Params
from allennlp.data import Vocabulary

from seq2rel.common.util import HINT_SEP_SYMBOL
from seq2rel.training.callbacks.concatenation_augmentation import ConcatenationAugmentationCallback


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

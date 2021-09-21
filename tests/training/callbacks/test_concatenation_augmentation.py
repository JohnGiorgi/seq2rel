from pathlib import Path

import pytest
from seq2rel.training.callbacks.concatenation_augmentation import ConcatenationAugmentationCallback


class TestConcatenationAugmentationCallback:
    def test_aug_frac_value_error(self) -> None:
        with pytest.raises(ValueError):
            _ = ConcatenationAugmentationCallback(
                serialization_dir="", train_data_path="", aug_frac=1.1
            )
        with pytest.raises(ValueError):
            _ = ConcatenationAugmentationCallback(
                serialization_dir="", train_data_path="", aug_frac=-0.1
            )

    def test_on_start(self, concatenation_augmentation: ConcatenationAugmentationCallback) -> None:
        # Ensure that on object instantiation, there are two training examples.
        train_data = (
            Path(concatenation_augmentation._train_data_path).read_text().strip().splitlines()
        )
        assert len(train_data) == 2

        # Ensure that on training start, there are two plus one training examples.
        concatenation_augmentation.on_start(trainer="")
        train_data = (
            Path(concatenation_augmentation._train_data_path).read_text().strip().splitlines()
        )
        assert len(train_data) == 3

    def test_on_epoch(self, concatenation_augmentation: ConcatenationAugmentationCallback) -> None:
        # Ensure that on object instantiation, there are two training examples.
        train_data = (
            Path(concatenation_augmentation._train_data_path).read_text().strip().splitlines()
        )
        assert len(train_data) == 2

        # Ensure that on epoch end, there are two plus one training examples.
        concatenation_augmentation.on_epoch(trainer="")
        train_data = (
            Path(concatenation_augmentation._train_data_path).read_text().strip().splitlines()
        )
        assert len(train_data) == 3

    def test_on_end(self, concatenation_augmentation: ConcatenationAugmentationCallback) -> None:
        # This is the train data BEFORE any augmentation.
        expected = (
            Path(concatenation_augmentation._train_data_path).read_text().strip().splitlines()
        )
        # Purposefully modify the training data on disk, and check that `on_end` restores it
        Path(concatenation_augmentation._train_data_path).write_text(expected[0].strip())
        concatenation_augmentation.on_end(trainer="")
        actual = Path(concatenation_augmentation._train_data_path).read_text().strip().splitlines()
        assert actual == expected

    def test_format_instance(
        self, concatenation_augmentation: ConcatenationAugmentationCallback
    ) -> None:
        first_instance = "I am the first instance"
        second_instance = "I am the second instance"

        # Test with no sep_token provided
        sep_token = " "
        expected = first_instance + sep_token + second_instance
        actual = concatenation_augmentation._format_instance(first_instance, second_instance)
        assert actual == expected

        # Test with sep_token provided
        concatenation_augmentation._sep_token = "[SEP]"
        expected = first_instance + f" {concatenation_augmentation._sep_token} " + second_instance
        actual = concatenation_augmentation._format_instance(first_instance, second_instance)
        assert actual == expected

    def test_augment(self, concatenation_augmentation: ConcatenationAugmentationCallback) -> None:
        # Load the training data and create a concatenated example.
        train_data = (
            Path(concatenation_augmentation._train_data_path).read_text().strip().splitlines()
        )
        first_source, first_target = train_data[0].split("\t")
        second_source, second_target = train_data[1].split("\t")
        concatenated_one = f"{first_source} {second_source}\t{first_target} {second_target}"
        concatenated_two = f"{second_source} {first_source}\t{second_target} {first_target}"

        # This works because there is only two possible augmentated examples given
        # `concatenation_augmentation._train_data` and `concatenation_augmentation._aug_frac`.
        expected_one = train_data + [concatenated_one]
        expected_two = train_data + [concatenated_two]
        actual = concatenation_augmentation._augment()
        assert actual == expected_one or actual == expected_two

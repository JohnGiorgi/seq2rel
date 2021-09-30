import logging
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from allennlp.training.callbacks.callback import TrainerCallback

if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer


logger = logging.getLogger(__name__)


@TrainerCallback.register("concat_aug")
class ConcatenationAugmentationCallback(TrainerCallback):
    def __init__(
        self,
        serialization_dir: str,
        train_data_path: str,
        aug_frac: float = 0.25,
        sep_token: Optional[str] = None,
    ) -> None:
        """
        Creates augmented training data via concatentation. Before training and after each
        epoch, a fraction of training examples, `aug_frac`, will be randomly sampled from
        the training set at `train_data_path`. The source and target text will be concatenated
        by `sep_token` (or a space if not provided), added to to the existing training data and
        written to `train_data_path`. The file `train_data_path` will be restored to its original
        state at the end of training.

        Note, for this to work, the dataset reader will need to be invoked at the end of each
        epoch. This can be achieved by setting the `max_instances_in_memory` of your data loader
        to be equal to the train set size (including the augmented examples).

        See the following papers for more information on the augmentation strategy:
            - https://arxiv.org/abs/2105.01691
            - https://arxiv.org/abs/2104.08478

        # Parameters

        train_data_path: `str`, required
            Path to the training data.
        aug_frac: `float`, optional (default = `0.20`)
            The fraction of training examples to randomly sample to create the augmented data.
        sep_token: `str`, optional (default = `None`)
            The token used to join the source and target text when concatenating examples
            (not including whitespace). If not provided, a single whitespace token will be used.
        """
        if aug_frac > 1.0 or aug_frac <= 0.0:
            raise ValueError(f"aug_frac must be <=1.0 or > 0.0. Got {aug_frac}.")

        super().__init__(serialization_dir)
        self._train_data_path = train_data_path
        self._train_data = Path(self._train_data_path).read_text().strip().splitlines()
        self._sep_token = sep_token
        self._aug_frac = aug_frac

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs: Any
    ) -> None:
        if not is_primary:
            return None
        logger.info("Training started. Augmenting training data.")
        augmented = self._augment()
        Path(self._train_data_path).write_text("\n".join(augmented).strip())

    def on_epoch(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs: Any
    ) -> None:
        if not is_primary:
            return None
        logger.info("Epoch finished. Augmenting training data.")
        augmented = self._augment()
        Path(self._train_data_path).write_text("\n".join(augmented).strip())

    def on_end(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs: Any
    ) -> None:
        if not is_primary:
            return None
        logger.info(f"Training finished. Restoring training data file: {self._train_data_path}.")
        Path(self._train_data_path).write_text("\n".join(self._train_data).strip())

    def _format_instance(self, first_instance: str, second_instance: str) -> str:
        formatted_instance = f"{first_instance.strip()}"
        if self._sep_token:
            formatted_instance += f" {self._sep_token}"
        formatted_instance += f" {second_instance.strip()}"
        return formatted_instance

    def _augment(self) -> List[str]:
        # Take a random `aug_frac` percent of the examples to augment.
        num_aug = math.ceil(len(self._train_data) * self._aug_frac)
        train_data = random.sample(self._train_data, num_aug)
        # Create augmented examples via concatentation.
        augmented = []
        for i in range(0, len(train_data) - 1):
            first_source, first_target = train_data[i].split("\t")
            second_source, second_target = train_data[i + 1].split("\t")
            # Concatenate examples to form the new source and target.
            source = self._format_instance(first_source, second_source)
            target = self._format_instance(first_target, second_target)
            augmented.append(f"{source}\t{target}")
        return self._train_data + augmented

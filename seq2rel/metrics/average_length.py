from typing import List, Dict
from allennlp.training.metrics import Metric


@Metric.register("average_length")
class AverageLength(Metric):
    """Compute the average length of the decoded and target sequences. This is useful as a
    diagnostic. E.g., if the average length of decoded sequences is longer than target sequences,
    you may want to add or increase the length penalty (and vice versa).
    """

    supports_distributed = False

    def __init__(self):
        self._decoded_lengths = []
        self._target_lengths = []

    def __call__(self, predictions: List[List[str]], targets: List[List[str]]) -> None:
        for pred, target in zip(predictions, targets):
            self._decoded_lengths.append(len(pred))
            self._target_lengths.append(len(target))

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        decoded_mean_length = sum(self._decoded_lengths) / len(self._decoded_lengths)
        target_mean_length = sum(self._target_lengths) / len(self._target_lengths)

        if reset:
            self.reset()

        return {
            "decoded_mean_length": round(decoded_mean_length, 2),
            "target_mean_length": round(target_mean_length, 2),
        }

    def reset(self) -> None:
        self._decoded_lengths = []
        self._target_lengths = []

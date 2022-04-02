from typing import Dict, List

from allennlp.training.metrics import Metric


@Metric.register("average_length")
class AverageLength(Metric):
    """Compute the average length of the decoded and target sequences. This is useful as a
    diagnostic. E.g., if the average length of decoded sequences is longer than target sequences,
    you may want to add or increase the length penalty (and vice versa).
    """

    supports_distributed = False

    def __init__(self):
        self._prediction_lengths = []
        self._target_lengths = []

    def __call__(self, predictions: List[List[str]], targets: List[List[str]]) -> None:
        for pred, target in zip(predictions, targets):
            self._prediction_lengths.append(len(pred))
            self._target_lengths.append(len(target))

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        prediction_mean_length = (
            sum(self._prediction_lengths) / len(self._prediction_lengths)
            if len(self._prediction_lengths) > 0
            else 0
        )
        target_mean_length = (
            sum(self._target_lengths) / len(self._target_lengths)
            if len(self._target_lengths) > 0
            else 0
        )

        if reset:
            self.reset()

        return {
            "predictions_mean_length": round(prediction_mean_length, 2),
            "targets_mean_length": round(target_mean_length, 2),
            "predictions_to_targets_length_ratio": round(
                prediction_mean_length / target_mean_length, 2
            )
            if target_mean_length > 0
            else 0,
        }

    def reset(self) -> None:
        self._prediction_lengths = []
        self._target_lengths = []

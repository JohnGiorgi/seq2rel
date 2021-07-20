import re
from typing import Dict, List, Optional

from allennlp.training.metrics import Metric
from seq2rel.common.util import REL_PATTERN, deserialize_annotations


@Metric.register("valid_sequences")
class ValidSequences(Metric):
    """Compute the count of (valid) relations in the decoded and target sequences. This is useful
    as a diagnositic. E.g., if the ratio of predicted relations to valid predicted ratios is much
    greater than 1.0, it suggests that the model is having trouble generating valid relations. Note
    that the argument `ground_truths` to `__call__` is ignored and provided only for API consistency.
    """

    def __init__(self):
        # The number of predictions, valid or not.
        self._pred_count = 0
        # The number of valid predictions.
        self._valid_pred_count = 0

    def __call__(self, predictions: List[str], ground_truths: Optional[List[str]] = None) -> None:

        self._valid_pred_count += sum(
            len(deserialized.values()) for deserialized in deserialize_annotations(predictions)
        )
        for pred in predictions:
            # Check how many relations we can parse out.
            self._pred_count += len(REL_PATTERN.findall(pred))
            # After these have been considered, check if the string is non-empty.
            # Count this as a prediction if so.
            self._pred_count += int(bool(re.sub(REL_PATTERN, "", pred).strip()))

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        pred_count = float(self._pred_count)
        valid_pred_count = float(self._valid_pred_count)
        num_predicted_to_valid_relations_ratio = (
            pred_count / valid_pred_count if valid_pred_count > 0 else 0.0
        )

        if reset:
            self.reset()

        return {
            "num_predicted_relations": pred_count,
            "num_valid_predicted_relations": valid_pred_count,
            "num_predicted_to_valid_relations_ratio": round(
                num_predicted_to_valid_relations_ratio, 2
            ),
        }

    def reset(self) -> None:
        self._pred_count = 0
        self._valid_pred_count = 0

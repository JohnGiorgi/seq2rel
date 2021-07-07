from typing import List, Optional, Set

import torch
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
from allennlp.training.metrics.metric import Metric
from seq2rel.common.util import EntityAnnotation, deserialize_annotations


def _fuzzy_cluster_match(
    pred_rel: EntityAnnotation,
    gold_rels: Set[EntityAnnotation],
    threshold: float = 0.5,
) -> bool:
    """Given some predicted relation `pred_rel`, returns True if there is a fuzzy match to any
    relation in the ground truth relations `gold_rels`. A fuzzy match occurs if there exists a
    ground truth relation where, for every predicted cluster, P, there is a gold cluster G such
    that | P ∩ G | / |P| > cluster_threshold. The number of predicted clusters and their predicted
    entity classes must exactly match the ground truth regardless of threshold.
    """
    for gold_rel in gold_rels:
        # If the number of gold and predicted clusters differ then we don't have a match.
        if len(gold_rel) != len(pred_rel):
            continue
        matched = True
        for (pred_mentions, pred_label), (gold_mentions, gold_label) in zip(pred_rel, gold_rel):
            # Convert to a set, as we don't care about duplicates or order.
            pred = set(pred_mentions)
            gold = set(gold_mentions)
            # A predicted cluster (P) matches a gold cluster (G) if:
            #   1. | P ∩ G | / |P| > threshold
            #   2. The predicted cluster label matches the gold cluster label
            if (len(pred & gold) / len(pred)) <= threshold or pred_label != gold_label:
                matched = False
                break
        # Found a fuzzy match for all clusters, and therefore the predicted relation is correct.
        if matched:
            return True

    return False


@Metric.register("fbeta_seq2rel")
class FBetaMeasureSeq2Rel(FBetaMeasure):
    """A thin wrapper around FBetaMeasure, which computes the precision, recall and F-measure for
    the output a Seq2Rel model. Besides `labels` and `ordered_ents`, the parameters are the same as
    the parent class. For details, please see:
    [FBetaMeasure](https://github.com/allenai/allennlp/blob/main/allennlp/training/metrics/fbeta_measure.py)

    # Parameters

    labels: `list`
        The set of labels to include (and their order if `average is None`.)
        Labels present in the data can be excluded, for example to calculate a
        multi-class average ignoring a majority negative class. Labels not present
        in the data will result in 0 components in a macro or weighted average.
    cluster_threshold : `float`, optional (default = `None`)
        If `cluster_threshold`, use fuzzy matching, where a predicted cluster (P) is considered a
        true positive if | P ∩ G | / | P | > `cluster_threshold` for at least one gold cluster (G).
        A reasonable threshold value is `0.5`.
    """

    supports_distributed = True

    def __init__(
        self,
        labels: List[str],
        cluster_threshold: Optional[float] = None,
        beta: float = 1.0,
        average: Optional[str] = None,
    ) -> None:
        super().__init__(beta=beta, average=average)
        # Unlike the parent class, we require labels to be not None. To be compatible with
        # the parent class, self._labels needs to be a list of integers representing the
        # positions of each class. For our purposes, these labels can just be [0,...,len(labels)]
        self._str_labels = labels
        self._labels = list(range(len(labels)))
        self._num_classes = len(self._labels)

        if cluster_threshold is not None and (cluster_threshold <= 0 or cluster_threshold > 1):
            raise ValueError(f"cluster_threshold must be between (0, 1]. Got {cluster_threshold}.")
        self._cluster_threshold = cluster_threshold

    def __call__(self, predictions: List[str], ground_truths: List[str]) -> None:
        """
        # Parameters

        predictions : `list`, required.
            A list of predictions.
        ground_truths : `torch.Tensor`, required.
            A list corresponding to some ground truths to evaluate against.
        """
        if len(ground_truths) != len(predictions):
            raise ValueError(
                f"len(ground_truths) must equal len(predictions)."
                f" Got {len(ground_truths)} and {len(predictions)}."
            )

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:  # type: ignore
            self._true_positive_sum = torch.zeros(self._num_classes)
            self._true_sum = torch.zeros(self._num_classes)
            self._pred_sum = torch.zeros(self._num_classes)
            self._total_sum = torch.zeros(self._num_classes)

        pred_annotations = deserialize_annotations(predictions)
        gold_annotations = deserialize_annotations(ground_truths)

        # Predictions and ground truths are contained with equal length lists as they are per-batch.
        for pred_ann, gold_ann in zip(pred_annotations, gold_annotations):
            if gold_ann:
                for rel_label, gold_rels in gold_ann.items():
                    # Filter out any labels not provided at instantiation.
                    if self._labels and rel_label not in self._str_labels:
                        continue
                    # Get the predicted relations for this label.
                    class_index = self._str_labels.index(rel_label)
                    pred_rels = pred_ann.get(rel_label, [])
                    # Convert to a set, as we don't care about duplicates or order.
                    dedup_pred_rels = set(pred_rels)
                    dedup_gold_rels = set(gold_rels)
                    # If cluster_threshold, use fuzzy matching to determine true positives.
                    if self._cluster_threshold:
                        for rel in dedup_pred_rels:
                            if _fuzzy_cluster_match(rel, dedup_gold_rels, self._cluster_threshold):
                                self._true_positive_sum[class_index] += 1  # type: ignore
                            self._pred_sum[class_index] += 1
                    else:
                        self._true_positive_sum[class_index] += len(  # type: ignore
                            dedup_pred_rels & dedup_gold_rels
                        )
                        self._pred_sum[class_index] += len(dedup_pred_rels)
                    self._true_sum[class_index] += len(dedup_gold_rels)
            # No corresponding gold annotation, so these are all false-positives.
            else:
                for rel_label, pred_rels in pred_ann.items():
                    dedup_pred_rels = set(pred_rels)
                    if self._labels and rel_label not in self._str_labels:
                        continue
                    class_index = self._str_labels.index(rel_label)
                    self._pred_sum[class_index] += len(dedup_pred_rels)

        # We need to set the total sum to be compatible with the parent class.
        # Because we do not support masking, it is equal to the "true sum".
        self._total_sum = self._true_sum.detach().clone()


@Metric.register("f1_seq2rel")
class F1MeasureSeq2Rel(FBetaMeasureSeq2Rel):
    def __init__(
        self,
        labels: List[str],
        cluster_threshold: Optional[float] = None,
        average: Optional[str] = None,
    ) -> None:
        super().__init__(
            labels=labels, cluster_threshold=cluster_threshold, beta=1.0, average=average
        )

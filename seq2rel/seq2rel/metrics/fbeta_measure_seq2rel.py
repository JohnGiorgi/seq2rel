import re
from typing import Dict, List, Optional, Set, Tuple, Iterable

import torch
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
from allennlp.training.metrics.metric import Metric


@Metric.register("fbeta_seq2rel")
class FBetaMeasureSeq2Rel(FBetaMeasure):
    """A thin wrapper around FBetaMeasure, which computes the precision, recall and F-measure for
    the output a Seq2Rel model. Besides `labels` and `ordered_ents`, the parameters are the same as
    the parent class. For details, please see:
    [FBetaMeasure](https://github.com/allenai/allennlp/blob/master/allennlp/training/metrics/fbeta_measure.py)

    # Parameters

    labels: `list`
        The set of labels to include (and their order if `average is None`.)
        Labels present in the data can be excluded, for example to calculate a
        multi-class average ignoring a majority negative class. Labels not present
        in the data will result in 0 components in a macro or weighted average.
    ordered_ents: `bool`, optional (default = `True`)
        Whether or not the entities within a relations should be considered ordered.
    """

    supports_distributed = True

    def __init__(
        self,
        labels: List[str],
        ordered_ents: bool = True,
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
        self._ordered_ents = ordered_ents

        # RegEx patterns to split serialized string representations into entities and relations.
        self._ent_pattern = re.compile(r"(?:\s?)(.*?)(?:\s?)<([A-Z][A-Z0-9]*)\b[^>]*>")
        self._rel_pattern = re.compile(r"<([A-Z][A-Z0-9]*)\b[^>]*>(.*?)</\1>")

    def __call__(self, predictions: List[str], gold_labels: List[str]) -> None:
        """
        # Parameters

        predictions : `list`, required.
            A list of predictions.
        gold_labels : `torch.Tensor`, required.
            A list corresponding to some gold labels to evaluate against.
        """
        if len(gold_labels) != len(predictions):
            raise ValueError(
                f"len(gold_labels) must equal len(predictions)."
                f" Got {len(gold_labels)} and {len(predictions)}."
            )

        pred_rels = self._get_ents_and_rels(predictions)
        gold_rels = self._get_ents_and_rels(gold_labels)

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:  # type: ignore
            self._true_positive_sum = torch.zeros(self._num_classes)
            self._true_sum = torch.zeros(self._num_classes)
            self._pred_sum = torch.zeros(self._num_classes)
            self._total_sum = torch.zeros(self._num_classes)

        for pred, gold in zip(pred_rels, gold_rels):
            for rel_type, gold_ents in gold.items():
                if rel_type not in self._str_labels:
                    continue
                class_index = self._str_labels.index(rel_type)
                pred_ents = pred.get(rel_type, set())
                self._true_positive_sum[class_index] += len(pred_ents & gold_ents)  # type: ignore
                self._pred_sum[class_index] += len(pred_ents)
                self._true_sum[class_index] += len(gold_ents)
        # We need to set the total sum to be compatible with the parent class.
        # Because we do not support masking, it is equal to the "true sum".
        self._total_sum = self._true_sum.detach().clone()

    def _get_ents_and_rels(
        self, serialized_strings: List[str]
    ) -> List[Dict[str, Set[Tuple[Tuple[str]]]]]:
        """Returns dictionaries containing the entities and the relations that are present
        in the `serialized_strings`, the string serialized representation of entities and relations.

        # Parameters

        serialized_strings: `list`
            A list containing the string serialized representation of entities and relations.

        # Returns

        A list of dictionaries, keyed by class name, containing the relations of the string
        serialized representations `serialized_strings`.

        """
        parsed_rels = []  # type: ignore
        for string in serialized_strings:
            parsed_rels.append({})
            for rel in self._rel_pattern.findall(string):
                rel_type, rel_string = rel
                ents: Iterable[str] = self._ent_pattern.findall(rel_string)
                # We can enforce order by casting the entities as a tuple.
                ents = tuple(ents) if self._ordered_ents else set(ents)
                if rel_type in parsed_rels[-1]:
                    parsed_rels[-1][rel_type].add(ents)
                else:
                    parsed_rels[-1][rel_type] = {
                        ents,
                    }
        return parsed_rels


@Metric.register("f1_seq2rel")
class F1MeasureSeq2Rel(FBetaMeasureSeq2Rel):
    def __init__(
        self,
        labels: List[str],
        ordered_ents: bool = True,
        average: Optional[str] = None,
    ) -> None:
        super().__init__(labels=labels, ordered_ents=ordered_ents, beta=1.0, average=average)

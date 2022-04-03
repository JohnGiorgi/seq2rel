import copy
from typing import List, Set

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given
from torch.testing import assert_allclose

from seq2rel.common.util import EntityAnnotation
from seq2rel.metrics.fbeta_measure_seq2rel import (
    F1MeasureSeq2Rel,
    FBetaMeasureSeq2Rel,
    _relaxed_entity_match,
)


def test_relaxed_entity_match() -> None:
    threshold = 0.5
    # The matching gold annotation purposely comes second to ensure that order doesn't matter.
    gold_rels: Set[EntityAnnotation] = set(
        (
            (
                (("methamphetamine", "meth"), "CHEMICAL"),
                (("psychotic disorders", "psychosis"), "DISEASE"),
            ),
            (
                (("suxamethonium chloride", "suxamethonium", "sch"), "CHEMICAL"),
                (("fasciculations", "fasciculation"), "DISEASE"),
            ),
        ),
    )
    # Wrong entity type
    pred_rel: EntityAnnotation = (
        (("suxamethonium chloride", "suxamethonium", "sch"), "ARBITRARY"),
        (("fasciculations", "fasciculation"), "DISEASE"),
    )
    assert not _relaxed_entity_match(pred_rel, gold_rels, threshold=threshold)
    # Missing an entire entity
    pred_rel = (
        # 0 / 1, NOT over threshold
        (("arbitrary",), "CHEMICAL"),
        # 2 / 2, over threshold
        (("fasciculations", "fasciculation"), "DISEASE"),
    )
    assert not _relaxed_entity_match(pred_rel, gold_rels, threshold=threshold)
    # Two additional mentions in each entity
    pred_rel = (
        # 3 / 5, over threshold
        (("suxamethonium chloride", "suxamethonium", "sch", "wrong", "incorrect"), "CHEMICAL"),
        # 2 / 4, NOT the threshold
        (("fasciculations", "fasciculation", "wrong", "incorrect"), "DISEASE"),
    )
    assert not _relaxed_entity_match(pred_rel, gold_rels, threshold=threshold)
    # Missing a single mention in each entity
    pred_rel = (
        # 2 / 3, over threshold
        (("suxamethonium chloride", "suxamethonium"), "CHEMICAL"),
        # 1 / 1, over threshold
        (("fasciculations",), "DISEASE"),
    )
    assert _relaxed_entity_match(pred_rel, gold_rels, threshold=threshold)
    # One additional mention in each entity
    pred_rel = (
        # 2 / 3, over threshold
        (("suxamethonium chloride", "suxamethonium", "arbitrary"), "CHEMICAL"),
        # 2 / 2, over threshold
        (("fasciculations", "fasciculation", "arbitrary"), "DISEASE"),
    )
    assert _relaxed_entity_match(pred_rel, gold_rels, threshold=threshold)
    # Mention order differs
    pred_rel = (
        # 2 / 3, over threshold
        (("suxamethonium", "suxamethonium chloride", "sch"), "CHEMICAL"),
        # 2 / 2, over threshold
        (("fasciculation", "fasciculations"), "DISEASE"),
    )
    assert _relaxed_entity_match(pred_rel, gold_rels, threshold=threshold)
    # Entity order differs but `ordered_ents=False`
    pred_rel = (
        # 2 / 3, over threshold
        (("fasciculations", "fasciculation", "arbitrary"), "DISEASE"),
        # 2 / 2, over threshold
        (("suxamethonium", "suxamethonium chloride", "arbitrary"), "CHEMICAL"),
    )
    assert _relaxed_entity_match(pred_rel, gold_rels, threshold=threshold, ordered_ents=False)
    # Entity order differs but `ordered_ents=True`
    pred_rel = (
        # 2 / 3, over threshold
        (("fasciculations", "fasciculation", "arbitrary"), "DISEASE"),
        # 2 / 2, over threshold
        (("suxamethonium", "suxamethonium chloride", "arbitrary"), "CHEMICAL"),
    )
    assert not _relaxed_entity_match(pred_rel, gold_rels, threshold=threshold, ordered_ents=True)


class FBetaMeasureSeq2RelTestCase:
    def setup_method(self):
        self.labels = ["PHYSICAL", "GENETIC"]
        self.predictions = [
            # This is a false positive
            "fgf-2 @GGP@ rps19 @GGP@ @PHYSICAL@",
            "I don't contain anything of interest!",
            # This prediction contains a relation with an incorrect order of entities
            "atg1 @GGP@ atg1 @GGP@ @PHYSICAL@ atg17 @GGP@ atg1 @GGP@ @PHYSICAL@",
            # This prediction is missing a relation
            "b-myb @GGP@ cbp @GGP@ @PHYSICAL@ b-myb @GGP@ cbp @GGP@ @GENETIC@",
            # This prediction contains coreferent mentions, where one entity is missing a mention
            "insulin @GGP@ peroxiredoxin 4; prdx4 @GGP@ @PHYSICAL@",
        ]
        self.targets = [
            "",
            "I don't contain anything of interest!",
            "atg1 @GGP@ atg1 @GGP@ @PHYSICAL@ atg1 @GGP@ atg17 @GGP@ @PHYSICAL@",
            (
                "b-myb @GGP@ cbp @GGP@ @GENETIC@"
                " b-myb @GGP@ cbp @GGP@ @PHYSICAL@"
                " myb @GGP@ cbp @GGP@ @GENETIC@ "
            ),
            "proinsulin; insulin @GGP@ peroxiredoxin 4; prdx4 @GGP@ @PHYSICAL@",
        ]

        # Detailed target state
        self.pred_sum = [5, 1]
        self.true_sum = [4, 2]
        self.true_positive_sum = [3, 1]
        self.total_sum = [4, 2]

        desired_precisions = [3 / 5, 1.00]
        desired_recalls = [3 / 4, 1 / 2]
        desired_fscores = [
            (2 * p * r) / (p + r) if p + r != 0.0 else 0.0
            for p, r in zip(desired_precisions, desired_recalls)
        ]
        self.desired_precisions = desired_precisions
        self.desired_recalls = desired_recalls
        self.desired_fscores = desired_fscores

        # Threshold used for relaxed entity matching
        self.threshold = 0.5


class TestFBetaMeasureSeq2Rel(FBetaMeasureSeq2RelTestCase):
    """Tests for FBetaMeasureSeq2Rel. Loosely based on:
    https://github.com/allenai/allennlp/blob/main/tests/training/metrics/fbeta_measure_test.py
    """

    def setup_method(self):
        super().setup_method()

    @given(threshold=st.floats(min_value=-1, max_value=1))
    def test_fbeta_seq2rel_invalid_threshold_raises_value_error(self, threshold: float):
        if threshold <= 0.0 or threshold > 1.0:
            with pytest.raises(ValueError):
                _ = FBetaMeasureSeq2Rel(labels=self.labels, threshold=threshold)
        # Sanity check that valid values don't raise an error.
        else:
            _ = FBetaMeasureSeq2Rel(labels=self.labels, threshold=threshold)

    def test_fbeta_seq2rel_diff_pred_and_ground_truth_lens_raises_value_error(
        self,
    ):
        fbeta = FBetaMeasureSeq2Rel(labels=self.labels)
        with pytest.raises(ValueError):
            fbeta(self.predictions, self.targets[:-1])

    def test_fbeta_seq2rel_multiclass_state(
        self,
    ):
        fbeta = FBetaMeasureSeq2Rel(labels=self.labels)
        fbeta(self.predictions, self.targets)

        # check state
        assert_allclose(fbeta._pred_sum.tolist(), self.pred_sum)
        assert_allclose(fbeta._true_sum.tolist(), self.true_sum)
        assert_allclose(fbeta._true_positive_sum.tolist(), self.true_positive_sum)
        assert_allclose(fbeta._total_sum.tolist(), self.total_sum)

    def test_fbeta_seq2rel_multiclass_metric(self):
        fbeta = FBetaMeasureSeq2Rel(labels=self.labels)
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        # check value
        assert_allclose(precisions, self.desired_precisions)
        assert_allclose(recalls, self.desired_recalls)
        assert_allclose(fscores, self.desired_fscores)

        # check type
        assert isinstance(precisions, List)
        assert isinstance(recalls, List)
        assert isinstance(fscores, List)

    def test_fbeta_seq2rel_multiclass_metric_relaxed_entity_match(self):
        fbeta = FBetaMeasureSeq2Rel(labels=self.labels, threshold=self.threshold)
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        # With relaxed entity matching, one of the preds for the class at index 0 is now correct.
        # increment the true positives by 1 and recompute the desired values.
        true_positive_sum = copy.deepcopy(self.true_positive_sum)
        desired_precisions = copy.deepcopy(self.desired_precisions)
        desired_recalls = copy.deepcopy(self.desired_recalls)
        true_positive_sum[0] += 1
        desired_precisions[0] = (true_positive_sum[0]) / self.pred_sum[0]
        desired_recalls[0] = (true_positive_sum[0]) / self.true_sum[0]
        desired_fscores = [
            (2 * p * r) / (p + r) if p + r != 0.0 else 0.0
            for p, r in zip(desired_precisions, desired_recalls)
        ]

        # check value
        assert_allclose(precisions, desired_precisions)
        assert_allclose(recalls, desired_recalls)
        assert_allclose(fscores, desired_fscores)

        # check type
        assert isinstance(precisions, List)
        assert isinstance(recalls, List)
        assert isinstance(fscores, List)

    def test_fbeta_seq2rel_multiclass_metric_ordered_ents(self):
        fbeta = FBetaMeasureSeq2Rel(labels=self.labels, ordered_ents=True)
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        # With `ordered_ents=True`, one of the predictions for the class at index 0 is now incorrect.
        # decrement the true positives by 1 and recompute the desired values.
        true_positive_sum = copy.deepcopy(self.true_positive_sum)
        desired_precisions = copy.deepcopy(self.desired_precisions)
        desired_recalls = copy.deepcopy(self.desired_recalls)
        true_positive_sum[0] -= 1
        desired_precisions[0] = (true_positive_sum[0]) / self.pred_sum[0]
        desired_recalls[0] = (true_positive_sum[0]) / self.true_sum[0]
        desired_fscores = [
            (2 * p * r) / (p + r) if p + r != 0.0 else 0.0
            for p, r in zip(desired_precisions, desired_recalls)
        ]

        # check value
        assert_allclose(precisions, desired_precisions)
        assert_allclose(recalls, desired_recalls)
        assert_allclose(fscores, desired_fscores)

        # check type
        assert isinstance(precisions, List)
        assert isinstance(recalls, List)
        assert isinstance(fscores, List)

    def test_fbeta_seq2rel_multiclass_metric_remove_duplicate_ents(self):
        fbeta = FBetaMeasureSeq2Rel(labels=self.labels, remove_duplicate_ents=True)
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        # With `remove_duplicate_ents=True`, one of the predictions for the class at index 0 is now
        # removed. Decrement the true positives by 1 and recompute the desired values.
        pred_sum = copy.deepcopy(self.pred_sum)
        true_positive_sum = copy.deepcopy(self.true_positive_sum)
        desired_precisions = copy.deepcopy(self.desired_precisions)
        desired_recalls = copy.deepcopy(self.desired_recalls)
        pred_sum[0] -= 1
        true_positive_sum[0] -= 1
        desired_precisions[0] = (true_positive_sum[0]) / pred_sum[0]
        desired_recalls[0] = (true_positive_sum[0]) / self.true_sum[0]
        desired_fscores = [
            (2 * p * r) / (p + r) if p + r != 0.0 else 0.0
            for p, r in zip(desired_precisions, desired_recalls)
        ]

        # check value
        assert_allclose(precisions, desired_precisions)
        assert_allclose(recalls, desired_recalls)
        assert_allclose(fscores, desired_fscores)

        # check type
        assert isinstance(precisions, List)
        assert isinstance(recalls, List)
        assert isinstance(fscores, List)

    def test_fbeta_seq2rel_multiclass_metric_filtered_relations(self):
        fbeta = FBetaMeasureSeq2Rel(labels=self.labels)
        # Remove a false positive prediction by providing it via `filtered_relations`
        filtered_relations = [""] * len(self.predictions)
        filtered_relations[0] = self.predictions[0]
        fbeta(self.predictions, self.targets, filtered_relations=filtered_relations)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        # With `filtered_relations`, one of false positive predictions for the class at index 0
        # is now removed. Decrement the predicted sum by 1 and recompute the desired values.
        pred_sum = copy.deepcopy(self.pred_sum)
        true_positive_sum = copy.deepcopy(self.true_positive_sum)
        desired_precisions = copy.deepcopy(self.desired_precisions)
        desired_recalls = copy.deepcopy(self.desired_recalls)
        pred_sum[0] -= 1
        desired_precisions[0] = (true_positive_sum[0]) / pred_sum[0]
        desired_recalls[0] = (true_positive_sum[0]) / self.true_sum[0]
        desired_fscores = [
            (2 * p * r) / (p + r) if p + r != 0.0 else 0.0
            for p, r in zip(desired_precisions, desired_recalls)
        ]

        # check value
        assert_allclose(precisions, desired_precisions)
        assert_allclose(recalls, desired_recalls)
        assert_allclose(fscores, desired_fscores)

        # check type
        assert isinstance(precisions, List)
        assert isinstance(recalls, List)
        assert isinstance(fscores, List)

    def test_fbeta_seq2rel_multiclass_macro_average_metric(self):
        fbeta = FBetaMeasureSeq2Rel(labels=self.labels, average="macro")
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        # We keep the expected values in CPU because FBetaMeasure returns them in CPU.
        macro_precision = torch.tensor(self.desired_precisions).mean()
        macro_recall = torch.tensor(self.desired_recalls).mean()
        macro_fscore = torch.tensor(self.desired_fscores).mean()
        # check value
        assert_allclose(precisions, macro_precision)
        assert_allclose(recalls, macro_recall)
        assert_allclose(fscores, macro_fscore)

        # check type
        assert isinstance(precisions, float)
        assert isinstance(recalls, float)
        assert isinstance(fscores, float)

    def test_fbeta_seq2rel_multiclass_micro_average_metric(self):
        fbeta = FBetaMeasureSeq2Rel(labels=self.labels, average="micro")
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        # We keep the expected values in CPU because FBetaMeasure returns them in CPU.
        true_positives = torch.tensor([3, 1], dtype=torch.float32)
        false_positives = torch.tensor([2, 0], dtype=torch.float32)
        false_negatives = torch.tensor([1, 1], dtype=torch.float32)
        mean_true_positive = true_positives.mean()
        mean_false_positive = false_positives.mean()
        mean_false_negative = false_negatives.mean()

        micro_precision = mean_true_positive / (mean_true_positive + mean_false_positive)
        micro_recall = mean_true_positive / (mean_true_positive + mean_false_negative)
        micro_fscore = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        # check value
        assert_allclose(precisions, micro_precision)
        assert_allclose(recalls, micro_recall)
        assert_allclose(fscores, micro_fscore)


class TestF1MeasureSeq2Rel(FBetaMeasureSeq2RelTestCase):
    """Tests for F1MeasureSeq2Rel. Because F1MeasureSeq2Rel is a just a wrapper on
    FBetaMeasureSeq2Rel and introduces no new logic, this exists mainly just to ensure we
    can instantiate F1MeasureSeq2Rel without error.
    """

    def setup_method(self):
        super().setup_method()

    def test_f1_seq2rel(
        self,
    ):
        fbeta = F1MeasureSeq2Rel(labels=self.labels)
        assert fbeta._beta == 1.0

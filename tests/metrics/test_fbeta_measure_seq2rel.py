from typing import List

import pytest
import torch
from hypothesis import given
from hypothesis.strategies import floats
from seq2rel.metrics.fbeta_measure_seq2rel import (
    F1MeasureSeq2Rel,
    FBetaMeasureSeq2Rel,
    _fuzzy_cluster_match,
)
from torch.testing import assert_allclose


def test_fuzzy_cluster_match() -> None:
    threshold = 0.5
    # Missing an entire cluster
    pred_rel = (
        # 0 / 1, NOT over threshold
        (("arbitrary"), "CHEMICAL"),
        # 2 / 2, over threshold
        (("fasciculations", "fasciculation"), "DISEASE"),
    )
    gold_rels = [
        (
            (("suxamethonium chloride", "suxamethonium", "sch"), "CHEMICAL"),
            (("fasciculations", "fasciculation"), "DISEASE"),
        )
    ]
    assert not _fuzzy_cluster_match(pred_rel, gold_rels, threshold=threshold)
    # Two additional mentions in each cluster
    pred_rel = (
        # 3 / 5, over threshold
        (("suxamethonium chloride", "suxamethonium", "sch", "wrong", "incorrect"), "CHEMICAL"),
        # 2 / 4, NOT the threshold
        (("fasciculations", "fasciculation", "wrong", "incorrect"), "DISEASE"),
    )
    gold_rels = [
        (
            (("suxamethonium chloride", "suxamethonium", "sch"), "CHEMICAL"),
            (("fasciculations", "fasciculation"), "DISEASE"),
        )
    ]
    assert not _fuzzy_cluster_match(pred_rel, gold_rels, threshold=threshold)
    # Missing a single mention in each cluster
    pred_rel = (
        # 2 / 3, over threshold
        (("suxamethonium chloride", "suxamethonium"), "CHEMICAL"),
        # 1 / 1, over threshold
        (("fasciculations",), "DISEASE"),
    )
    gold_rels = [
        (
            (("suxamethonium chloride", "suxamethonium", "sch"), "CHEMICAL"),
            (("fasciculations", "fasciculation"), "DISEASE"),
        )
    ]
    assert _fuzzy_cluster_match(pred_rel, gold_rels, threshold=threshold)
    # One additional mention in each cluster
    pred_rel = (
        # 2 / 3, over threshold
        (("suxamethonium chloride", "suxamethonium", "arbitrary"), "CHEMICAL"),
        # 2 / 2, over threshold
        (("fasciculations", "fasciculation", "arbitrary"), "DISEASE"),
    )
    gold_rels = [
        (
            (("suxamethonium chloride", "suxamethonium", "sch"), "CHEMICAL"),
            (("fasciculations", "fasciculation"), "DISEASE"),
        )
    ]
    assert _fuzzy_cluster_match(pred_rel, gold_rels, threshold=threshold)


class FBetaMeasureSeq2RelTestCase:
    def setup_method(self):
        self.labels = ["PHYSICAL", "GENETIC"]
        self.predictions = [
            # This is a false positive
            "@PHYSICAL@ fgf-2 @GGP@ rps19 @GGP@ @EOR@",
            "I don't contain anything of interest!",
            # This prediction contains a relation with an incorrect order of entities
            "@PHYSICAL@ atg1 @GGP@ atg1 @GGP@ @EOR@ @PHYSICAL@ atg17 @GGP@ atg1 @GGP@ @EOR@",
            # This prediction is missing a relation
            "@GENETIC@ b-myb @GGP@ cbp @GGP@ @EOR@ @PHYSICAL@ b-myb @GGP@ cbp @GGP@ @EOR@",
        ]
        self.targets = [
            "",
            "I don't contain anything of interest!",
            "@PHYSICAL@ atg1 @GGP@ atg1 @GGP@ @EOR@ @PHYSICAL@ atg1 @GGP@ atg17 @GGP@ @EOR@",
            (
                "@GENETIC@ b-myb @GGP@ cbp @GGP@ @EOR@"
                " @PHYSICAL@ b-myb @GGP@ cbp @GGP@ @EOR@"
                " @GENETIC@ myb @GGP@ cbp @GGP@ @EOR@"
            ),
        ]

        # detailed target state
        self.pred_sum = [4, 1]
        self.true_sum = [3, 2]
        self.true_positive_sum = [2, 1]
        self.total_sum = [3, 2]

        desired_precisions = [2 / 4, 1.00]
        desired_recalls = [2 / 3, 1 / 2]
        desired_fscores = [
            (2 * p * r) / (p + r) if p + r != 0.0 else 0.0
            for p, r in zip(desired_precisions, desired_recalls)
        ]
        self.desired_precisions = desired_precisions
        self.desired_recalls = desired_recalls
        self.desired_fscores = desired_fscores


class TestFBetaMeasureSeq2Rel(FBetaMeasureSeq2RelTestCase):
    """Tests for FBetaMeasureSeq2Rel. Note that for now, these tests assume
    that entities in a relation have an inherent order.

    Loosely based on: https://github.com/allenai/allennlp/blob/main/tests/training/metrics/fbeta_measure_test.py
    """

    def setup_method(self):
        super().setup_method()

    @given(cluster_threshold=floats(min_value=-1, max_value=1))
    def test_fbeta_seq2rel_invalid_cluster_threshold_raises_value_error(
        self, cluster_threshold: float
    ):
        if cluster_threshold <= 0.0 or cluster_threshold > 1.0:
            with pytest.raises(ValueError):
                _ = FBetaMeasureSeq2Rel(labels=self.labels, cluster_threshold=cluster_threshold)
        # Sanity check that valid values don't raise an error.
        else:
            _ = FBetaMeasureSeq2Rel(labels=self.labels, cluster_threshold=cluster_threshold)

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
        true_positives = torch.tensor([2, 1], dtype=torch.float32)
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

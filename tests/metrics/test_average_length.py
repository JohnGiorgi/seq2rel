from seq2rel.metrics.average_length import AverageLength


class AverageLengthTestCase:
    def setup_method(self):
        self.predictions = ["They may take us, but they’ll never take our freedom!".split()]
        self.targets = ["They may take our lives, but they’ll never take our freedom!".split()]
        self.predictions_mean_length = round(len(self.predictions[0]) / len(self.predictions), 2)
        self.targets_mean_length = round(len(self.targets[0]) / len(self.targets), 2)
        self.predictions_to_targets_length_ratio = round(
            self.predictions_mean_length / self.targets_mean_length, 2
        )


class TestAverageLength(AverageLengthTestCase):
    def setup_method(self):
        super().setup_method()

    def test_average_length(self):
        metric = AverageLength()
        metric(self.predictions, self.targets)
        expected = {
            "predictions_mean_length": self.predictions_mean_length,
            "targets_mean_length": self.targets_mean_length,
            "predictions_to_targets_length_ratio": self.predictions_to_targets_length_ratio,
        }
        actual = metric.get_metric(reset=True)
        assert actual == expected

        # Test case where predictions are empty.
        predictions = [[]]
        metric(predictions, self.targets)
        expected = {
            "predictions_mean_length": 0,
            "targets_mean_length": self.targets_mean_length,
            "predictions_to_targets_length_ratio": round(0 / self.targets_mean_length, 2),
        }
        actual = metric.get_metric(reset=True)
        assert actual == expected

        # Test case where targets are empty.
        targets = [[]]
        metric(self.predictions, targets)
        expected = {
            "predictions_mean_length": self.predictions_mean_length,
            "targets_mean_length": 0,
            "predictions_to_targets_length_ratio": 0,
        }
        actual = metric.get_metric(reset=True)
        assert actual == expected

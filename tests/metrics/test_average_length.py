from seq2rel.metrics.average_length import AverageLength


class AverageLengthTestCase:
    def setup_method(self):
        self.predictions = ["They may take us, but they’ll never take our freedom!".split()]
        self.targets = ["They may take our lives, but they’ll never take our freedom!".split()]
        self.decoded_mean_length = len(self.predictions[0]) / len(self.predictions)
        self.target_mean_length = len(self.targets[0]) / len(self.targets)


class TestAverageLength(AverageLengthTestCase):
    def setup_method(self):
        super().setup_method()

    def test_average_length(self):
        average_length = AverageLength()
        average_length(self.predictions, self.targets)
        expected = {
            "decoded_mean_length": round(self.decoded_mean_length, 2),
            "target_mean_length": round(self.target_mean_length, 2),
            "ratio": round(self.decoded_mean_length / self.target_mean_length, 2),
        }
        actual = average_length.get_metric()
        assert actual == expected

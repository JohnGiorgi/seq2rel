from seq2rel.metrics.valid_sequences import ValidSequences


class ValidSequencesTestCase:
    def setup_method(self):
        self.predictions = [
            # Valid (with 2 relations of different types)
            "@CID@ lithium carbonate @CHEMICAL@ neurologic depression @DISEASE@ @EOR@"
            " @GDA@ prothrombin @GENE@ thrombophilia @DISEASE@ @EOR@",
            # Invalid: missing a second entity
            "@CID@ lithium carbonate @CHEMICAL@ @EOR@",
            # Invalid: contains no entities
            "@CID@ @EOR@",
            # Invalid: does not contain any of the special tokens
            "this is arbitrary",
        ]
        self.num_predicted_relations = 5
        self.num_valid_predicted_relations = 2
        self.num_predicted_to_valid_relations_ratio = round(
            self.num_predicted_relations / self.num_valid_predicted_relations, 2
        )


class TestValidSequences(ValidSequencesTestCase):
    def setup_method(self):
        super().setup_method()

    def test_valid_sequences(self):
        metric = ValidSequences()
        metric(self.predictions)
        expected = {
            "num_predicted_relations": self.num_predicted_relations,
            "num_valid_predicted_relations": self.num_valid_predicted_relations,
            "num_predicted_to_valid_relations_ratio": self.num_predicted_to_valid_relations_ratio,
        }
        actual = metric.get_metric()
        assert actual == expected

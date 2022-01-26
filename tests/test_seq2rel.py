from seq2rel import __version__
from seq2rel import Seq2Rel


def test_version():
    assert __version__ == "0.1.0"


class TestSeq2Rel:
    def test_bc5cdr_model(self):
        model = Seq2Rel("bc5cdr")

        # These are roughly organized in order of increasing difficulty.
        texts = [
            "Famotidine-associated delirium.",
            "Acute hepatitis associated with clopidogrel: a case report and review of the literature.",
            "Acute interstitial nephritis due to nicergoline (Sermion).",
        ]
        expected = [
            "famotidine @CHEMICAL@ delirium @DISEASE@ @CID@",
            "clopidogrel @CHEMICAL@ hepatitis @DISEASE@ @CID@",
            "nicergoline ; sermion @CHEMICAL@ interstitial nephritis @DISEASE@ @CID@",
        ]
        actual = model(texts)
        assert actual == expected

    def test_gda_model(self):
        model = Seq2Rel("gda")

        # These are roughly organized in order of increasing difficulty.
        texts = [
            "Lung disease associated with the IVS8 5T allele of the CFTR gene.",
            "Variations in the monoamine oxidase B (MAOB) gene are associated with Parkinson's disease (PD).",
            (
                "Association of essential hypertension in elderly Japanese with I/D polymorphism of"
                " the angiotensin-converting enzyme (ACE) gene."
            ),
        ]
        expected = [
            "cftr @GENE@ lung disease @DISEASE@ @GDA@",
            "monoamine oxidase b ; maob @GENE@ parkinson's disease ; pd @DISEASE@ @GDA@",
            "angiotensin - converting enzyme ; ace @GENE@ hypertension @DISEASE@ @GDA@",
        ]
        actual = model(texts)
        assert actual == expected

    def test_docred_model(self):
        model = Seq2Rel("docred")

        # Includes a negative example where the model should generate the empty string.
        texts = [
            "Acute hepatitis associated with clopidogrel: a case report and review of the literature.",
            'Ernest Julian "Ernie" Cole (1916 – November 9 , 2000) was an engineer and politician'
            " in Saskatchewan, Canada.",
        ]

        expected = [
            "",
            'ernest julian " ernie " cole @PER@ 1916 @TIME@ @DATE_OF_BIRTH@'
            ' ernest julian " ernie " cole @PER@ november 9, 2000 @TIME@ @DATE_OF_DEATH@'
            ' ernest julian " ernie " cole @PER@ canada @LOC@ @COUNTRY_OF_CITIZENSHIP@'
            " saskatchewan @LOC@ canada @LOC@ @LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY@"
            " saskatchewan @LOC@ canada @LOC@ @COUNTRY@"
            " canada @LOC@ saskatchewan @LOC@ @CONTAINS_ADMINISTRATIVE_TERRITORIAL_ENTITY@",
        ]
        actual = model(texts)
        assert actual == expected

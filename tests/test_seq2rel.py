from pathlib import Path

from seq2rel import Seq2Rel, __version__


def test_version():
    assert __version__ == "0.1.0"


class TestSeq2Rel:
    def test_cdr_model(self):
        model = Seq2Rel("cdr")

        # These are roughly organized in order of increasing difficulty.
        texts = [
            "Famotidine-associated delirium.",
            "Acute hepatitis associated with clopidogrel: a case report and review of the literature.",
            "Acute interstitial nephritis due to nicergoline (Sermion).",
        ]
        expected = [
            "famotidine @CHEMICAL@ delirium @DISEASE@ @CID@",
            "clopidogrel @CHEMICAL@ hepatitis @DISEASE@ @CID@",
            "nicergoline ; sermion @CHEMICAL@ acute interstitial nephritis @DISEASE@ @CID@",
        ]
        actual = model(texts)
        assert actual == expected

        # Test that we can provide a string as input.
        actual = model(texts[0])
        assert actual[0] == expected[0]

        # Test that we can provide a filepath as input
        input_fp = str(
            (Path(__file__).parent / ".." / "test_fixtures" / "data").resolve() / "test.txt"
        )
        actual = model(input_fp)
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

    def test_dgm_model(self):
        model = Seq2Rel("dgm")

        # These are roughly organized in order of increasing difficulty.
        texts = [
            (
                "The tumor was found to harbor an epidermal growth factor receptor (EGFR) mutation of"
                " L858R, so he was subsequently treated with gefitinib (250 mg/day) and his general"
                " condition immediately improved."
            ),
            (
                "Recently, the addition of cetuximab to afatinib has yielded impressive results in the"
                " treatment of EGFR reversible TKI resistant lung cancer due to T790M mutation."
            ),
            (
                "Imatinib inhibition of KIT phosphorylation in mast cell lines correlates with the"
                " inhibition of cellular proliferation (Zermati et al, 2003), suggesting that the"
                " cells are critically dependent on KIT activity. In systemic mastocytosis, the"
                " most common activating mutation in C-KIT D816V, occurs in the catalytic region of"
                " the enzyme, and also prevents binding of imatinib."
            ),
        ]
        expected = [
            "gefitinib @DRUG@ epidermal growth factor receptor ( egfr ) ; egfr @GENE@ l858r @VARIANT@ @DGM@",
            "cetuximab @DRUG@ egfr @GENE@ t790m @VARIANT@ @DGM@ afatinib @DRUG@ egfr @GENE@ t790m @VARIANT@ @DGM@",
            "imatinib @DRUG@ kit ; c - kit @GENE@ d816v @VARIANT@ @DGM@",
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
            # The model duplicates this output
            ' ernest julian " ernie " cole @PER@ canada @LOC@ @COUNTRY_OF_CITIZENSHIP@'
            ' ernest julian " ernie " cole @PER@ canada @LOC@ @COUNTRY_OF_CITIZENSHIP@'
            " saskatchewan @LOC@ canada @LOC@ @LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY@"
            " saskatchewan @LOC@ canada @LOC@ @COUNTRY@"
            " canada @LOC@ saskatchewan @LOC@ @CONTAINS_ADMINISTRATIVE_TERRITORIAL_ENTITY@",
        ]
        actual = model(texts)
        assert actual == expected

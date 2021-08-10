from typing import List, Tuple

import pytest
from seq2rel import Seq2Rel

PRETRAINED_ADE_MODEL = "ade"


@pytest.fixture(scope="module")
def pretrained_ade_model() -> Seq2Rel:
    # TODO: The overrides exists because this model was trained before the dataset reader rename.
    # Remove when this is updated to a newer pretrained model.
    return Seq2Rel(PRETRAINED_ADE_MODEL, overrides={"dataset_reader.type": "seq2rel"})


@pytest.fixture()
def ade_examples() -> Tuple[List[str], List[str]]:
    texts = [
        "Vincristine induced cranial polyneuropathy.",
        "Intravenous diazepam exacerbated the seizures.",
        "Acute myocardial infarction due to coronary spasm associated with L-thyroxine therapy.",
    ]
    annotations = [
        "@ADE@ vincristine @DRUG@ cranial polyneuropathy @EFFECT@ @EOR@",
        "@ADE@ diazepam @DRUG@ seizures @EFFECT@ @EOR@",
        (
            "@ADE@ l - thyroxine @DRUG@ acute myocardial infarction @EFFECT@ @EOR@"
            " @ADE@ l - thyroxine @DRUG@ coronary spasm @EFFECT@ @EOR@"
        ),
    ]
    return texts, annotations

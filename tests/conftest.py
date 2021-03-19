from typing import List, Tuple

import pytest
from seq2rel import Seq2Rel

PRETRAINED_ADE_MODEL = "ade"


@pytest.fixture(scope="module")
def pretrained_ade_model() -> Seq2Rel:
    return Seq2Rel(PRETRAINED_ADE_MODEL)


@pytest.fixture()
def ade_examples() -> Tuple[List[str], List[str]]:
    texts = [
        "Vincristine induced cranial polyneuropathy.",
        "Intravenous diazepam exacerbated the seizures.",
        "Acute myocardial infarction due to coronary spasm associated with L-thyroxine therapy.",
    ]
    annotations = [
        "<ADE> vincristine <DRUG> cranial polyneuropathy <EFFECT> </ADE>",
        "<ADE> diazepam <DRUG> seizures <EFFECT> </ADE>",
        "<ADE> l - thyroxine <DRUG> acute myocardial infarction <EFFECT> </ADE> <ADE> l - thyroxine <DRUG> coronary spasm <EFFECT> </ADE>",
    ]
    return texts, annotations

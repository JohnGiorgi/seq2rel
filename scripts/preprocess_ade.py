from typing import Any, Iterable, List, Tuple, Optional

from pathlib import Path

import typer
from allennlp.common.file_utils import cached_path
from seq2rel.common.util import ENT_TYPE, REL_START, REL_END
from sklearn.model_selection import train_test_split

BASE_URL = "https://raw.githubusercontent.com/trunghlt/AdverseDrugReaction/master/ADE-Corpus-V2/"
DRUG_AE_URL = BASE_URL + "DRUG-AE.rel"
DRUG_DOSE_URL = BASE_URL + "DRUG-DOSE.rel"

ADR_REL_START = REL_START.format("ADR")
ADR_REL_END = REL_END.format("ADR")
DD_REL_START = REL_START.format("DD")
DD_REL_END = REL_END.format("DD")
DRUG_ENT = ENT_TYPE.format("DRUG")
ADVERSE_EVENT_ENT = ENT_TYPE.format("ADE")
DRUG_DOSE = ENT_TYPE.format("DOSE")

app = typer.Typer()


def format_adverse_drug_reaction(ent_1: str, ent_2: str) -> str:
    return f"{ADR_REL_START} {ent_1} {ADVERSE_EVENT_ENT} {ent_2} {DRUG_ENT} {ADR_REL_END}"


def format_drug_dose(ent_1: str, ent_2: str) -> str:
    return f"{DD_REL_START} {ent_1} {DRUG_DOSE} {ent_2} {DRUG_ENT} {DD_REL_END}"


def parse_ade_v2(filepath: str, rel_type: str) -> List[str]:
    lines = Path(filepath).read_text().split("\n")
    parsed_content = {}
    for line in lines:
        if not line:
            continue
        _, text, ent_1, _, _, ent_2, _, _ = line.strip().split("|")
        formatted_relation = (
            format_adverse_drug_reaction(ent_1, ent_2)
            if rel_type == "ADR"
            else format_drug_dose(ent_1, ent_2)
        )

        if text in parsed_content:
            if formatted_relation in parsed_content[text]:
                continue
            parsed_content[text].append(formatted_relation)
        else:
            parsed_content[text] = [formatted_relation]
    formatted_content = [
        f"{key}\t{' '.join(sorted(value))}" for key, value in parsed_content.items()
    ]
    return formatted_content


def train_valid_test_split(
    data: Iterable[Any],
    train_size: int = 0.7,
    valid_size: int = 0.1,
    test_size: int = 0.2,
    stratify: Optional[Iterable[Any]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    # https://datascience.stackexchange.com/a/53161
    X_train, X_test = train_test_split(data, test_size=1 - train_size, stratify=stratify)
    X_valid, X_test = train_test_split(X_test, test_size=test_size / (test_size + valid_size))
    return X_train, X_valid, X_test


@app.command()
def main(output_dir: str, binary: bool = True, sorting: Optional[str] = None) -> None:
    drug_ae_parsed = parse_ade_v2(cached_path(DRUG_AE_URL), rel_type="ADR")
    # drug_dose_parsed = parse_ade_v2(cached_path(DRUG_DOSE_URL), rel_type="DOSE")
    # parsed_content = drug_ae_parsed + drug_dose_parsed
    parsed_content = drug_ae_parsed

    # stratify = [0] * len(drug_ae_parsed) + [1] * len(drug_dose_parsed)
    # X_train, X_valid, X_test = train_valid_test_split(parsed_content, stratify=stratify)
    X_train, X_valid, X_test = train_valid_test_split(parsed_content)

    output_dir: Path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.tsv").write_text("\n".join(X_train))
    (output_dir / "valid.tsv").write_text("\n".join(X_valid))
    (output_dir / "test.tsv").write_text("\n".join(X_test))


if __name__ == "__main__":
    app()

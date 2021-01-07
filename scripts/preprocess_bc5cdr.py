from typing import List

from pathlib import Path

import typer
from seq2rel.common.util import ENT_TYPE, REL_START, REL_END

CHEMICAL = ENT_TYPE.format("CHEMICAL")
DISEASE = ENT_TYPE.format("DISEASE")

app = typer.Typer()


def parse_bc5cdr(filepath: str) -> List[str]:
    chunks = Path(filepath).read_text().split("\n\n")

    parsed_content = {}
    for chunk in chunks:
        if not chunk:
            continue
        lines = chunk.split("\n")
        title = lines[0].split("|")[-1].strip()
        abstract = lines[1].split("|")[-1].strip()
        text = f"{title} {abstract}"
        formatted_ents = {}
        for line in lines[2:]:
            if not line:
                continue
            ann = line.split("\t")
            if len(ann) > 4:  # this is an entity
                start, _, ent_text, ent_type, ent_ids = ann[2:7]
                ent_ids = ent_id.split("|")
                for ent_id in ent_ids:
                    if ent_id in formatted_ents:
                        continue
                    formatted_ents[ent_id] = (
                        f"{ent_text} {ENT_TYPE.format(ent_type.upper())}",
                        start,
                    )
            else:  # this is a relation
                rel_type, ent_1_id, ent_2_id = ann[1:4]
                (
                    ent_1_text,
                    ent_1_start,
                ) = formatted_ents[ent_1_id]
                (
                    ent_2_text,
                    ent_2_start,
                ) = formatted_ents[ent_2_id]

                formatted_rel = f"{REL_START.format(rel_type)} {ent_1_text} {ent_2_text} {REL_END.format(rel_type)}"
                parsed_content[text].append((formatted_relation, ent_1_start + ent_2_start))

    formatted_content = []
    for key, value in parsed_content.items():
        value.sort(key=itemgetter(-1))
        value = " ".join([x[0] for x in value])
        formatted_content.append(f"{key}\t{value}")
    return formatted_content


@app.command()
def main(input_dir: str, output_dir: str) -> None:
    train_filepath = Path(input_dir) / "CDR_TrainingSet.PubTator.txt"
    dev_filepath = Path(input_dir) / "CDR_DevelopmentSet.PubTator.txt"
    test_filepath = Path(input_dir) / "CDR_TestSet.PubTator.txt"

    parsed_content_train = parse_bc5cdr(train_filepath)
    parsed_content_dev = parse_bc5cdr(dev_filepath)
    parsed_content_test = parse_bc5cdr(test_filepath)

    output_dir: Path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.tsv").write_text("\n".join(parsed_content_train))
    (output_dir / "valid.tsv").write_text("\n".join(parsed_content_dev))
    (output_dir / "test.tsv").write_text("\n".join(parsed_content_test))


if __name__ == "__main__":
    app()

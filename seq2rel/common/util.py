import re
from typing import Any, Dict, List, Union

END_OF_REL_SYMBOL = "@EOR@"
COREF_SEP_SYMBOL = ";"
ENT_PATTERN = re.compile(r"(?:\s?)(.*?)(?:\s?)@([^\s]*)\b[^@]*@")
REL_PATTERN = re.compile(fr"@([^\s]*)\b[^@]*@(.*?){END_OF_REL_SYMBOL}")


def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text


def deserialize_annotations(
    serialized_annotations: Union[str, List[str]],
) -> List[Dict[str, Any]]:
    """Returns dictionaries containing the entities and the relations that are present in the
    `serialized_annotations`, the string serialized representation of entities and relations.

    # Parameters

    serialized_annotations: `list`
        A list containing the string serialized representation of entities and relations.

    # Returns

    A list of dictionaries, keyed by class name, containing the relations of the string
    serialized representations `serialized_strings`.
    """
    if isinstance(serialized_annotations, str):
        serialized_annotations = [serialized_annotations]

    deserialized: List[Dict[str, Any]] = []
    for annotation in serialized_annotations:
        deserialized.append({})
        rels = REL_PATTERN.findall(annotation)
        for rel in rels:
            rel_label, rel_string = rel
            ents = tuple(ENT_PATTERN.findall(rel_string))
            # TODO. We can enforce order by casting the entities as a tuple.
            # ents = tuple(ents) if self._ordered_ents else set(ents)
            if rel_label in deserialized[-1]:
                # Don't retain duplicates
                if ents not in deserialized[-1][rel_label]:
                    deserialized[-1][rel_label].append(ents)
            else:
                deserialized[-1][rel_label] = [ents]
    return deserialized

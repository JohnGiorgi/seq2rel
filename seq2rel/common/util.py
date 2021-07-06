import re
from typing import Dict, List, Tuple, Union

END_OF_REL_SYMBOL = "@EOR@"
COREF_SEP_SYMBOL = ";"
REL_PATTERN = re.compile(fr"@([^\s]*)\b[^@]*@(.*?){END_OF_REL_SYMBOL}")
CLUSTER_PATTERN = re.compile(r"(?:\s?)(.*?)(?:\s?)@([^\s]*)\b[^@]*@")

# Custom annotation types
ClusterAnnotation = Tuple[Tuple[str, ...], str]
EntityAnnotation = Tuple[ClusterAnnotation, ...]
RelationAnnotation = Dict[str, List[EntityAnnotation]]


# Public functions #


def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text


def deserialize_annotations(
    serialized_annotations: Union[str, List[str]],
) -> List[RelationAnnotation]:
    """Returns dictionaries containing the entities and relations present in
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

    deserialized: List[RelationAnnotation] = []
    for annotation in serialized_annotations:
        deserialized.append({})
        rels = REL_PATTERN.findall(annotation)
        for rel_label, rel_string in rels:
            raw_clusters = tuple(CLUSTER_PATTERN.findall(rel_string))
            # Normalizes clusters so that evaluation is insensitive to order, case and duplicates.
            clusters = _normalize_clusters(raw_clusters)  # type: ignore
            if rel_label in deserialized[-1]:
                # Don't retain duplicates
                if clusters not in deserialized[-1][rel_label]:
                    deserialized[-1][rel_label].append(clusters)
            else:
                deserialized[-1][rel_label] = [clusters]
    return deserialized


# Private functions #


def _normalize_clusters(clusters: ClusterAnnotation) -> EntityAnnotation:
    """Normalize clusters (coreferent mentions) by sorting mentions, removing duplicates, and
    lowercasing the text."""
    preprocessed_clusters = tuple(
        # Evaluation is insensitive to...
        (
            tuple(
                # ...duplicate mentions
                dict.fromkeys(
                    # ...order
                    sorted(
                        # ...case
                        (coref.strip().lower() for coref in mentions.split(COREF_SEP_SYMBOL)),
                        key=len,
                        reverse=True,
                    )
                ),
            ),
            label,
        )
        for mentions, label in clusters
    )
    return preprocessed_clusters

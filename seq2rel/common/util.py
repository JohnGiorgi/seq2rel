import re
from typing import Dict, List, Optional, Tuple, Union

# Used to separate coreferent mentions in a linearized relation string.
COREF_SEP_SYMBOL = ";"
# Used to separate entity hints from input text in the source string.
HINT_SEP_SYMBOL = "@HINTS@"
# Regex patterns used to parse the string serialized representation of entities and relations.
REL_PATTERN = re.compile(r"(.*?@)\s*(?:@)([^\s]*)\b[^@]*@")
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
    ordered_ents: bool = False,
    remove_duplicate_ents: bool = False,
    filtered_relations: Optional[Union[str, List[str]]] = None,
) -> List[RelationAnnotation]:
    """Returns dictionaries containing the entities and relations present in
    `serialized_annotations`, the string serialized representation of entities and relations.

    # Parameters

    serialized_annotations: `list`
        A list containing the string serialized representation of entities and relations.
    ordered_ents : `bool`, optional (default = `False`)
        True if the entities should be considered ordered (e.g. there are distinct head and tail
        entities). Defaults to False.
    remove_duplicate_ents : `bool`, optional (default = `False`)
        True if non-unique entities within a relation should be removed. These are not common so
        removing them can improve performance. However, in some domains they are possible
        (e.g. homodimers in protein-protein interactions). Defaults to False.

    # Returns

    A list of dictionaries, keyed by relation class name, containing the relations of the string
    serialized representations `serialized_strings`.
    """
    if isinstance(serialized_annotations, str):
        serialized_annotations = [serialized_annotations]
    if isinstance(filtered_relations, str):
        filtered_relations = [filtered_relations]

    deserialized: List[RelationAnnotation] = []
    for i, annotation in enumerate(serialized_annotations):
        deserialized.append({})
        rels = REL_PATTERN.findall(annotation)
        # If filtered_relations was provided, we remove them from the given relations.
        if filtered_relations:
            for filtered_rel in REL_PATTERN.findall(filtered_relations[i]):
                rels.remove(filtered_rel)

        for rel_string, rel_label in rels:
            raw_clusters = tuple(CLUSTER_PATTERN.findall(rel_string))
            # Normalizes entity mentions so that evaluation is insensitive to order, case and duplicates.
            clusters = _normalize_clusters(
                raw_clusters, remove_duplicate_ents=remove_duplicate_ents
            )  # type: ignore
            # Optional sort the entities to make evaluation insensitive to order.
            if not ordered_ents:
                clusters = tuple(sorted(clusters))
            # A relation must contain at least two entities. These are easy to detect at training
            # and at inference, so we purposfully drop them.
            if len(clusters) < 2:
                continue
            if rel_label in deserialized[-1]:
                # Don't retain duplicate relations
                if clusters not in deserialized[-1][rel_label]:
                    deserialized[-1][rel_label].append(clusters)
            else:
                deserialized[-1][rel_label] = [clusters]
    return deserialized


# Private functions #


def _normalize_clusters(
    clusters: Tuple[Tuple[str, str], ...], remove_duplicate_ents: bool = False
) -> EntityAnnotation:
    """Normalize clusters (coreferent mentions) by sorting mentions, removing duplicates, and
    lowercasing the text."""
    preprocessed_clusters = tuple(
        # Evaluation is insensitive to...
        (
            tuple(
                # ...order
                sorted(
                    # ...duplicate mentions
                    dict.fromkeys(
                        # ...case
                        (
                            coref.strip().lower()
                            for coref in mentions.split(COREF_SEP_SYMBOL)
                            if coref.strip()
                        )
                    ),
                    key=len,
                    reverse=True,
                ),
            ),
            label,
        )
        for mentions, label in clusters
    )
    # Drop clusters with no predicted mentions
    preprocessed_clusters = tuple(
        (mentions, label) for mentions, label in preprocessed_clusters if mentions
    )
    # Optionally remove duplicate clusters
    if remove_duplicate_ents:
        preprocessed_clusters = tuple(dict.fromkeys(preprocessed_clusters))
    return preprocessed_clusters

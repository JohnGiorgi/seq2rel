import re
from itertools import zip_longest
from typing import Dict, List, Optional, Tuple

# Used to separate coreferent mentions in a linearized relation string.
COREF_SEP_SYMBOL = ";"
# Used to separate entity hints from input text in the source string.
HINT_SEP_SYMBOL = "@HINTS@"
# Regex patterns used to parse the string serialized representation of entities and relations.
REL_PATTERN = re.compile(r"(.*?@)\s*(?:@)([^\s]*)\b[^@]*@")
ENT_PATTERN = re.compile(r"(?:\s?)(.*?)(?:\s?)@([^\s]*)\b[^@]*@")

# Custom annotation types
MentionAnnotation = Tuple[Tuple[str, ...], str]
EntityAnnotation = Tuple[MentionAnnotation, ...]
RelationAnnotation = Dict[str, List[EntityAnnotation]]


def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text


def extract_entities(
    linearization: str, ordered_ents: bool = False, remove_duplicate_ents: bool = False
) -> EntityAnnotation:
    """Given a linearized relation string `linearization`, extracts the entities and
    returns them as a tuple of tuples, where the inner tuples contain an entities mentions and its
    label.

    # Parameters

    linearization: `str`
        A string containing a linearized relation string.
    ordered_ents : `bool`, optional (default = `False`)
        True if entities should be considered ordered (e.g. there are distinct head and tail entities).
        Defaults to False.
    remove_duplicate_ents : `bool`, optional (default = `False`)
        True if non-unique entities should be removed. These are not common, so
        removing them can improve performance. However, in some domains, they are possible
        (e.g. homodimers in protein-protein interactions). Defaults to False.

    # Returns

    A tuple of tuples, where the inner tuples contain an entities mentions and its label.
    """
    entities = tuple(ENT_PATTERN.findall(linearization))
    # Normalizes entity mentions so that evaluation is insensitive to...
    entities = tuple(
        (
            tuple(
                # ...mention order
                sorted(
                    # ...duplicate mentions
                    dict.fromkeys(
                        # ...case
                        (
                            mention.strip().lower()
                            for mention in mentions.split(COREF_SEP_SYMBOL)
                            if mention.strip()
                        )
                    ),
                    key=len,
                    reverse=True,
                ),
            ),
            label,
        )
        for mentions, label in entities
        # Additionaly, drop entities with no predicted mentions.
        if mentions
    )
    # Optionally remove duplicate entities.
    if remove_duplicate_ents:
        entities = tuple(dict.fromkeys(entities))
    # Optionally sort the entities to make evaluation insensitive to order.
    if not ordered_ents:
        entities = tuple(sorted(entities))
    return entities


def extract_relations(
    linearizations: List[str],
    ordered_ents: bool = False,
    remove_duplicate_ents: bool = False,
    filtered_relations: Optional[List[str]] = None,
) -> List[RelationAnnotation]:
    """Given a batch of linearized relation strings `linearizations`, extracts the relations and
    returns them as a list of dictionaries, where the keys are relation labels, and the values are
    lists of relations belonging to that label.

    # Parameters

    linearizations: `list`
        A list containing a batch of linearized relation strings.
    ordered_ents : `bool`, optional (default = `False`)
        True if entities should be considered ordered (e.g. there are distinct head and tail entities).
        Defaults to False.
    remove_duplicate_ents : `bool`, optional (default = `False`)
        True if non-unique entities within a relation should be removed. These are not common, so
        removing them can improve performance. However, in some domains, they are possible
        (e.g. homodimers in protein-protein interactions). Defaults to False.
    filtered_relations : `list`, optional (default = `None`)
        A list containing a batch of linearized relation strings in the same format as
        `linearizations`. If provided, these relations will be excluded from the output.
        Defaults to None.

    # Returns

    A list of dictionaries, keyed by the relation label name, containing the relations extracted
    from the linearized relation strings `linearizations`.
    """
    if filtered_relations is not None:
        if len(linearizations) != len(filtered_relations):
            raise ValueError(
                "Arguments 'linearizations' and 'filtered_relations' to"
                " 'seq2rel.common.util.extract_relations' must be the same length. Got"
                f" {len(linearizations)} and {len(filtered_relations)} respectively."
            )
    else:
        filtered_relations = []

    extracted_relations: List[RelationAnnotation] = []

    for linearization, filtered in zip_longest(linearizations, filtered_relations):
        extracted_relations.append({})
        # Extract relations from the linearized relation string.
        relations = REL_PATTERN.findall(linearization)
        for rel_string, rel_label in relations:
            entities = extract_entities(
                rel_string, ordered_ents=ordered_ents, remove_duplicate_ents=remove_duplicate_ents
            )
            # A relation must contain at least two entities. These are easy to detect at training
            # and at inference, so we purposfully drop them.
            if len(entities) < 2:
                continue
            # Accumulate unique relations
            if rel_label not in extracted_relations[-1]:
                extracted_relations[-1][rel_label] = []
            if entities not in extracted_relations[-1][rel_label]:
                extracted_relations[-1][rel_label].append(entities)
        # If a filtered relation is provided, remove it from the output.
        if filtered is not None:
            relations = REL_PATTERN.findall(filtered)
            for rel_string, rel_label in relations:
                entities = extract_entities(
                    rel_string,
                    ordered_ents=ordered_ents,
                    remove_duplicate_ents=remove_duplicate_ents,
                )
                if entities in extracted_relations[-1].get(rel_label, []):
                    extracted_relations[-1][rel_label].remove(entities)

    return extracted_relations

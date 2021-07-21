// This config contains anything that doesn"t change across experiments and datasets
local COMMON = import "transformer_copynet_common.libsonnet";

// ** THESE MUST BE SET BY THE USER **//
// A list containing the special tokens in your target vocabulary
local special_tokens = [
    "@TIME@",
    "@ORG@",
    "@PER@",
    "@NUM@",
    "@LOC@",
    "@MISC@",
    "@SCREENWRITER@",
    "@DATE_OF_BIRTH@",
    "@AWARD_RECEIVED@",
    "@LOCATED_IN_OR_NEXT_TO_BODY_OF_WATER@",
    "@CHAIRPERSON@",
    "@FATHER@",
    "@POINT_IN_TIME@",
    "@MANUFACTURER@",
    "@DATE_OF_DEATH@",
    "@CONFLICT@",
    "@TERRITORY_CLAIMED_BY@",
    "@HAS_PART@",
    "@CAST_MEMBER@",
    "@LANGUAGES_SPOKEN_WRITTEN_OR_SIGNED@",
    "@DEVELOPER@",
    "@FOUNDED_BY@",
    "@BASIN_COUNTRY@",
    "@DIRECTOR@",
    "@COUNTRY_OF_CITIZENSHIP@",
    "@EMPLOYER@",
    "@PUBLISHER@",
    "@PLACE_OF_BIRTH@",
    "@CREATOR@",
    "@MILITARY_BRANCH@",
    "@PRODUCTION_COMPANY@",
    "@START_TIME@",
    "@PRODUCER@",
    "@WORK_LOCATION@",
    "@INSTANCE_OF@",
    "@REPLACES@",
    "@MOUTH_OF_THE_WATERCOURSE@",
    "@REPLACED_BY@",
    "@LEAGUE@",
    "@CHARACTERS@",
    "@MEMBER_OF_POLITICAL_PARTY@",
    "@FOLLOWS@",
    "@CHILD@",
    "@PRODUCT_OR_MATERIAL_PRODUCED@",
    "@RESIDENCE@",
    "@AUTHOR@",
    "@COUNTRY@",
    "@SERIES@",
    "@PUBLICATION_DATE@",
    "@PARTICIPANT_OF@",
    "@COUNTRY_OF_ORIGIN@",
    "@INFLUENCED_BY@",
    "@MEMBER_OF_SPORTS_TEAM@",
    "@END_TIME@",
    "@RECORD_LABEL@",
    "@CONTINENT@",
    "@PART_OF@",
    "@APPLIES_TO_JURISDICTION@",
    "@PRESENT_IN_WORK@",
    "@ETHNIC_GROUP@",
    "@NARRATIVE_LOCATION@",
    "@OPERATOR@",
    "@LEGISLATIVE_BODY@",
    "@PERFORMER@",
    "@SIBLING@",
    "@HEAD_OF_STATE@",
    "@GENRE@",
    "@PARTICIPANT@",
    "@NOTABLE_WORK@",
    "@INCEPTION@",
    "@SISTER_CITY@",
    "@HEADQUARTERS_LOCATION@",
    "@HEAD_OF_GOVERNMENT@",
    "@RELIGION@",
    "@SPOUSE@",
    "@UNEMPLOYMENT_RATE@",
    "@PARENT_TAXON@",
    "@POSITION_HELD@",
    "@OWNED_BY@",
    "@DISSOLVED_ABOLISHED_OR_DEMOLISHED@",
    "@PLATFORM@",
    "@CAPITAL@",
    "@LOCATION@",
    "@COMPOSER@",
    "@LYRICS_BY@",
    "@CONTAINS_ADMINISTRATIVE_TERRITORIAL_ENTITY@",
    "@LOCATED_ON_TERRAIN_FEATURE@",
    "@ORIGINAL_NETWORK@",
    "@OFFICIAL_LANGUAGE@",
    "@PLACE_OF_DEATH@",
    "@FOLLOWED_BY@",
    "@EDUCATED_AT@",
    "@LOCATION_OF_FORMATION@",
    "@ORIGINAL_LANGUAGE_OF_WORK@",
    "@PARENT_ORGANIZATION@",
    "@SEPARATED_FROM@",
    "@MOTHER@",
    "@LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY@",
    "@MEMBER_OF@",
    "@CAPITAL_OF@",
    "@SUBCLASS_OF@",
    "@SUBSIDIARY@",
];
// A list of relation labels in your dataset
local labels = [
    "SCREENWRITER",
    "DATE_OF_BIRTH",
    "AWARD_RECEIVED",
    "LOCATED_IN_OR_NEXT_TO_BODY_OF_WATER",
    "CHAIRPERSON",
    "FATHER",
    "POINT_IN_TIME",
    "MANUFACTURER",
    "DATE_OF_DEATH",
    "CONFLICT",
    "TERRITORY_CLAIMED_BY",
    "HAS_PART",
    "CAST_MEMBER",
    "LANGUAGES_SPOKEN_WRITTEN_OR_SIGNED",
    "DEVELOPER",
    "FOUNDED_BY",
    "BASIN_COUNTRY",
    "DIRECTOR",
    "COUNTRY_OF_CITIZENSHIP",
    "EMPLOYER",
    "PUBLISHER",
    "PLACE_OF_BIRTH",
    "CREATOR",
    "MILITARY_BRANCH",
    "PRODUCTION_COMPANY",
    "START_TIME",
    "PRODUCER",
    "WORK_LOCATION",
    "INSTANCE_OF",
    "REPLACES",
    "MOUTH_OF_THE_WATERCOURSE",
    "REPLACED_BY",
    "LEAGUE",
    "CHARACTERS",
    "MEMBER_OF_POLITICAL_PARTY",
    "FOLLOWS",
    "CHILD",
    "PRODUCT_OR_MATERIAL_PRODUCED",
    "RESIDENCE",
    "AUTHOR",
    "COUNTRY",
    "SERIES",
    "PUBLICATION_DATE",
    "PARTICIPANT_OF",
    "COUNTRY_OF_ORIGIN",
    "INFLUENCED_BY",
    "MEMBER_OF_SPORTS_TEAM",
    "END_TIME",
    "RECORD_LABEL",
    "CONTINENT",
    "PART_OF",
    "APPLIES_TO_JURISDICTION",
    "PRESENT_IN_WORK",
    "ETHNIC_GROUP",
    "NARRATIVE_LOCATION",
    "OPERATOR",
    "LEGISLATIVE_BODY",
    "PERFORMER",
    "SIBLING",
    "HEAD_OF_STATE",
    "GENRE",
    "PARTICIPANT",
    "NOTABLE_WORK",
    "INCEPTION",
    "SISTER_CITY",
    "HEADQUARTERS_LOCATION",
    "HEAD_OF_GOVERNMENT",
    "RELIGION",
    "SPOUSE",
    "UNEMPLOYMENT_RATE",
    "PARENT_TAXON",
    "POSITION_HELD",
    "OWNED_BY",
    "DISSOLVED_ABOLISHED_OR_DEMOLISHED",
    "PLATFORM",
    "CAPITAL",
    "LOCATION",
    "COMPOSER",
    "LYRICS_BY",
    "CONTAINS_ADMINISTRATIVE_TERRITORIAL_ENTITY",
    "LOCATED_ON_TERRAIN_FEATURE",
    "ORIGINAL_NETWORK",
    "OFFICIAL_LANGUAGE",
    "PLACE_OF_DEATH",
    "FOLLOWED_BY",
    "EDUCATED_AT",
    "LOCATION_OF_FORMATION",
    "ORIGINAL_LANGUAGE_OF_WORK",
    "PARENT_ORGANIZATION",
    "SEPARATED_FROM",
    "MOTHER",
    "LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY",
    "MEMBER_OF",
    "CAPITAL_OF",
    "SUBCLASS_OF",
    "SUBSIDIARY",
];
// Max length of input text and max/min number of decoding steps
// These should be set based on your dataset
local max_length = 512;
local max_steps = 1024;

// Do not modify.
local tokens_to_add = special_tokens + COMMON["special_tokens"];

local SOURCE_TOKENIZER = {
    "type": "pretrained_transformer",
    "model_name": COMMON["model_name"],
    "max_length": max_length,
    "add_special_tokens": true,
};

local TARGET_TOKENIZER = {
    "type": "pretrained_transformer",
    "model_name": COMMON["model_name"],
    "add_special_tokens": false,
    "tokenizer_kwargs": {
        // HF tokenizers name this parameter one of two things, including both here.
        "special_tokens": tokens_to_add,
        "additional_special_tokens": tokens_to_add,
    },
};

{
    "vocabulary": {
        // This is a hacky way to ensure the target vocab contains only the
        // special tokens (i.e. the COPY token and anything in tokens_to_add)
        "max_vocab_size": {
            [COMMON["target_namespace"]]: 1
        },
        "tokens_to_add" : {
            [COMMON["target_namespace"]]: tokens_to_add
        },
    },
    "train_data_path": COMMON["train_data_path"],
    "validation_data_path": COMMON["validation_data_path"],
    "dataset_reader": {
        "type": "seq2rel",
        "target_namespace": COMMON["target_namespace"],
        "source_tokenizer": SOURCE_TOKENIZER,
        "target_tokenizer": TARGET_TOKENIZER,
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": COMMON["model_name"],
            }
        },
    },
    "model": {
        "type": "copynet_seq2rel",
        "source_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": COMMON["model_name"],
                    "tokenizer_kwargs": source_tokenizer_kwargs
                },
            },
        },
        "target_tokenizer": TARGET_TOKENIZER,
        "dropout": 0.1,
        "token_based_metric": {
            "type": "average_length"
        },
        "sequence_based_metrics": [
            {
                "type": "f1_seq2rel",
                "labels": rel_tokens,
                "average": "micro"
            },
            {
                "type": "valid_sequences",
            },
        ],
        "attention": {
            "type": "dk_scaled_dot_product"
        },
        "target_embedding_dim": COMMON["target_embedding_dim"],
        "beam_search": {
            "max_steps": max_steps,
            "beam_size": COMMON["beam_size"],
            "final_sequence_scorer": {
                "type": "length-normalized-sequence-log-prob",
                // Larger values favour longer decodings and vice versa
                "length_penalty": 1.0
            },
            "constraints": [
                {
                    "type": "seq2rel",
                    "rel_tokens": rel_tokens,
                    "ent_tokens": ent_tokens,
                    "target_namespace": COMMON["target_namespace"]
                },
            ],
        },
    },
    "data_loader": COMMON["data_loader"],
    "validation_data_loader": COMMON["validation_data_loader"],
    "trainer": COMMON["trainer"],
}

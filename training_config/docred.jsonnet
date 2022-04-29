// =================== Configurable Settings ======================

// The pretrained model to use as encoder.
// Should be a registered name in the Transformers library (see https://huggingface.co/models) 
// OR a path on disk to a serialized transformer model.
local model_name = "bert-base-uncased";

// These are reasonable defaults.
local max_length = 512;       // Max length of input text
local max_steps = 400;        // Max number of decoding steps

local num_epochs = 50;        // Number of training epochs
local batch_size = 4;         // Per-GPU batch size
local grad_acc_steps = 1;     // Number of training steps before backpropagating gradients
local decoder_lr = 7.8e-5;    // Learning rate for decoder params

local encoder_lr = 2e-5;      // Learning rate for encoder params
local encoder_wd = 0.01;      // Weight  decay for encoder params
local reinit_layers = 1;      // Re-initializes the last N layers of the encoder
local dropout = 0.10;         // Dropout applied to decoder inputs and cross-attention weights
local weight_dropout = 0.50;  // Weight dropout applied to hidden-to-hidden decoder weights

local beam_size = 8;          // Beam size to use during decoding (test time only)
local length_penalty = 1.4;   // >1.0 favours longer decodings and <1.0 shorter (test time only)

// Number of GPUs to use. 0 means CPU only, 1 means one GPU, etc.
local num_gpus = 1;

// Set to `true` to use automatic mixed precision.
local use_amp = true;

// ================================================================

// Lists containing the special entity/relation tokens in your target vocabulary
local ent_tokens = [
    "@TIME@",
    "@ORG@",
    "@PER@",
    "@NUM@",
    "@LOC@",
    "@MISC@",
];

local rel_tokens = [
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

// These are provided as external variables
local train_data_path = std.extVar("train_data_path");
local validation_data_path = std.extVar("valid_data_path");
local dataset_size = std.parseInt(std.extVar('dataset_size'));

// Validation will begin at the end of this epoch.
local validation_start = std.max(std.floor(num_epochs - 2), 0);
// ...and continues for every validation_interval epochs after that
local validation_interval = 1;

// ------ !! You probably don't need to edit below here !! --------

// Learning rate will be linearly increased for the first 10% of training steps.
local warmup_steps = std.floor(dataset_size / batch_size * num_epochs * 0.10);

// Assumes relation labels match the special relation tokens minus the "@" symbol
local rel_labels = [std.stripChars(token, "@") for token in rel_tokens];
// Special tokens used in the source and target strings
local special_target_tokens = ent_tokens + rel_tokens + [";", "@start@", "@end@"];

// Define source and target namespaces
local source_namespace = "source_tokens";
local target_namespace = "target_tokens";
// Determines which namespace the bucket batch sampler will sort on
local sorting_keys = [source_namespace];

// Setup source tokenizer
local source_tokenizer_kwargs = {
    "do_lower_case": true
};
local SOURCE_TOKENIZER = {
    "type": "pretrained_transformer",
    "model_name": model_name,
    "max_length": max_length,
    "add_special_tokens": true,
    "tokenizer_kwargs": source_tokenizer_kwargs
};

// Setup target tokenizer
local target_tokenizer_kwargs = {
    "additional_special_tokens": special_target_tokens,
    "do_lower_case": true
};
local TARGET_TOKENIZER = {
    "type": "pretrained_transformer",
    "model_name": model_name,
    "add_special_tokens": false,
    "tokenizer_kwargs": target_tokenizer_kwargs
};

{
    "vocabulary": {
        // This is a hacky way to ensure the target vocab contains only the
        // special tokens (i.e. the COPY token and anything in tokens_to_add)
        "max_vocab_size": {
            [target_namespace]: 1
        },
        "tokens_to_add" : {
            [target_namespace]: special_target_tokens
        },
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "dataset_reader": {
        "type": "seq2rel",
        "max_length": max_length,
        "target_namespace": target_namespace,
        "source_tokenizer": SOURCE_TOKENIZER,
        "target_tokenizer": TARGET_TOKENIZER,
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name,
                "tokenizer_kwargs": source_tokenizer_kwargs,
            }
        },
    },
    "model": {
        "type": "copynet_seq2rel",
        "source_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": model_name,
                    "tokenizer_kwargs": source_tokenizer_kwargs,
                    "reinit_modules": reinit_layers,
                },
            },
        },
        "target_tokenizer": TARGET_TOKENIZER,
        "dropout": dropout,
        "weight_dropout": weight_dropout,
        "sequence_based_metrics": [
            {
                "type": "f1_seq2rel",
                "labels": rel_labels,
                "average": "micro",
                // DocRED has distinct head or tail entities, so this must be set to True to take
                // that into account during evaluation.
                "ordered_ents": true,
                "remove_duplicate_ents": true,
            },
        ],
        "attention": {
            "type": "multihead_attention",
            "num_heads": 6,
            "dropout": dropout,
        },
        "target_embedding_dim": 256,
        "beam_search": {
            "max_steps": max_steps,
            "beam_size": beam_size,
            "final_sequence_scorer": {
                "type": "length-normalized-sequence-log-prob",
                // Larger values favour longer decodings and vice versa
                "length_penalty": length_penalty,
            },
        },
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size,
            "sorting_keys": sorting_keys,
        },
    },
    "validation_data_loader": {
        "batch_sampler": {
            "type": "bucket",
            // To speed up validation, we set the batch size to a multiple of
            // the batch size used during training.
            "batch_size": batch_size * 32,
            "sorting_keys": sorting_keys,
            "padding_noise": 0.0,
        },
    },
        "trainer": {
        "num_epochs": num_epochs,
        "validation_metric": "+fscore",
        "num_gradient_accumulation_steps": grad_acc_steps,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": decoder_lr,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "correct_bias": true,
            "parameter_groups": [
                // All parameters of the transformer excluding biases and LayerNorm
                // RegEx unit tests: https://regex101.com/r/e5MakA/1
                [
                    ["transformer_model(?!.*(?:bias|LayerNorm|layer_norm))"],
                    {"lr": encoder_lr, "weight_decay": encoder_wd}
                ],
                // All parameters of of the transformer that include biases and LayerNorm
                // RegEx unit tests: https://regex101.com/r/RWo1yv/1
                [
                    ["transformer_model(?=.*(?:bias|LayerNorm|layer_norm))"],
                    {"lr": encoder_lr, "weight_decay": 0.0}
                ],
            ],  
        },
        "learning_rate_scheduler": {
            "type": "linear_with_warmup",
            "warmup_steps": warmup_steps
        },
        "grad_norm": 1.0,
        "use_amp": use_amp,
        "callbacks": [
            {
                "type": "should_validate_callback",
                "validation_start": validation_start,
                "validation_interval": validation_interval,
            },
	    ],
        "checkpointer": {
            "keep_most_recent_by_count": 1
        },
    },
    [if num_gpus > 1 then "distributed"]: {
        "cuda_devices": std.range(0, num_gpus - 1),
    },
}
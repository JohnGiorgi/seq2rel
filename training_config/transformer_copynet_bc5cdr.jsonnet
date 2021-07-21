// This config contains anything that doesn"t change across experiments and datasets
local COMMON = import "common.libsonnet";

// ** THESE MUST BE SET BY THE USER **//
// Entity hints used in the text
local ent_hints = [
    "@START_CHEMICAL@",
    "@END_CHEMICAL@",
    "@START_DISEASE@",
    "@END_DISEASE@",
];
// Lists containing the special entity/relation tokens in your target vocabulary
local ent_tokens = [
    "@CHEMICAL@",
    "@DISEASE@",
];
local rel_tokens = [
    "@CID@",
];
// Max length of input text and max/min number of decoding steps
// These should be set based on your dataset
local max_length = 512;
local max_steps = 256;

// Do not modify.
local special_target_tokens = ent_tokens + rel_tokens + COMMON["special_target_tokens"];
local source_tokenizer_kwargs = {
    // HF tokenizers name this parameter one of two things, including both here.
    "special_tokens": ent_hints,
    "additional_special_tokens": ent_hints
};
local target_tokenizer_kwargs = {
    "special_tokens": special_target_tokens,
    "additional_special_tokens": special_target_tokens
};

local SOURCE_TOKENIZER = {
    "type": "pretrained_transformer",
    "model_name": COMMON["model_name"],
    "max_length": max_length,
    "add_special_tokens": true,
    "tokenizer_kwargs": source_tokenizer_kwargs
};

local TARGET_TOKENIZER = {
    "type": "pretrained_transformer",
    "model_name": COMMON["model_name"],
    "add_special_tokens": false,
    "tokenizer_kwargs": target_tokenizer_kwargs
};

{
    "vocabulary": {
        // This is a hacky way to ensure the target vocab contains only the
        // special tokens (i.e. the COPY token and anything in tokens_to_add)
        "max_vocab_size": {
            [COMMON["target_namespace"]]: 1
        },
        "tokens_to_add" : {
            [COMMON["source_namespace"]]: ent_hints,
            [COMMON["target_namespace"]]: special_target_tokens
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
                "tokenizer_kwargs": source_tokenizer_kwargs
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
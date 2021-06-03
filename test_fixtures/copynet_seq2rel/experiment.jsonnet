// This config contains anything that doesn"t change across experiments and datasets
local COMMON = import "common.libsonnet";

// ** THESE MUST BE SET BY THE USER **//
// Entity hints used in the source text
local ent_hints = [
    "@START_GGP@",
    "@END_GGP@",
];
// A list containing the special tokens in your target vocabulary
local special_tokens = [
    "@GGP@",
    "@PHYSICAL@",
    "@GENETIC@",
];
// A list of relation labels in your dataset
local labels = ["GENETIC", "PHYSICAL"];
// Max length of input text and max/min number of decoding steps
// These should be set based on your dataset
local max_length = 16;
local max_steps = 8;
// Usually, this will be the special relation tokens (2) + two enties and their tag tokens (4)
// Has little to no impact on performance, but it may be worth experimenting with
local min_steps = null;

// Do not modify.
local tokens_to_add = special_tokens + COMMON["special_tokens"];
local source_tokenizer_kwargs = {
    // HF tokenizers name this parameter one of two things, including both here.
    "special_tokens": ent_hints,
    "additional_special_tokens": ent_hints
};
local target_tokenizer_kwargs = {
    "special_tokens": tokens_to_add,
    "additional_special_tokens": tokens_to_add
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
            [COMMON["target_namespace"]]: tokens_to_add
        },
    },
    "train_data_path": COMMON["train_data_path"],
    "validation_data_path": COMMON["validation_data_path"],
    "dataset_reader": {
        "type": "seq2rel.data.dataset_readers.copynet_seq2rel.CopyNetSeq2RelDatasetReader",
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
        "type": "seq2rel.models.copynet_seq2rel.CopyNetSeq2Rel",
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
        "token_based_metric": {
            "type": "seq2rel.metrics.AverageLength"
        },
        "sequence_based_metric": {
            "type": "seq2rel.metrics.F1MeasureSeq2Rel",
            "labels": labels,
            "average": "micro"
        },
        "attention": {
            "type": "seq2rel.modules.attention.dk_scaled_dot_product_attention.DkScaledDotProductAttention"
        },
        "beam_search": {
            "max_steps": max_steps,
            "min_steps": min_steps,
            "beam_size": COMMON["beam_size"],
            "final_sequence_scorer": {
                "type": "length-normalized-sequence-log-prob",
                // Larger values favour longer decodings and vice versa
                "length_penalty": 1.0
            },
        },
        "target_embedding_dim": COMMON["target_embedding_dim"],
    },
    "data_loader": COMMON["data_loader"],
    "validation_data_loader": COMMON["validation_data_loader"],
    "trainer": COMMON["trainer"],
}

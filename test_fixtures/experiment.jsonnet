// =================== Configurable Settings ======================

// The pretrained model to use as encoder. This is a reasonable default for biomedical text.
// Should be a registered name in the Transformers library (see https://huggingface.co/models) 
// OR a path on disk to a serialized transformer model.
local model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext";

// These are reasonable defaults.
local max_length = 16;        // Max length of input text
local max_steps = 16;         // Max number of decoding steps

local num_epochs = 1;         // Number of training epochs
local batch_size = 1;         // Per-GPU batch size
local grad_acc_steps = 1;     // Number of training steps before backpropagating gradients
local decoder_lr = 5e-4;      // Learning rate for decoder params

local reinit_layers = 2;      // Re-initializes the last N layers of the encoder
local dropout = 0.0;          // Dropout applied to decoder inputs and cross-attention weights
local weight_dropout = 0.0;   // Weight dropout applied to hidden-to-hidden decoder weights

local beam_size = 1;          // Beam size to use during decoding (test time only)
local length_penalty = 1.0;   // >1.0 favours longer decodings and <1.0 shorter (test time only)

// Number of GPUs to use. 0 means CPU only, 1 means one GPU, etc.
local num_gpus = 0;

// Set to `true` to use automatic mixed precision.
local use_amp = false;

// Set to `true` to log to Weights & Biases.
local use_wandb = false;

// ================================================================

// ------ !! You probably don't need to edit below here !! --------

// This config contains anything that doesn't change across experiments and datasets
local COMMON = import "common.libsonnet";

// These are provided as external variables
local train_data_path = "test_fixtures/data/train.tsv";
local validation_data_path = "test_fixtures/data/valid.tsv";

// Validation will begin at the end of this epoch.
local validation_start = std.floor(num_epochs - 4);
// Learning rate will be linearly increased for the first 10% of training steps.
local warmup_steps = std.floor(0.10 * num_epochs);

// Lists containing the special entity/relation tokens in your target vocabulary
local ent_tokens = [
    "@CHEMICAL@",
    "@DISEASE@",
];
local rel_tokens = [
    "@CID@",
];

local special_source_tokens = ent_tokens;
local special_target_tokens = ent_tokens + rel_tokens + COMMON["special_target_tokens"];

local source_tokenizer_kwargs = {
    "additional_special_tokens": special_source_tokens,
    "do_lower_case": true
};
local target_tokenizer_kwargs = {
    "additional_special_tokens": special_target_tokens,
    "do_lower_case": true
};

local SOURCE_TOKENIZER = {
    "type": "pretrained_transformer",
    "model_name": model_name,
    "max_length": max_length,
    "add_special_tokens": true,
    "tokenizer_kwargs": source_tokenizer_kwargs
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
            [COMMON["target_namespace"]]: 1
        },
        "tokens_to_add" : {
            [COMMON["target_namespace"]]: special_target_tokens
        },
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "dataset_reader": {
        "type": "seq2rel.dataset_reader.Seq2RelDatasetReader",
        "max_length": max_length,
        "target_namespace": COMMON["target_namespace"],
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
        "type": "seq2rel.models.copynet_seq2rel.CopyNetSeq2Rel",
        "source_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": model_name,
                    "tokenizer_kwargs": source_tokenizer_kwargs,
                    // TODO: Add this back when we update to AllenNLP>2.8.0
                    // "reinit_modules": reinit_layers,
                },
            },
        },
        "target_tokenizer": TARGET_TOKENIZER,
        "dropout": dropout,
        "weight_dropout": weight_dropout,
        "token_based_metric": {
            "type": "seq2rel.metrics.AverageLength"
        },
        "sequence_based_metrics": [
            {
                "type": "seq2rel.metrics.F1MeasureSeq2Rel",
                "labels": ["CID"],
                "average": "micro",
                "remove_duplicate_ents": true,
            },
            {
                "type": "seq2rel.metrics.ValidSequences",
            },
        ],
        "attention": {
            "type": "seq2rel.modules.attention.multihead_attention.MultiheadAttention",
            "num_heads": 6,
            "dropout": dropout,
        },
        "target_embedding_dim": 128,
        "beam_search": {
            "max_steps": max_steps,
            "beam_size": beam_size,
            "final_sequence_scorer": {
                "type": "length-normalized-sequence-log-prob",
                // Larger values favour longer decodings and vice versa
                "length_penalty": length_penalty,
            },
            "constraints": [
                {
                    "type": "seq2rel.nn.constraints.EnforceValidLinearization",
                    "ent_tokens": ent_tokens,
                    "rel_tokens": rel_tokens,
                    "target_namespace": COMMON["target_namespace"]
                },
            ],
        },
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size,
            "sorting_keys": COMMON["sorting_keys"],
        },
    },
    "validation_data_loader": {
        "batch_sampler": {
            "type": "bucket",
            // To speed up validation, we set the batch size to a multiple of
            // the batch size used during training.
            "batch_size": batch_size * 32,
            "sorting_keys": COMMON["sorting_keys"],
            // We don't care about deterministic batches during validation, so drop
            // padding noise to further speed things up.
            "padding_noise": 0.0
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
                    {"lr": COMMON["encoder_lr"], "weight_decay": COMMON["encoder_wd"]}
                ],
                // All parameters of of the transformer that include biases and LayerNorm
                // RegEx unit tests: https://regex101.com/r/RWo1yv/1
                [
                    ["transformer_model(?=.*(?:bias|LayerNorm|layer_norm))"],
                    {"lr": COMMON["encoder_lr"], "weight_decay": 0.0}
                ],
            ],  
        },
        "grad_norm": 1.0,
        "learning_rate_scheduler": {
            "type": "linear_with_warmup",
            "warmup_steps": warmup_steps
        },
        "use_amp": use_amp,
        // TODO: Add back when we update to AllenNLP>2.8.0
        // "callbacks": [
        //     {
        //         "type": "should_validate_callback",
        //         "validation_start": validation_start,
        //         "validation_interval": 1
        //     },
	    // ],
    },
    [if num_gpus > 1 then "distributed"]: {
        "cuda_devices": std.range(0, num_gpus - 1),
    },
}
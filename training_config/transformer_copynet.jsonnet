// ** THESE MUST BE SET BY THE USER **//
// A list containing the special tokens in your vocabulary
local tokens_to_add = [null];
// The relation labels in your dataset
local labels = [null];

// These are good defaults and likely should not be changed
local sorting_keys = ["source_tokens", "target_tokens"];

// These are hyperparameters that are set using environment variables.
// This also allows us to tune them automatically with Optuna.
local train_data_path = std.extVar("TRAIN_DATA_PATH");
local valid_data_path = std.extVar("VALID_DATA_PATH");
// This should be a registered name in the Transformers library (see https://huggingface.co/models) 
// OR a path on disk to a serialized transformer model.
local model_name = std.extVar("MODEL_NAME");

local beam_size = std.parseInt(std.extVar("beam_size"));
local target_embedding_dim = std.parseInt(std.extVar("target_embedding_dim"));
local batch_size = std.parseInt(std.extVar("batch_size"));
local encoder_lr = std.parseJson(std.extVar('encoder_lr'));
local decoder_lr = std.parseJson(std.extVar('decoder_lr'));
local weight_decay = std.parseJson(std.extVar('weight_decay'));

local SOURCE_TOKENIZER = {
    "type": "pretrained_transformer",
    "model_name": model_name,
    "max_length": 512,
    "add_special_tokens": true,
};

local TARGET_TOKENIZER = {
    "type": "pretrained_transformer",
    "model_name": model_name,
    "add_special_tokens": false,
    "tokenizer_kwargs": {
        // HF tokenizers name this parameter one of two things, including both here.
        "special_tokens": tokens_to_add,
        "additional_special_tokens": tokens_to_add
    },
};

{
    "distributed": {
        "cuda_devices": [0, 1]
    },
    "vocabulary": {
        // This is a hacky way to ensure the target vocab contains only the
        // special tokens (i.e. the COPY token and anything in tokens_to_add)
        "max_vocab_size": {
            "target_tokens": 1
        },
        "tokens_to_add" : {
            "target_tokens": tokens_to_add
        },
    },
    "train_data_path": train_data_path,
    "validation_data_path": valid_data_path,
    "dataset_reader": {
        "type": "copynet_seq2rel",
        "target_namespace": "target_tokens",
        "source_tokenizer": SOURCE_TOKENIZER,
        "target_tokenizer": TARGET_TOKENIZER,
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name,
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
                },
            },
        },
        "target_tokenizer": TARGET_TOKENIZER,
        "sequence_based_metric": {
            "type": "f1_seq2rel",
            "labels": labels,
            "average": "micro"
        },
        "attention": {
            "type": "dk_scaled_dot_product"
        },
        "beam_size": beam_size,
        "max_decoding_steps": 128,
        "target_embedding_dim": target_embedding_dim,
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
            "batch_size": batch_size * 6,
            "sorting_keys": sorting_keys,
        },
    },
    "trainer": {
        "num_epochs": 20,
        "validation_metric": "+fscore",
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": decoder_lr,
            "weight_decay": 0.0,
            "parameter_groups": [
                // Apply weight decay to pre-trained params, excluding LayerNorm params and biases
                // Regex: https://regex101.com/r/6XnPtH/1
                [
                    ["(?=.*transformer_model)(?!.*(LayerNorm\\.weight|bias)).*$"],
                    {"lr": encoder_lr, "weight_decay": weight_decay}
                ],
                // Use different learning rate for the pre-trained weights.
                [["(?=.*transformer_model)(?=.*(LayerNorm\\.weight|bias)).*$"], {"lr": encoder_lr}],
            ],
        },
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
        },
        "callbacks": [
            {
                "type": "tensorboard",
                "summary_interval": 4,
                "should_log_learning_rate": true
            }
        ],
        "grad_norm": 1.0,
        "use_amp": true,
    }
}
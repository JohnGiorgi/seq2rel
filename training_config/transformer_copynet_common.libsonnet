// This config contains anything that does not change across experiments and datasets
// To import it in another jsonnet config, place the following line at the top of the file:
// local COMMON = import 'transformer_copynet_common.libsonnet';
// Fields can then be accessed like a dictionary, e.g. COMMON['batch_size']

// These are special tokens used by the decoder
local special_target_tokens = [
    // These are seq2rel specific
    "@EOR@",
    ";",
    // These are AllenNLP specific
    "@start@",
    "@end@"
];

// Define the name spaces of the source and target tokens respectively
local source_namespace = "source_tokens";
local target_namespace = "target_tokens";
local sorting_keys = [source_namespace];

// These are set using environment variables.
// This allows us to tune them automatically with Optuna.
local train_data_path = std.extVar("TRAIN_DATA_PATH");
local validation_data_path = std.extVar("VALID_DATA_PATH");
// This should be a registered name in the Transformers library (see https://huggingface.co/models) 
// OR a path on disk to a serialized transformer model.
local model_name = std.extVar("MODEL_NAME");
// Hyperparameters
local beam_size = std.parseInt(std.extVar("beam_size"));
local target_embedding_dim = std.parseInt(std.extVar("target_embedding_dim"));
local batch_size = std.parseInt(std.extVar("batch_size"));
local num_epochs = std.parseInt(std.extVar("num_epochs"));
local encoder_lr = std.parseJson(std.extVar('encoder_lr'));
local decoder_lr = std.parseJson(std.extVar('decoder_lr'));
local weight_decay = std.parseJson(std.extVar('weight_decay'));

{
    "special_target_tokens": special_target_tokens,
    "source_namespace": source_namespace,
    "target_namespace": target_namespace,
    "sorting_keys": sorting_keys,
    // Hyperparameters
    // We include them here to expose them to any config that may import this one.
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "model_name": model_name,
    "beam_size": beam_size,
    "target_embedding_dim": target_embedding_dim,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "encoder_lr": encoder_lr,
    "decoder_lr": decoder_lr,
    "weight_decay": weight_decay,
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
            // of the batch size used during training.
            "batch_size": batch_size * 64,
            "sorting_keys": sorting_keys,
            // We don't care about deterministic batches during validation, so drop
            // padding noise to further speed things up.
            "padding_noise": 0.0
        },
    },
    "trainer": {
        "num_epochs": num_epochs,
        "validation_metric": "+fscore",
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": decoder_lr,
            "weight_decay": 0.01,
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
        "grad_norm": 1.0,
        "use_amp": true,
    }
}

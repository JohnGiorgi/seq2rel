// =================== Configurable Settings ======================

// These are reasonable defaults.
local encoder_lr = 2e-5;  // Learning rate for encoder params
local encoder_wd = 0.01;  // Weight decay for encoder params

// ================================================================

// ------ !! You probably don't need to edit below here !! --------

// These are special tokens used by the decoder
local special_target_tokens = [
    // seq2rel specific
    ";",
    // AllenNLP specific
    "@start@",
    "@end@"
];

local source_namespace = "source_tokens";
local target_namespace = "target_tokens";
# Determines which namespace the bucket batch sampler will sort on
local sorting_keys = [source_namespace];

{
    "special_target_tokens": special_target_tokens,
    "source_namespace": source_namespace,
    "target_namespace": target_namespace,
    "sorting_keys": sorting_keys,
    // Hyperparameters
    // We include them here to expose them to any config that may import this one.
    "encoder_lr": encoder_lr,
    "encoder_wd": encoder_wd,
}

// This config contains anything that doesn't change across experiments and datasets
local COMMON = import 'transformer_copynet_common.libsonnet';

// ** THESE MUST BE SET BY THE USER **//
// A list containing the special tokens in your vocabulary
local special_tokens = [
    "@PRGE@",
    "@PHYSICAL@",
    "@GENETIC@",
];
// A list of relation labels in your dataset
local labels = ["GENETIC", "PHYSICAL"];
// Max length of input text and max number of decoding steps
// These should be set based on your dataset
local max_length = 512;
local max_decoding_steps = 160;

// Do not modify.
local tokens_to_add = special_tokens + COMMON['special_tokens'];

local SOURCE_TOKENIZER = {
    'type': 'pretrained_transformer',
    'model_name': COMMON['model_name'],
    'max_length': max_length,
    'add_special_tokens': true,
};

local TARGET_TOKENIZER = {
    'type': 'pretrained_transformer',
    'model_name': COMMON['model_name'],
    'add_special_tokens': false,
    'tokenizer_kwargs': {
        // HF tokenizers name this parameter one of two things, including both here.
        'special_tokens': tokens_to_add,
        'additional_special_tokens': tokens_to_add,
    },
};

{
    'vocabulary': {
        // This is a hacky way to ensure the target vocab contains only the
        // special tokens (i.e. the COPY token and anything in tokens_to_add)
        'max_vocab_size': {
            [COMMON['target_namespace']]: 1
        },
        'tokens_to_add' : {
            [COMMON['target_namespace']]: tokens_to_add
        },
    },
    'train_data_path': COMMON['train_data_path'],
    'validation_data_path': COMMON['validation_data_path'],
    'dataset_reader': {
        'type': 'copynet_seq2rel',
        'target_namespace': COMMON['target_namespace'],
        'source_tokenizer': SOURCE_TOKENIZER,
        'target_tokenizer': TARGET_TOKENIZER,
        'source_token_indexers': {
            'tokens': {
                'type': 'pretrained_transformer',
                'model_name': COMMON['model_name'],
            }
        },
    },
    'model': {
        'type': 'copynet_seq2rel',
        'source_embedder': {
            'token_embedders': {
                'tokens': {
                    'type': 'pretrained_transformer',
                    'model_name': COMMON['model_name'],
                },
            },
        },
        'target_tokenizer': TARGET_TOKENIZER,
        'sequence_based_metric': {
            'type': 'f1_seq2rel',
            'labels': labels,
            'average': 'micro'
        },
        'attention': {
            'type': 'dk_scaled_dot_product'
        },
        'beam_size': COMMON['beam_size'],
        'max_decoding_steps': max_decoding_steps,
        'target_embedding_dim': COMMON['target_embedding_dim'],
    },
    'data_loader': COMMON['data_loader'],
    'validation_data_loader': COMMON['validation_data_loader'],
    'trainer': COMMON['trainer'],
}

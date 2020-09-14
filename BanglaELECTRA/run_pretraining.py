#%%
from simpletransformers.language_modeling import LanguageModelingModel
from tokenizers import BertWordPieceTokenizer
import logging
import os
import glob
from electraDataset import ElectraDataset, LazyElectraDataset
import ddp_helper


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# training tokenizer
if not os.path.isdir('./vocab'):
    os.makedirs('./vocab', exist_ok=True)
    train_files = [f"./inputs/pretrain/{f}" for f in os.listdir('./inputs/pretrain')]
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
    )
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    
    tokenizer.train(
        files=train_files,
        vocab_size=32000,
        min_frequency=2,
        special_tokens=special_tokens,
        limit_alphabet=500,
        wordpieces_prefix="##",
    )

    tokenizer.save_model('./vocab')

vocab_file = f'./vocab/{os.listdir("./vocab")[0]}'
print(vocab_file)

with open(vocab_file) as f:
    for vocab_size, _ in enumerate(f, 1):
        pass

print(f'Vocab size: {vocab_size}')

ELECTRA_SMALL_DEFAULT = {
    'generator_config': {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "vocab_size": vocab_size
    },
    'discriminator_config': {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "vocab_size": vocab_size
    }
}

ELECTRA_SMALL_PAPER = {
    'generator_config': {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 256,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 1,
        "num_hidden_layers": 12,
        "vocab_size": vocab_size
    },
    'discriminator_config': {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "vocab_size": vocab_size
    }
}


ELECTRA_SMALL_PAPER_MODIFIED = {
    'generator_config': {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 256,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 128,
        "num_attention_heads": 1,
        "num_hidden_layers": 12,
        "vocab_size": vocab_size
    },
    'discriminator_config': {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "vocab_size": vocab_size
    }
}

ELECTRA_BASE_PAPER_MODIFIED = {
    'generator_config': {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 2048,
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "vocab_size": vocab_size,
        "type_vocab_size": 4
    },
    'discriminator_config': {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 2048,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": vocab_size,
        "type_vocab_size": 4
    }
}



model = ELECTRA_SMALL_PAPER_MODIFIED


train_args = {
    # logging settings
    # "wandb_project": "ELECTRA_LR_Runs",
    # "wandb_kwargs": {"name": "Electra-BASE-Final", "notes": "Final run."},
    # "wandb_disable_gradient_logging": "True",
    # "logging_steps": 100,

    # training settings
    "num_train_epochs": 40,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": True,
    "save_steps": 50000,
    "save_best_model": True,
    # "max_steps": 500000,
    
    # validation settings
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 2500,
    "evaluate_during_training_verbose": True,
    "use_cached_eval_features": True,
    
    # optimization settings
    "learning_rate": 2e-4,
    "warmup_steps": 10000,
    "gradient_accumulation_steps": 4,
    "adam_epsilon": 1e-6,
    "weight_decay": 0.01,
    
    # batch sizes
    "train_batch_size": 8,
    "eval_batch_size": 16,
    "block_size": 128,
    "max_seq_length": 128,

    # model settings
    "generator_config": model['generator_config'],
    "mlm_probability": 0.15,
    "do_whole_word_mask" : False,
    "discriminator_config": model['discriminator_config'],
    "tie_generator_and_discriminator_embeddings": True,
    "discriminator_loss_weight": 50.0,
    
    # dataset/tokenizer settings
    "tokenizer_name": vocab_file,
    "reprocess_input_data": False,
    "dataset_class": LazyElectraDataset,
    "vocab_size": vocab_size,
    "clean_text": True,
    "handle_chinese_chars": False,
    "strip_accents": False,
    "do_lower_case": False,
    "min_frequency": 100,

    # gpu / mixed precision settings
    "n_gpu": 8, # when gt 1, this will be reset to 1 since we are using ddp
    "fp16": True,
    "fp16_opt_level": "O1",

    # misc. settings
    "manual_seed": 3435,

    # output settings
    "overwrite_output_dir": True,
    "best_model_dir": "outputs/pretrain/electra_base_paper_final/best_model",
    "cache_dir": "cache_dir/pretrain",
    "output_dir": "outputs/pretrain/electra_base_paper_final/",

}

train_file = "inputs/pretrain/train-wikidump-books.en"
# test_file = "inputs/pretrain/test.txt"


model = LanguageModelingModel("electra", None, args=train_args)

model.load_and_cache_examples(train_file)
# model.load_and_cache_examples(test_file)


# def globalizer(fn=None, kwargs=None):
#     if fn:
#         fn(**kwargs)


# if __name__ == "__main__":
#     if train_args['n_gpu'] > 1:
#         fn = model.train_model
#         kwargs = {
#             'train_file': train_file, 
#             'args': train_args, 
#             'eval_file': test_file
#         }
#         ddp_helper.spawn(globalizer, kwargs={'fn': fn, 'kwargs': kwargs}, nprocs=train_args['n_gpu'])
#     else:
#         model.train_model(train_file=train_file, args=train_args, eval_file=test_file)


#     model.eval_model(test_file)
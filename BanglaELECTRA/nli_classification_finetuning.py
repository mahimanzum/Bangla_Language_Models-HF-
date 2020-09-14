#%%
from simpletransformers.classification import ClassificationModel
from tokenizers import BertWordPieceTokenizer
import logging
import os
import glob
import sklearn


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


train_args = {
    # logging settings
    "wandb_project": "ELECTRA_NLI_Runs",
    "wandb_kwargs": {"name": "ELECTRA-BASE-Final NLI-Finetuning", "notes": "Demo run."},
    "wandb_disable_gradient_logging": "True",
    "logging_steps": 50,

    # training settings
    "num_train_epochs": 3,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "save_steps": 2000,
    "save_best_model": True,
    "max_steps": -1,
    
    # validation settings
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 200,
    "evaluate_during_training_verbose": True,
    "use_cached_eval_features": True,
    
    # optimization settings
    "learning_rate": 1e-4,
    'warmup_ratio': 0.1,
    "gradient_accumulation_steps": 4,
    "adam_epsilon": 1e-6,
    "weight_decay": 0.01,
    "layerwise_lr_decay": 0.8,

    
    # batch sizes
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "block_size": 256,
    "max_seq_length": 160,

    # model settings
    # (loaded from the pretrained model)
    
    
    # dataset/tokenizer settings
    "reprocess_input_data": True,
    "clean_text": True,
    "handle_chinese_chars": False,
    "strip_accents": False,
    "do_lower_case": False,
    "min_frequency": 2,

    # gpu / mixed precision settings
    "n_gpu": 1, # when gt 1, this will be reset to 1 since we are using ddp
    "fp16": False,
    "fp16_opt_level": "O1",

    # misc. settings
    "manual_seed": 3435,

    # output settings
    "overwrite_output_dir": True,
    "best_model_dir": "outputs/finetune/nli/mnli/electra_base_final/best_model",
    "cache_dir": "cache_dir/finetune/nli/mnli",
    "output_dir": "outputs/finetune/nli/mnli/electra_base_final/",

}


inputDir = './inputs/finetune/nli/mnli'

train_file = f"{inputDir}/train.txt"
eval_file = f"{inputDir}/eval.txt"
test_file = f"{inputDir}/test.txt"

# eval_file = f"{inputDir}/eval_matched.txt"
# test_file_matched = f"{inputDir}/test_matched.txt"
# test_file_mismatched = f"{inputDir}/test_mismatched.txt"

model_dir = 'outputs/pretrain/electra_base_final/discriminator_model'

kwargs = {
    'accuracy' : sklearn.metrics.accuracy_score,
    # 'precsion' : sklearn.metrics.precision_score,
    # 'recall' : sklearn.metrics.recall_score,
    # 'f1': sklearn.metrics.f1_score
}

labels_list = ["contradiction", "entailment", "neutral"]
labels_map = None

train_args['labels_list'] = labels_list
train_args['labels_map'] = labels_map


model = ClassificationModel("electra", model_dir, num_labels=len(labels_list), weight=None, args=train_args)
# model = ClassificationModel("bert", "bert-base-multilingual-cased", num_labels=len(labels_list), weight=None, args=train_args)
# model = ClassificationModel('bert', 'bert-base-multilingual-uncased', num_labels=len(labels_list), use_cuda=True, args={
#     'reprocess_input_data' : True,
#     'use_cached_eval_features': False,
#     'overwrite_output_dir': True,
#     'num_train_epochs': 4,
#     'output_dir': 'outputs/tmp',
#     'best_model_dir': 'outputs/tmp/best',
#     'cache_dir': 'cache_dir/tmp',
#     'handle_chinese_chars': True,
#     'labels_list': labels_list,
#     'fp16': False,
#     "learning_rate": 2e-5,
#     "gradient_accumulation_steps": 2,
#     'warmup_ratio': 0.1,
#     "wandb_project": "ELECTRA_Runs",
#     "wandb_kwargs": {"name": "mBERT-uncased NLI-Finetuning", "notes": "Demo run."},
#     "logging_steps": 50,
#     "train_batch_size": 16,
#     "eval_batch_size": 16,
#     "evaluate_during_training": True,
#     "evaluate_during_training_steps": 50,
#     "evaluate_during_training_verbose": True,
#     "save_eval_checkpoints": False,
#     "save_model_every_epoch": False,
#     "save_steps": 10000,
#     "save_best_model": True

# })



model.train_model(train_file, eval_df=eval_file, **kwargs)
model.eval_model(test_file, **kwargs)

# model.eval_model(test_file_matched, **kwargs)
# model.eval_model(test_file_mismatched, **kwargs)

model = ClassificationModel("electra", train_args['best_model_dir'], num_labels=len(labels_list), weight=None, args=train_args)
model.eval_model(test_file, **kwargs)






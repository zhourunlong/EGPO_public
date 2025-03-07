import deepspeed
import json
import os
import pdb
import time
import torch

from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    Trainer, TrainingArguments

from utils import tokenize_pref_dataset, EXP_ROOT, DATA_ROOT, MODEL_ROOT

TAG = "Preference"

batch_size = 64
micro_batch_size = 1
learning_rate = 1e-5

EXP_ROOT = os.path.join(EXP_ROOT, "pref_models")
os.environ["WANDB_PROJECT"] = "preference_models"

model_name = "google/gemma-2-2b-it"
# model_name = "google/gemma-2-9b-it"
dataset_name = "weqweasdas/preference_dataset_mixture2_and_safe_pku"
max_seq_length = 1024

deepspeed.init_distributed()

# Configuration
num_gpus = torch.cuda.device_count()
gradient_accumulation_steps = batch_size // (micro_batch_size * num_gpus)

exp_base_name = model_name.split("/")[-1].replace(f"-{TAG}", "") \
    + "-" + dataset_name.split("/")[-1] + f"-{TAG}"
run_name = exp_base_name + "-" + time.strftime("%Y%m%d-%H%M%S")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    cache_dir=MODEL_ROOT,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

tokenized_path=os.path.join(DATA_ROOT,
                            "tokenized",
                            exp_base_name + "-" + str(max_seq_length))

local_rank = int(os.environ.get("LOCAL_RANK", 0))
if not os.path.exists(tokenized_path):
    if local_rank == 0:
        print("Didn't find tokenized dataset. Tokenizing now.")
        tokenize_pref_dataset(dataset_name,
                              tokenizer,
                              tokenized_path,
                              max_seq_length)
    
    deepspeed.comm.barrier()

train_dataset = load_from_disk(tokenized_path)

config = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "learning_rate": learning_rate,
    "weight_decay": 0.1,
    "bf16": True,
    "warmup_steps": 1000,
}

ds_config = {
    "train_batch_size": batch_size,
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "optimizer": {
        "type": "AdamW"
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_type": "linear",
            "warmup_max_lr": learning_rate,
        }
    },
    "zero_optimization": {
        "stage": 0
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10000000000,
    "wall_clock_breakdown": False,
    "zero_allow_untested_optimizer": True
}

training_args = TrainingArguments(
    run_name=run_name,
    output_dir=os.path.join(EXP_ROOT, run_name),
    **config,
    save_steps=0.1,
    save_total_limit=1,
    logging_steps=50,
    logging_dir='./logs',
    report_to="wandb",
    deepspeed=json.dumps(ds_config),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()

hf_username = os.environ.get("HF_USR_NAME", "")
if local_rank == 0 and hf_username != "":
    hf_token = os.environ.get("HF_TOKEN", "")
    hub_path = hf_username + "/" + exp_base_name
    print("Pushing model to hub at " + hub_path)
    model.push_to_hub(hub_path, private=True, use_auth_token=hf_token)
    tokenizer.push_to_hub(hub_path, private=True, use_auth_token=hf_token)
    print("Model pushed to hub.")

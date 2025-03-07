import deepspeed
import json
import os
import pdb
import time
import torch
import wandb

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from configs import get_ds_config
from utils import (
    tokenize_dataset,
    EXP_ROOT,
    DATA_ROOT,
    AverageLossCallback,
    load_model_tokenizer,
)

TAG = "SFT"

batch_size = 256
micro_batch_size = 1
learning_rate = 1e-5

EXP_ROOT = os.path.join(EXP_ROOT, "sft_models")
os.environ["WANDB_PROJECT"] = "sft_models"

model_name = "google/gemma-2-2b-it"
dataset_name = "yahma/alpaca-cleaned"
max_seq_length = 512

deepspeed.init_distributed()

# Configuration
num_gpus = torch.cuda.device_count()
gradient_accumulation_steps = batch_size // (micro_batch_size * num_gpus)

exp_base_name = model_name.split("/")[-1].replace(f"-{TAG}", "") \
    + "-" + dataset_name.split("/")[-1] + f"-{TAG}"
run_name = exp_base_name + "-" + time.strftime("%Y%m%d-%H%M%S")

model, tokenizer = load_model_tokenizer(model_name)

tokenized_path=os.path.join(DATA_ROOT,
                            "tokenized",
                            exp_base_name + "-" + str(max_seq_length))

local_rank = int(os.environ.get("LOCAL_RANK", 0))
if not os.path.exists(tokenized_path):
    if local_rank == 0:
        print("Didn't find tokenized dataset. Tokenizing now.")
        tokenize_dataset(dataset_name,
                         tokenizer,
                         tokenized_path,
                         max_seq_length)
    
    deepspeed.comm.barrier()

train_dataset = load_from_disk(tokenized_path)
train_dataset = train_dataset.map(
    lambda x: {**{k: x[f"sft_{k}"]
                  for k in ["input_ids", "attention_mask", "labels"]}},
    batched=True,
    remove_columns=train_dataset.column_names,
)

config = {
    "num_train_epochs": 5,
    "per_device_train_batch_size": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "learning_rate": learning_rate,
    "weight_decay": 0.1,
    "bf16": True,
    "warmup_steps": 100,
}

ds_config = get_ds_config(
    batch_size=batch_size,
    micro_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
)

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
    callbacks=[AverageLossCallback(gradient_accumulation_steps)],
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

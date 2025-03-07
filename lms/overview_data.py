import deepspeed
import json
import os
import pdb
import time
import torch

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    Trainer, TrainingArguments

from utils import tokenize_dataset, EXP_ROOT, DATA_ROOT, MODEL_ROOT

batch_size = 64
micro_batch_size = 1
learning_rate = 1e-5

EXP_ROOT = os.path.join(EXP_ROOT, "sft_models")
os.environ["WANDB_PROJECT"] = "sft_models"

model_name = "google/gemma-2-2b-it"
dataset_name = "yahma/alpaca-cleaned"
max_seq_length = 512

exp_base_name = model_name.split("/")[-1].replace("-SFT", "") \
    + "-" + dataset_name.split("/")[-1]
run_name = exp_base_name + "-" + time.strftime("%Y%m%d-%H%M%S")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

tokenized_path=os.path.join(DATA_ROOT,
                            "tokenized",
                            exp_base_name + "-" + str(max_seq_length))

train_dataset = load_from_disk(tokenized_path)
train_dataset = train_dataset.map(
    lambda x: {**{k: x[f"nlhf_{k}"]
                  for k in ["input_ids", "attention_mask", "labels"]}},
    batched=True,
    remove_columns=train_dataset.column_names,
)

new_model_name = "vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT"
model = AutoModelForCausalLM.from_pretrained(
    new_model_name,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    cache_dir=MODEL_ROOT,
).cuda()

# sum = 0
# for i in range(100):
#     data = {k: torch.tensor([v]).cuda() for k, v in train_dataset[i].items()}
#     with torch.no_grad():
#         outputs = model(**data)
#     sum += outputs.loss.item()
# print(sum / 100)

data = {k: torch.tensor([v]).cuda() for k, v in train_dataset[0].items()}
output = model.generate(inputs=data["input_ids"], max_length=512)

print(tokenizer.decode(train_dataset[0]["input_ids"]))
print(tokenizer.decode(output[0]))

pdb.set_trace()


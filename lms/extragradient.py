import argparse
import json
import os
import pdb
import torch

from datasets import load_dataset
from peft import get_peft_model

from configs import server_address, lora_config, get_ds_config
# from judges.pair_judge import PairJudge
from judges.local_server_judge import LocalServerJudge
from trainers.extragradient_config import ExtragradientConfig
from trainers.extragradient_trainer import ExtragradientTrainer
from utils import (
    EXP_ROOT,
    DATA_ROOT,
    AverageLossCallback,
    PushToHubCallback,
    load_model_tokenizer,
    format_names,
    transform_into_chat,
    get_latest_checkpoint,
)

argparser = argparse.ArgumentParser()
argparser.add_argument("--epochs", type=int, default=10)
argparser.add_argument("--lr", type=float, default=5e-7)
argparser.add_argument("--y_yp_mixture_coef", type=float, default=0)
argparser.add_argument("--y_yp_temperature", type=float, default=2.0)
argparser.add_argument("--y_yp_top_k", type=int, default=10)
argparser.add_argument("--y_yp_min_p", type=float, default=0.0)
argparser.add_argument("--seed", type=int, default=42)
argparser.add_argument("--local_rank", type=int, default=0)
argparser.add_argument("--load_dir", type=str, default=None)

args = argparser.parse_args()

EXP_ROOT = os.path.join(EXP_ROOT, "nlhf")
os.environ["WANDB_PROJECT"] = "nlhf"

ALG = "Extragradient"

lora = True

num_gpus = torch.cuda.device_count()

batch_size = 64
micro_batch_size = 8 if lora else 1
gen_micro_batch_size = min(2 * micro_batch_size, batch_size // num_gpus)
samples_per_prompt = 1
prefix_chunk_num = 1
gradient_accumulation_steps = batch_size // (micro_batch_size * num_gpus)

model_name = "vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT"
pref_model_name = "vectorzhou/gemma-2-2b-it-preference_dataset_mixture2_and_safe_pku-Preference"
dataset_name = "PKU-Alignment/PKU-SafeRLHF"

config = {
    "num_train_epochs": args.epochs,
    "per_device_train_batch_size": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "per_device_generate_batch_size": gen_micro_batch_size,
    "samples_per_prompt": samples_per_prompt,
    "learning_rate": args.lr,
    "weight_decay": 0.01,
    "bf16": True,
    "warmup_steps": 1000,
    "beta": 0.1,
    "y_yp_mixture_coef": args.y_yp_mixture_coef,
    "y_yp_temperature": args.y_yp_temperature,
    "y_yp_top_k": args.y_yp_top_k,
    "y_yp_min_p": args.y_yp_min_p,
    "prefix_chunk_num": prefix_chunk_num,
    "seed": args.seed,
}

ds_config = get_ds_config(
    batch_size=batch_size,
    micro_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=args.lr,
)

exp_base_name, run_name = format_names(ALG=ALG,
                                       model_name=model_name,
                                       dataset_name=dataset_name,
                                       lora=lora,
                                       config=config)

# judge = PairJudge(pref_model_name)
judge = LocalServerJudge(server_address)

model, tokenizer = load_model_tokenizer(model_name)
if lora:
    model = get_peft_model(model, lora_config)

dataset = load_dataset(dataset_name, cache_dir=DATA_ROOT)
train_dataset = transform_into_chat(dataset["train"])

# Set the output directory
if args.load_dir:
    if args.load_dir[-1] == "/":
        args.load_dir = args.load_dir[:-1]
    run_name = args.load_dir.split("/")[-1]
output_dir = os.path.join(EXP_ROOT, run_name)

# Define training configuration
training_args = ExtragradientConfig(
    run_name=run_name,
    output_dir=os.path.join(EXP_ROOT, run_name),
    **config,
    save_steps=0.1,
    save_total_limit=1,
    logging_steps=1,
    logging_dir='./logs',
    report_to="wandb",
    deepspeed=json.dumps(ds_config),
)

hf_username = os.environ.get("HF_USR_NAME", "")
hf_token = os.environ.get("HF_TOKEN", "")
hub_path = hf_username + "/" + run_name

# Determine checkpoint to resume from
resume_checkpoint = get_latest_checkpoint(output_dir)
if args.local_rank == 0:
    print(f"Resuming from {resume_checkpoint}")

# Initialize trainer
trainer = ExtragradientTrainer(
    model=model,
    judge=judge,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    callbacks=[AverageLossCallback(gradient_accumulation_steps),
               PushToHubCallback(base_repo_name=hub_path,
                                 hf_token=hf_token,
                                 is_resume=bool(resume_checkpoint))],
)

# Train the model
trainer.train(resume_from_checkpoint=resume_checkpoint)

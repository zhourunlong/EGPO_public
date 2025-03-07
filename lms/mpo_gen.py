import argparse
import csv
import json
import os
import pdb
import torch

from datasets import load_dataset
from peft import get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs import lora_config
from utils import (
    DATA_ROOT,
    load_model_tokenizer,
)

argparser = argparse.ArgumentParser()
argparser.add_argument("--epochs", type=int, nargs="+", default=None)
args = argparser.parse_args()

NUM_DATA = 100
NUM_RESPONSES = 10
BATCH_SIZE = 4

NUM_EPOCHS = 10

PICK_TYPE = "top"
NUM_TOP = 2

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

dataset_name = "PKU-Alignment/PKU-SafeRLHF"
dataset = load_dataset(dataset_name, cache_dir=DATA_ROOT)
test_dataset = dataset["test"]

base_path = "/local1/vectorzh/experiments/nlhf/MPO"

def generate_responses(all_models):
    os.makedirs("eval_results/generation", exist_ok=True)
    for (model_name, _model_name) in all_models:
        print(model_name)

        model = None

        response_path = os.path.join("eval_results", "generation", f"{_model_name}.json")
        if os.path.exists(response_path):
            with open(response_path) as f:
                responses = json.load(f)
        else:
            responses = []
        
        need_generate = False
        for i in range(NUM_DATA):
            prompt = test_dataset[i]["prompt"]
            if len(responses) <= i or responses[i]["prompt"] != prompt:
                data = {
                    "prompt": prompt,
                    "responses": [],
                }
                if len(responses) > i:
                    responses[i] = data
                else:
                    responses.append(data)
            
            responses[i]["responses"] = [x.strip() for x in responses[i]["responses"] if x.strip() != ""]
            
            if len(responses[i]["responses"]) < NUM_RESPONSES:
                need_generate = True
        
        if not need_generate:
            continue

        model, tokenizer = load_model_tokenizer(model_name)
        model = model.half().cuda()
        model.eval()

        print(f"Generating responses for {_model_name}")
        pbar = tqdm(total=NUM_DATA)
        i = 0
        multiplier = 1
        while i < NUM_DATA:
            rem_num_responses = NUM_RESPONSES - len(responses[i]["responses"])
            if rem_num_responses <= 0:
                i += 1
                multiplier = 1
                pbar.update(1)
                continue

            j = i
            while j < NUM_DATA and len(responses[j]["responses"]) == len(responses[i]["responses"]):
                j += 1
            j = min(j, i + BATCH_SIZE)
            
            prompts = [PROMPT_INPUT.format(input=responses[k]["prompt"])
                       for k in range(i, j)]
            inputs = tokenizer.batch_encode_plus(prompts,
                                                 return_tensors="pt",
                                                 max_length=512,
                                                 padding=True,
                                                 truncation=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    num_return_sequences=multiplier * rem_num_responses,
                    max_length=512,
                    do_sample=True,
                    temperature=1,
                    top_k=100,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                ).reshape(j - i, rem_num_responses, -1)

            input_length = inputs["input_ids"].shape[1]
            for k in range(len(outputs)):
                response = tokenizer.batch_decode(
                    outputs[k, :, input_length:],
                    skip_special_tokens=True
                )
                response = [r.strip() for r in response if r.strip() != ""]
                responses[i + k]["responses"].extend(response)
            
            multiplier *= 2

        with open(response_path, "w") as f:
            json.dump(responses, f)


epochs = range(NUM_EPOCHS) if args.epochs is None else args.epochs

all_models = [
    (f"{base_path}-epoch-{epoch+1}", f"MPO-epoch-{epoch+1}") for epoch in epochs
]

generate_responses(all_models)

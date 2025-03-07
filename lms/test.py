import csv
import json
import os
import pdb
import torch

from datasets import load_dataset
from tqdm import tqdm
from trl.data_utils import maybe_apply_chat_template

from judges.azure_openai_judge import AzureOpenAIJudge
from judges.pair_judge import PairJudge
from utils import (
    DATA_ROOT,
    load_model_tokenizer,
    transform_into_chat,
)

NUM_DATA = 100
NUM_RESPONSES = 10
BATCH_SIZE = 16

NUM_EPOCHS = 10

JUDGE_TYPE = "pref_model"
# JUDGE_TYPE = "gpt"

PICK_TYPE = "top"
NUM_TOP = 2

dataset_name = "PKU-Alignment/PKU-SafeRLHF"
dataset = load_dataset(dataset_name, cache_dir=DATA_ROOT)
test_dataset = transform_into_chat(dataset["test"])

PREF_MODEL = "vectorzhou/gemma-2-2b-it-preference_dataset_mixture2_and_safe_pku-Preference"

BASE_NAMES = [
    "vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT-PKU-SafeRLHF-OnlineIPO1-lora-0227213453",
    "vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT-PKU-SafeRLHF-OnlineIPO2-lora-0227214805",
    "vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT-PKU-SafeRLHF-NashMD-lora-0227215018",
    "vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT-PKU-SafeRLHF-NashMDPG-lora-0301154042",
    "dummy/MPO",
    "vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT-PKU-SafeRLHF-Extragradient-lora-0224142549",
    "vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT-PKU-SafeRLHF-Extragradient-lora-0304090509",
]
OTHER_MODELS = []
REF_MODEL = "vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT"
ALL_MODELS = OTHER_MODELS + [REF_MODEL]
for base_name in BASE_NAMES:
    for epoch in range(NUM_EPOCHS):
        ALL_MODELS.append(f"{base_name}-epoch-{epoch + 1}")

def generate_responses(all_models):
    os.makedirs("eval_results/generation", exist_ok=True)
    for model_name in all_models:
        model = None

        _model_name = model_name.split("/")[-1]
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

        try:
            model, tokenizer = load_model_tokenizer(model_name)
            model = model.cuda()
        except:
            continue

        print(f"Generating responses for {model_name}")
        i = 0
        multiplier = 1
        for i in tqdm(range(NUM_DATA)):
            while True:
                rem_num_responses = NUM_RESPONSES - len(responses[i]["responses"])
                if rem_num_responses <= 0:
                    multiplier = 1
                    break

                j = i
                while j < NUM_DATA and len(responses[j]["responses"]) == len(responses[i]["responses"]):
                    j += 1
                j = min(j, i + BATCH_SIZE)
                
                prompts = [maybe_apply_chat_template(responses[k],
                                                     tokenizer)["prompt"]
                           for k in range(i, j)]
                
                inputs = tokenizer.batch_encode_plus(prompts,
                                                     return_tensors="pt",
                                                     max_length=512,
                                                     padding=True,
                                                     truncation=True)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
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

def compare(judge_type, mode, all_models):
    assert mode in ["pairwise", "ref"]

    if judge_type == "pref_model":
        judge = PairJudge(PREF_MODEL)
    elif judge_type == "gpt":
        judge = AzureOpenAIJudge(
            resource_name="gcraoai9sw1",
            api_version="2024-08-01-preview",
            model="gpt-4o-mini_2024-07-18",
        )
    else:
        raise NotImplementedError

    responses = {}
    for model_name in ALL_MODELS:
        _model_name = model_name.split("/")[-1]
        response_path = os.path.join("eval_results", "generation", f"{_model_name}.json")
        with open(response_path) as f:
            responses[_model_name] = json.load(f)

    output_name = os.path.join("eval_results", "compare.json")
    if os.path.exists(output_name):
        with open(output_name) as f:
            compare_results = json.load(f)
    else:
        compare_results = {}

    all_models = [x for x in all_models if x != REF_MODEL]

    for i in range(len(all_models)):
        model1 = all_models[i].split("/")[-1]
        if compare_results.get(model1) is None:
            compare_results[model1] = {}

        model2_list = [REF_MODEL]
        if mode == "pairwise":
            model2_list += all_models[:i]
        model2_list = [x.split("/")[-1] for x in model2_list]

        for model2 in model2_list:
            if compare_results.get(model2, {}).get(model1) is not None:
                continue
            if compare_results[model1].get(model2) is None:
                compare_results[model1][model2] = []
            
            pair_results = compare_results[model1][model2]
            
            for k in tqdm(range(NUM_DATA)):
                prompt = responses[model1][k]["prompt"] 
                if len(pair_results) <= k or pair_results[k]["prompt"] != prompt:
                    result = {
                        "prompt": prompt,
                        "results": [],
                    }
                    if len(pair_results) > k:
                        pair_results[k] = result
                    else:
                        pair_results.append(result)
                else:
                    result = pair_results[k]
                
                rem_num = 0
                for l in range(NUM_RESPONSES):
                    if len(result["results"]) <= l or judge_type not in result["results"][l]:
                        rem_num += 1

                if rem_num == 0:
                    continue

                responses1 = responses[model1][k]["responses"][-rem_num:]
                responses2 = responses[model2][k]["responses"][-rem_num:]
                
                try:
                    _prompt = prompt[0]["content"]
                except:
                    _prompt = prompt

                judge_results = judge.judge([_prompt] * rem_num,
                                            list(zip(responses1, responses2)))
                
                ll = 0
                for l in range(NUM_RESPONSES):
                    if len(result["results"]) <= l:
                        result["results"].append({})
                    if judge_type not in result["results"][l]:
                        result["results"][l][judge_type] = judge_results[ll]
                        ll += 1
                
                pair_results[k] = result
            
                compare_results[model1][model2] = pair_results
            
            with open(output_name, "w") as f:
                json.dump(compare_results, f)
            
            avg = sum([sum([x[judge_type] for x in pair_results[k]["results"]]) for k in range(NUM_DATA)]) / (NUM_DATA * NUM_RESPONSES)

            print(f"{model1} vs {model2}", avg)
    
    return compare_results

def get_pref(compare_results, model1, model2, judge_type):
    model1 = model1.split("/")[-1]
    model2 = model2.split("/")[-1]

    try:
        pair_results = compare_results[model1][model2]
        flip = False
    except:
        pair_results = compare_results[model2][model1]
        flip = True

    avg = sum([sum([x[judge_type] for x in pair_results[k]["results"]]) for k in range(NUM_DATA)]) / (NUM_DATA * NUM_RESPONSES)
    if flip:
        avg = 1 - avg
    
    return avg

def write_csv(compare_results, all_models, judge_type, mode):
    all_models = [x for x in all_models if x != REF_MODEL]

    model2_list = [REF_MODEL]
    if mode == "pairwise":
        model2_list += all_models

    with open(f"eval_results/compare-{judge_type}-{mode}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow([""] + model2_list)
        for model1 in all_models:
            info = []
            for model2 in model2_list:
                if model1 == model2:
                    info.append("")
                else:
                    avg = get_pref(compare_results, model1, model2, judge_type)
                    info.append(f"{avg:.2%}")
            writer.writerow([model1] + info)

if __name__ == "__main__":
    generate_responses(ALL_MODELS) 
    compare_results = compare(JUDGE_TYPE, "ref", ALL_MODELS)
    write_csv(compare_results, ALL_MODELS, JUDGE_TYPE, "ref")

    all_models = OTHER_MODELS + [REF_MODEL]

    for base_name in BASE_NAMES:
        if PICK_TYPE == "last":
            for k in range(NUM_TOP):
                all_models.append(f"{base_name}-epoch-{NUM_EPOCHS - k}")
        elif PICK_TYPE == "top":
            win_rates = {}
            for epoch in range(NUM_EPOCHS):
                model1 = f"{base_name}-epoch-{epoch + 1}"
                win_rates[epoch] = get_pref(compare_results, model1, REF_MODEL, JUDGE_TYPE)
            sorted_win_rates = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)

            for k in range(NUM_TOP):
                all_models.append(f"{base_name}-epoch-{sorted_win_rates[k][0] + 1}")
        else:
            raise NotImplementedError
    
    for m in all_models:
        print(m)

    compare_results = compare(JUDGE_TYPE, "pairwise", all_models)
    write_csv(compare_results, all_models, JUDGE_TYPE, "pairwise")

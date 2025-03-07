import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import torch
import torch.nn as nn

from math import log
from tqdm import tqdm

from configs import EXP_CONFIG
from loss_funcs import LOSS_FUNCS
from policies import (
    TabularPolicy,
    NeuralPolicy
)
from utils import (
    set_random_seed,
    plot_multi_exps,
    game_value,
    response_logits,
    tabular_response_logits,
    write_data,
    read_data,
)

def dual_gap(P, beta, policy, policy_ref):
    response = tabular_response_logits(P, beta, policy, policy_ref)

    with torch.no_grad():
        gap = game_value(P, beta, response, policy.logits(), policy_ref.logits()) \
            - game_value(P, beta, policy.logits(), response, policy_ref.logits())
        
    return gap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALG = "empirical"
TYPE = "tabular"

FILES = [
    # "beta_0.01.txt",
    "beta_0.1.txt",
    # "beta_1.txt",
]

D = 10
NUM_SAMPLES = 100

FIG_PATH = f"figures/{ALG}/{TYPE}/"
DATA_PATH = f"data/{TYPE}/"
os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

FILES = [f"{DATA_PATH}{file}" for file in FILES]

picked_exps = {}

def run(file, seed):
    P, beta, names = read_data(file)
    P = torch.tensor(P, dtype=torch.float, device=device)
    set_random_seed(seed)

    RUNS, N = P.shape[:2]

    if TYPE == "neural":
        policy_ref = NeuralPolicy(RUNS, N, 3, D, device)
    elif TYPE == "tabular":
        policy_ref = TabularPolicy(RUNS, N, device)
    else:
        raise ValueError(f"Invalid policy type: {TYPE}")
    policy_ref.random_init()

    # write_data(P, beta, f"{DATA_PATH}tmp.txt")

    experiments = {}
    lr = None
    for exp_config in EXP_CONFIG[ALG][TYPE]:
        if abs(exp_config["beta"] - beta) < 1e-6:
            beta = exp_config["beta"]
            iters = exp_config["iters"]
            lr = exp_config["lr"]
            break
    
    if lr is None:
        raise ValueError(f"Beta={beta} not in config!")

    for name, loss_func in LOSS_FUNCS[ALG].items():
        true_lr = 2 * lr / (beta * N)

        full_name = f"{name}".strip("_")
        print(f"Beta {beta}: Running {full_name} with lr={lr}...")

        gaps = torch.zeros(RUNS, iters + 1, device=device)
        policy = copy.deepcopy(policy_ref)

        optimizer = torch.optim.SGD(policy.parameters(), lr)

        for i in tqdm(range(iters)):
            gaps[:, i] = dual_gap(P, beta, policy, policy_ref)

            optimizer.zero_grad()
            
            loss = loss_func(P=P,
                             beta=beta,
                             policy=policy,
                             policy_ref=policy_ref,
                             true_lr=true_lr,
                             num_samples=NUM_SAMPLES,
                             optimizer=optimizer)
            loss.backward()

            optimizer.step()

        gaps[:, iters] = dual_gap(P, beta, policy, policy_ref)

        for run in range(RUNS):
            key = f"beta={beta}_{names[run]}"
            if key not in experiments:
                experiments[key] = {}
            
            experiments[key].update({
                full_name: {
                    "Dual Gap": gaps[run].cpu().numpy(),
                }
            })

    file_name = file.split("/")[-1]
    file_name = file_name[:file_name.rfind(".")]
    print("Plotting...")
    plot_multi_exps(experiments,
                    path=FIG_PATH,
                    name_suffix=file_name)
    
    for key, val in experiments.items():
        picked_exps[key] = val
        break

if __name__ == "__main__":
    for file in FILES:
        run(file, 42)

    print("Plotting picked experiments...")
    plot_multi_exps(picked_exps,
                    path=FIG_PATH,
                    name_suffix="picked",
                    one_row=True)
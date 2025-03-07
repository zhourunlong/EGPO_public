import argparse
import copy
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
    gen_preference_matrix,
    plot_multi_exps,
    game_value,
    response_logits,
    tabular_response_logits,
    write_data,
)

def dual_gap(P, beta, policy, policy_ref):
    response = tabular_response_logits(P, beta, policy, policy_ref)

    with torch.no_grad():
        gap = game_value(P, beta, response, policy.logits(), policy_ref.logits()) \
            - game_value(P, beta, policy.logits(), response, policy_ref.logits())
        
    return gap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

argparser = argparse.ArgumentParser()
argparser.add_argument("--alg", type=str, default="exact")
argparser.add_argument("--type", type=str, default="tabular")

args = argparser.parse_args()

ALG = args.alg
TYPE = args.type

RUNS = 10
N = 10 if TYPE == "tabular" else 100
DEPTH = 3
DIM = 10
NUM_SAMPLES = 100

FIG_PATH = f"figures/{ALG}/{TYPE}/"
DATA_PATH = f"data/{TYPE}/"
os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

def run(seed, override_lr=None):
    set_random_seed(seed)

    if TYPE == "neural":
        policy_ref = NeuralPolicy(RUNS, N, DEPTH, DIM, device)
    elif TYPE == "tabular":
        policy_ref = TabularPolicy(RUNS, N, device)
    else:
        raise ValueError(f"Invalid policy type: {TYPE}")
    policy_ref.random_init()

    P = gen_preference_matrix(RUNS, N, "random").to(device)

    write_data(P, 0, f"{DATA_PATH}seed{str(seed)}.txt")
    
    picked_exps = {}
    for exp_config in EXP_CONFIG[ALG][TYPE]:
        beta = exp_config["beta"]
        iters = exp_config["iters"]
        lr = override_lr if override_lr else exp_config["lr"]
        true_lr = 2 * lr / (beta * N)

        experiments = {}
        for name, loss_func in LOSS_FUNCS[ALG].items():
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
                key = f"beta{beta}_lr{lr}_seed{seed}_run{run}"
                latex_key = f"$\\beta={beta}$ $\eta={lr}$ run#{run}"
                if key not in experiments:
                    experiments[key] = {"latex_key": latex_key}
                
                experiments[key].update({
                    full_name: {
                        "Dual Gap": gaps[run].cpu().numpy(),
                    }
                })

        key = f"beta{beta}_lr{lr}_seed{seed}_run0"
        picked_exps[key] = experiments[key].copy()
        print("Plotting...")
        plot_multi_exps(experiments, path=FIG_PATH)

    if len(picked_exps) > 1:
        print("Plotting picked experiments...")
        plot_multi_exps(picked_exps,
                        path=FIG_PATH,
                        one_row=True,
                        name_suffix="_picked")

if __name__ == "__main__":
    # for seed in [42, 142857, 2225393, 20000308, 2018011309]:
    for seed in [2018011309]:
        # for lr in [1e-3, 5e-4, 1e-4]:
        #       run(seed, lr)
        run(seed)
import base64
import copy
import hashlib
import inspect
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import jacrev, vmap

from scipy import stats
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def write_data(P, beta, file_name):
    runs, n = P.shape[:2]
    with open(file_name, 'w') as f:
        f.write(f"{runs} {n} {beta}\n")
        for i in range(runs):
            f.write(f"Run {i}\n")
            for j in range(n):
                for k in range(n):
                    f.write(f"{P[i, j, k]} ")
                f.write("\n")

def read_data(file_name):
    with open(file_name, 'r') as f:
        runs, n, beta = map(float, f.readline().strip().split())
        runs, n = int(runs), int(n)
        P = []
        names = []
        if runs == -1:
            runs = 999999999
        for i in range(runs):
            name = f.readline()
            if not name:
                break
            names.append(name.strip())
            _P = np.zeros((n, n))
            for j in range(n):
                _P[j] = np.array(list(map(float, f.readline().strip().split())))
            P.append(_P)
    return np.stack(P), beta, names

def extract_gradients(optimizer, coef=1.0):
    grads_dict = {}
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                grads_dict[id(param)] = coef * param.grad.clone()
    return grads_dict

def restore_gradients(optimizer, grads_dict):
    for group in optimizer.param_groups:
        for param in group['params']:
            if id(param) in grads_dict:
                param.grad = grads_dict[id(param)].clone()

def plot_multi_exps(experiments, path, window_size=10, max_points=1000, num_markers=10, eps=1e-6, one_row=False, name_suffix=""):
    legend_fontsize = 12
    fontsize = 14
    title_fontsize = 16
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': title_fontsize,
        'xtick.labelsize': fontsize - 2,
        'ytick.labelsize': fontsize - 2,
        'legend.fontsize': legend_fontsize,
    })

    num_experiments = len(experiments)
    if one_row:
        ncols = num_experiments
        nrows = 1
        figsize = (6 * ncols, 6)
    else:
        ncols = 2
        nrows = num_experiments // ncols + (num_experiments % ncols > 0)
        figsize = (20, 6 * nrows)

    colors = plt.get_cmap('tab10').colors
    desc_color_map = {}
    color_index = 0

    markevery, offset = None, None

    keys = list(list(experiments.values())[0].values())[-1].keys()
    
    for key in keys:
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        is_key_gap = any(x in key for x in ["Diff", "Gap"])
        axs = axs.flatten()
        for i, (exp_name, experiment) in enumerate(experiments.items()):
            main_name = exp_name[:exp_name.rfind("_")]
            x_max = 0
            num_algs = len(experiment) - 1
            for j, (desc, data) in enumerate(experiment.items()):
                if j == 0:
                    continue

                base_desc = desc.replace('_empirical', '')
                if base_desc not in desc_color_map:
                    desc_color_map[base_desc] = colors[color_index % len(colors)]
                    color_index += 1
                color = desc_color_map[base_desc]

                values = data[key]
                x_max = max(x_max, values.shape[0])

                if markevery is None:
                    markevery = min(x_max, max_points) // num_markers
                    offset = markevery // num_algs

                if window_size > 1:
                    values = np.convolve(values, np.ones(window_size), 'valid') / window_size
                if is_key_gap:
                    idx = np.argmax(values < eps)
                    if idx > 0:
                        values = values[:idx]
                epochs = np.arange(1, values.shape[0] + 1)

                if values.shape[0] > max_points:
                    idx = np.linspace(0, values.shape[0] - 1, max_points).astype(int)
                    epochs = epochs[idx]
                    values = values[idx]

                axs[i].plot(epochs,
                            values,
                            label=desc,
                            color=color,
                            linestyle="-" if "empirical" in desc else "--",
                            marker='x',
                            markevery=(offset * (j - 1), markevery),
                            markersize=6)

            axs[i].set_xlim(-0.05 * x_max, 1.05 * x_max)
            axs[i].set_xlabel('# Updates: $t$')
            axs[i].set_ylabel(key)
            axs[i].set_title(f'{experiment["latex_key"]} - {key}')
            axs[i].legend()
            if is_key_gap:
                axs[i].set_yscale('log')

        for i in range(len(experiments), len(axs)):
            fig.delaxes(axs[i])
    
        plt.tight_layout()
        plt.savefig(os.path.join(
            path,
            f'{key.lower().replace(" ", "_")}_{main_name}{name_suffix}.pdf'
        ))
        plt.close(fig)

def gen_preference_matrix(runs, n, type):
    diag = 0.5 * torch.stack([torch.eye(n) for _ in range(runs)])

    if type == "random":
        lower = torch.rand((runs, n, n))
    elif type == "tie":
        lower = 0.5 * torch.ones((runs, n, n))
    elif type == "lower_win":
        lower = torch.ones((runs, n, n))

    P = diag.clone()
    for i in range(1, n):
        P[:, i, :i] = lower[:, i, :i]
        P[:, :i, i] = 1 - lower[:, i, :i]
    
    assert torch.norm(P + P.transpose(-1, -2) - torch.ones((runs, n, n))) < 1e-6

    return P

def KL(logits_p, logits_q):
    p = torch.softmax(logits_p, dim=-2)
    log_p = torch.log_softmax(logits_p, dim=-2)
    log_q = torch.log_softmax(logits_q, dim=-2)
    return (p * (log_p - log_q)).sum(dim=-2).view(-1)

def game_value(P, beta, logits_1, logits_2, logits_ref):
    pi_1 = torch.softmax(logits_1, dim=-2)
    pi_2 = torch.softmax(logits_2, dim=-2)
    return (pi_1.transpose(-1, -2) @ P @ pi_2).view(-1) - beta * KL(logits_1, logits_ref) + beta * KL(logits_2, logits_ref)

def dual_gap(P, theta_1, theta_2, beta, thetas_ref):
    with torch.no_grad():
        pi_1 = torch.softmax(theta_1, dim=-2)
        pi_2 = torch.softmax(theta_2, dim=-2)

        response_1 = thetas_ref + P @ pi_2 / beta
        response_2 = thetas_ref + P @ pi_1 / beta

        return game_value(P, beta, response_1, theta_2, thetas_ref) \
            - game_value(P, beta, theta_1, response_2, thetas_ref)

def calc_pram_diff(P, thetas, beta, thetas_ref):
    with torch.no_grad():
        pi = torch.softmax(thetas, dim=-2)
        diff = (thetas - thetas_ref - P @ pi / beta).squeeze(-1)
        bias = torch.mean(diff, dim=-1, keepdim=True)
        return torch.norm(diff - bias, dim=-1)

def jacobian(policy, prefs, beta, thetas_ref, **kwargs):
    def func(theta, theta_ref, pref):
        return beta * (theta - theta_ref) - pref @ torch.softmax(theta, dim=-2)
    
    with torch.no_grad():
        jacobian = vmap(jacrev(func, argnums=0))(policy.thetas, thetas_ref, prefs).squeeze(2, 4)


def tabular_response_logits(P, beta, policy, policy_ref):
    with torch.no_grad():
        return policy_ref.logits() + P @ policy() / beta

def response_logits(P, beta, policy, policy_ref):
    cur = copy.deepcopy(policy_ref)
    optimizer = optim.Adam(cur.parameters(), lr=1e-2)

    logits = policy.logits().detach().clone()
    logits_ref = policy_ref.logits().detach().clone()
    prev_loss = 0
    while True:
        optimizer.zero_grad()
        loss = -torch.sum(game_value(P, beta, cur.logits(), logits, logits_ref) ** 2)
        loss.backward(retain_graph=True)
        optimizer.step()

        if abs(loss - prev_loss) < 1e-6:
            break
        prev_loss = loss

    return cur.logits().detach().clone()
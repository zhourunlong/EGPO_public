import torch

from utils import extract_gradients, restore_gradients

def online_ipo_1(P, beta, policy, policy_ref, **kwargs):
    runs, num_actions = P.shape[:2]

    log_pi_t = torch.log_softmax(policy.logits(), dim=-2)
    log_pi_ref = torch.log_softmax(policy_ref.logits(), dim=-2)
    
    log_rel_probs = log_pi_t - log_pi_ref
    log_probs_contrast = log_rel_probs.view(runs, -1, 1) - log_rel_probs.view(runs, 1, -1)

    prefs = P @ policy().detach()
    prefs_contrast = prefs.view(runs, -1, 1) - prefs.view(runs, 1, -1)

    return torch.sum((log_probs_contrast - prefs_contrast / beta) ** 2) / (num_actions ** 2)

def online_ipo_1_sample(P, beta, policy, policy_ref, num_samples, **kwargs):
    runs, num_actions = P.shape[:2]
    axis = torch.arange(runs).view(-1, 1)

    y = torch.randint(0, num_actions, (runs, num_samples))
    yp = torch.randint(0, num_actions, (runs, num_samples))
    ypp = torch.multinomial(policy().squeeze(-1),
                            num_samples,
                            replacement=True)
    
    log_pi_t = torch.log_softmax(policy.logits(), dim=-2).squeeze(-1)
    log_pi_ref = torch.log_softmax(policy_ref.logits(), dim=-2).squeeze(-1)
    
    log_rel_probs = log_pi_t - log_pi_ref
    log_probs_contrast = log_rel_probs[axis, y] - log_rel_probs[axis, yp]

    prefs_contrast = P[axis, y, ypp] - P[axis, yp, ypp]

    return torch.sum((log_probs_contrast - prefs_contrast / beta) ** 2) / num_samples

def online_ipo_2(P, beta, policy, policy_ref, **kwargs):
    runs = P.shape[0]

    log_pi_t = torch.log_softmax(policy.logits(), dim=-2)
    log_pi_ref = torch.log_softmax(policy_ref.logits(), dim=-2)
    
    log_rel_probs = log_pi_t - log_pi_ref
    log_probs_contrast = log_rel_probs.view(runs, -1, 1) - log_rel_probs.view(runs, 1, -1)

    pi = policy().detach()
    return (pi.transpose(1, 2) @ ((log_probs_contrast - (P - 0.5) / beta) ** 2) @ pi).sum()

def online_ipo_2_sample(P, beta, policy, policy_ref, num_samples, **kwargs):
    runs = P.shape[0]
    axis = torch.arange(runs).view(-1, 1)

    y = torch.multinomial(policy().squeeze(-1),
                          num_samples,
                          replacement=True)
    yp = torch.multinomial(policy().squeeze(-1),
                           num_samples,
                           replacement=True)
    
    log_pi_t = torch.log_softmax(policy.logits(), dim=-2).squeeze(-1)
    log_pi_ref = torch.log_softmax(policy_ref.logits(), dim=-2).squeeze(-1)
    
    log_rel_probs = log_pi_t - log_pi_ref
    log_probs_contrast = log_rel_probs[axis, y] - log_rel_probs[axis, yp]

    prefs_contrast = P[axis, y, yp] - 0.5

    return torch.sum((log_probs_contrast - prefs_contrast / beta) ** 2) / num_samples

def extragradient(P, beta, policy, policy_ref, optimizer, **kwargs):
    runs, num_actions = P.shape[:2]

    log_pi_t = torch.log_softmax(policy.logits(), dim=-2)
    log_pi_ref = torch.log_softmax(policy_ref.logits(), dim=-2)
    
    log_rel_probs = log_pi_t - log_pi_ref
    log_probs_contrast = log_rel_probs.view(runs, -1, 1) - log_rel_probs.view(runs, 1, -1)

    prefs = P @ policy().detach()
    prefs_contrast = prefs.view(runs, -1, 1) - prefs.view(runs, 1, -1)

    loss = torch.sum((log_probs_contrast - prefs_contrast / beta) ** 2) / (num_actions ** 2)
    loss.backward()

    grads_dict = extract_gradients(optimizer, -1.0)
    optimizer.step()
    optimizer.zero_grad()

    prefs = P @ policy().detach()
    prefs_contrast = prefs.view(runs, -1, 1) - prefs.view(runs, 1, -1)

    restore_gradients(optimizer, grads_dict)
    optimizer.step()
    optimizer.zero_grad()

    log_pi_t = torch.log_softmax(policy.logits(), dim=-2)
    log_pi_ref = torch.log_softmax(policy_ref.logits(), dim=-2)
    
    log_rel_probs = log_pi_t - log_pi_ref
    log_probs_contrast = log_rel_probs.view(runs, -1, 1) - log_rel_probs.view(runs, 1, -1)

    return torch.sum((log_probs_contrast - prefs_contrast / beta) ** 2) / (num_actions ** 2)

def extragradient_sample(P, beta, policy, policy_ref, num_samples, optimizer, **kwargs):
    runs, num_actions = P.shape[:2]
    axis = torch.arange(runs).view(-1, 1)

    y = torch.randint(0, num_actions, (runs, num_samples))
    yp = torch.randint(0, num_actions, (runs, num_samples))
    ypp = torch.multinomial(policy().squeeze(-1),
                            num_samples,
                            replacement=True)
    
    log_pi_t = torch.log_softmax(policy.logits(), dim=-2).squeeze(-1)
    log_pi_ref = torch.log_softmax(policy_ref.logits(), dim=-2).squeeze(-1)
    
    log_rel_probs = log_pi_t - log_pi_ref
    log_probs_contrast = log_rel_probs[axis, y] - log_rel_probs[axis, yp]

    prefs_contrast = P[axis, y, ypp] - P[axis, yp, ypp]

    loss = torch.sum((log_probs_contrast - prefs_contrast / beta) ** 2) / num_samples
    loss.backward()

    grads_dict = extract_gradients(optimizer, -1.0)
    optimizer.step()
    optimizer.zero_grad()

    y = torch.randint(0, num_actions, (runs, num_samples))
    yp = torch.randint(0, num_actions, (runs, num_samples))
    ypp = torch.multinomial(policy().squeeze(-1),
                            num_samples,
                            replacement=True)
    
    prefs_contrast = P[axis, y, ypp] - P[axis, yp, ypp]

    restore_gradients(optimizer, grads_dict)
    optimizer.step()
    optimizer.zero_grad()

    log_pi_t = torch.log_softmax(policy.logits(), dim=-2).squeeze(-1)
    log_pi_ref = torch.log_softmax(policy_ref.logits(), dim=-2).squeeze(-1)

    log_rel_probs = log_pi_t - log_pi_ref
    log_probs_contrast = log_rel_probs[axis, y] - log_rel_probs[axis, yp]

    return torch.sum((log_probs_contrast - prefs_contrast / beta) ** 2) / num_samples

def nash_md(P, beta, policy, policy_ref, true_lr, **kwargs):
    runs, num_actions = P.shape[:2]

    log_pi_t = torch.log_softmax(policy.logits(), dim=-2)
    log_pi_ref = torch.log_softmax(policy_ref.logits(), dim=-2)

    log_rel_probs = log_pi_t - log_pi_ref
    log_probs_contrast = log_rel_probs.view(runs, -1, 1) - log_rel_probs.view(runs, 1, -1)
    
    logits_mixture = (1 - true_lr * beta) * log_pi_t.detach() + true_lr * beta * log_pi_ref.detach()
    pi_mixture = torch.softmax(logits_mixture, dim=-2)

    prefs = P @ pi_mixture
    prefs_contrast = prefs.view(runs, -1, 1) - prefs.view(runs, 1, -1)

    return torch.sum((log_probs_contrast - prefs_contrast / beta) ** 2) / (num_actions ** 2)

def nash_md_sample(P, beta, policy, policy_ref, true_lr, num_samples, **kwargs):
    runs, num_actions = P.shape[:2]
    axis = torch.arange(runs).view(-1, 1)

    y = torch.randint(0, num_actions, (runs, num_samples))
    yp = torch.randint(0, num_actions, (runs, num_samples))
    
    log_pi_t = torch.log_softmax(policy.logits(), dim=-2).squeeze(-1)
    log_pi_ref = torch.log_softmax(policy_ref.logits(), dim=-2).squeeze(-1)
    
    log_rel_probs = log_pi_t - log_pi_ref
    log_probs_contrast = log_rel_probs[axis, y] - log_rel_probs[axis, yp]

    logits_mixture = (1 - true_lr * beta) * log_pi_t.detach() + true_lr * beta * log_pi_ref.detach()
    pi_mixture = torch.softmax(logits_mixture, dim=-2)
    ypp = torch.multinomial(pi_mixture.squeeze(-1),
                            num_samples,
                            replacement=True)

    prefs_contrast = P[axis, y, ypp] - P[axis, yp, ypp]

    return torch.sum((log_probs_contrast - prefs_contrast / beta) ** 2) / num_samples

def nash_md_pg(P, beta, policy, policy_ref, true_lr, **kwargs):
    log_pi_t = torch.log_softmax(policy.logits(), dim=-2)
    log_pi_ref = torch.log_softmax(policy_ref.logits(), dim=-2)
    
    logits_mixture = (1 - true_lr * beta) * log_pi_t.detach() + true_lr * beta * log_pi_ref.detach()
    pi_mixture = torch.softmax(logits_mixture, dim=-2)

    prefs = P @ pi_mixture
    pg_item_loss = (prefs - beta * (log_pi_t.detach() - log_pi_ref.detach())) * log_pi_t

    return -torch.sum(policy().detach().transpose(1, 2) @ pg_item_loss)

def nash_md_pg_sample(P, beta, policy, policy_ref, true_lr, num_samples,**kwargs):
    runs, num_actions = P.shape[:2]
    axis = torch.arange(runs).view(-1, 1)
    
    log_pi_t = torch.log_softmax(policy.logits(), dim=-2)
    log_pi_ref = torch.log_softmax(policy_ref.logits(), dim=-2)
    
    logits_mixture = (1 - true_lr * beta) * log_pi_t.detach() + true_lr * beta * log_pi_ref.detach()
    pi_mixture = torch.softmax(logits_mixture, dim=-2)

    y = torch.multinomial(policy().squeeze(-1),
                          num_samples,
                          replacement=True)
    yp = torch.multinomial(pi_mixture.squeeze(-1),
                           num_samples,
                           replacement=True)
    
    log_rel_probs = (log_pi_t - log_pi_ref).detach()

    pg_item_loss = (P[axis, y, yp] - beta * log_rel_probs[axis, y].squeeze(-1)) * log_pi_t[axis, y].squeeze(-1)

    return -torch.sum(pg_item_loss) / num_samples

EXACT_LOSS_FUNCS = {
    "Online IPO 1 (OMD)": online_ipo_1,
    "Online IPO 2": online_ipo_2,
    "Nash-MD": nash_md,
    "Nash-MD-PG": nash_md_pg,
    "EGPO": extragradient,
}

EMPIRICAL_LOSS_FUNCS = {
    "Online IPO 1 (OMD) Empirical": online_ipo_1_sample,
    "Online IPO 2 Empirical": online_ipo_2_sample,
    "Nash-MD Empirical": nash_md_sample,
    "Nash-MD-PG Empirical": nash_md_pg_sample,
    "EGPO Empirical": extragradient_sample,
}

LOSS_FUNCS = {
    "exact": EXACT_LOSS_FUNCS,
    "empirical": EMPIRICAL_LOSS_FUNCS,
}
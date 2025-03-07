import torch
import torch.nn as nn

class TabularPolicy(nn.Module):
    def __init__(self, runs, num_actions, device):
        super(TabularPolicy, self).__init__()
        self.thetas = nn.Parameter(
            torch.zeros((runs, num_actions, 1), device=device),
            requires_grad=True,
        )
        self.device = device

    def forward(self):
        return torch.softmax(self.thetas, dim=-2)
    
    def logits(self):
        return self.thetas
    
    def random_init(self):
        nn.init.normal_(self.thetas, 0, 1)

    def update_thetas(self, thetas):
        with torch.no_grad():
            bias = torch.mean(thetas, dim=-2, keepdim=True)
        self.thetas = nn.Parameter((thetas - bias).clone(), requires_grad=True)


class NeuralPolicy(nn.Module):
    def __init__(self, runs, num_actions, depth, hidden_dim, device):
        super(NeuralPolicy, self).__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))
        
        self.output_layer = nn.Linear(hidden_dim, num_actions, device=device)
        self.virtual_input = nn.Parameter(torch.normal(0, 1, (runs, hidden_dim), device=device), requires_grad=True)
        
        self.activation = nn.ReLU()

        self.device = device
    
    def random_init(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self):
        return torch.softmax(self.logits(), dim=-2)
    
    def logits(self):
        x = self.virtual_input
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x).unsqueeze(-1)

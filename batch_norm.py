import torch
import torch.nn as nn 
torch.manual_seed(42)

class BatchNorm(nn.Module):
    def __init__(self, channels, momentum):
        super().__init__()
        self.momentum = momentum 
        self.channels = channels
        self.gammas = nn.Parameter(torch.ones((1, self.channels)))
        self.betas = nn.Parameter(torch.zeros((1, self.channels)))
        
        self.running_mean = torch.zeros((1, self.channels)) # 1 x C
        self.running_var = torch.ones((1, self.channels)) # 1 x C
        self.eps = 1e-5
    
    def forward(self, inputs: torch.Tensor):
        # inputs B x C
        if not inputs.requires_grad:
            inputs_hat = inputs.sub(self.running_mean).div(self.running_var.add(self.eps).sqrt()).mul(self.gammas).add(self.betas)
            return inputs_hat
        
        mean = inputs.mean(dim=0)
        var = (inputs - mean).pow(2).mean(dim=0)
        inputs_hat = (inputs - mean) * self.gammas / (torch.sqrt(var + self.eps)) + self.betas
        inputs_hat2 = inputs.sub(mean).div(var.add(self.eps).sqrt()).mul(self.gammas).add(self.betas)
        self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean 
        self.running_var = (1-self.momentum) * self.running_var + self.momentum * var 
        return inputs_hat
torch_batch = nn.BatchNorm1d(4)
batch_norm = BatchNorm(4, 0.1)
input = nn.Parameter(torch.rand(3,4))
res = batch_norm(input)
res2 = torch_batch(input)
print(res)
0.1571
-0.157
0.02468041
0.024649
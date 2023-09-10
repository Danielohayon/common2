import torch
import torch.nn as nn 


class SpectralNorm():
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
    
    def normalize(self):
        for param_group in self.model.parameters():
            for param in param_group:
                param.data.div_(param.data.abs().max())

model = nn.Sequential(nn.Linear(4,2), nn.ReLU(), nn.Linear(2,1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

spectral_normalizer = SpectralNorm(model)
spectral_normalizer.normalize()
out = model(torch.rand(2,4))

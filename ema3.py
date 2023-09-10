import torch 
import torch.nn as nn 
import torch
import copy
from collections import OrderedDict
import typing

model = nn.Sequential(nn.Linear(4, 4,), nn.ReLU(), nn.Linear(4, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

class EMAWrapper():
    def __init__(self, model: nn.Module, beta: float) -> None:
        self.model = model
        self.beta = beta 
        self.shadow_model = copy.deepcopy(model)
    
    def update(self):
        if not self.model.training:
            raise Exception("Model should be in trainign when updating EMA")
        
        assert OrderedDict(self.model.named_parameters()).keys == self.shadow.keys()
        shadow_params = OrderedDict(self.shadow_model.named_parameters())
        model_params = OrderedDict(self.model.named_parameters())
        with torch.no_grad():
            for name, parameter in shadow_params.items():
                parameter.data = parameter.data * self.beta + model_params[name].data * (1-self.beta)
        
        shadow_buffers = OrderedDict(self.shadow_model.named_buffers())
        model_buffers = OrderedDict(self.model.named_buffers())
        assert shadow_buffers.keys() == model_buffers.keys()
        for name, buffer in shadow_buffers:
            buffer.copy_(model_buffers[name])



ema_wrapper = EMAWrapper(model, 0.9)

print()

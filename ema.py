import torch
import copy
from torch import nn
torch.manual_seed(42)

class EMASGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = copy.deepcopy(self.param_groups)
        self.itterations = 1

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        for old_param_group, new_param_group in zip(self.weights, self.param_groups):
            for old_param, new_param in zip(old_param_group['params'], new_param_group['params']):
                new_param.data = (new_param.data + (old_param.data * self.itterations))/(self.itterations + 1)

        self.itterations += 1 

model = nn.Sequential(nn.Linear(4, 4,), nn.ReLU(), nn.Linear(4, 1))
optimizer = EMASGD(model.parameters(), lr=0.05)

[print(i) for i in model.parameters()]
input = torch.rand((2,4))
output = model(input)
gt = torch.rand((2,1))
loss = torch.nn.MSELoss()(output, gt)
loss.backward()
optimizer.step()
[print(i) for i in model.parameters()]

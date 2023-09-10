import torch
import copy
from torch import nn
torch.manual_seed(42)

class EMAWrapper():
    def __init__(self,optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.weights = copy.deepcopy(self.optimizer.param_groups)
        self.itterations = 1

    def step(self, *args, **kwargs):
        self.optimizer.step(*args, **kwargs)
        for old_param_group, new_param_group in zip(self.weights, self.optimizer.param_groups):
            for old_param, new_param in zip(old_param_group['params'], new_param_group['params']):
                new_param.data = (new_param.data + (old_param.data * self.itterations))/(self.itterations + 1)

        self.itterations += 1 

model = nn.Sequential(nn.Linear(4, 4,), nn.ReLU(), nn.Linear(4, 1))
raw_optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
optimizer = EMAWrapper(optimizer=raw_optimizer)

[print(i) for i in model.parameters()]
input = torch.rand((2,4))
output = model(input)
gt = torch.rand((2,1))
loss = torch.nn.MSELoss()(output, gt)
loss.backward()
optimizer.step()
[print(i) for i in model.parameters()]


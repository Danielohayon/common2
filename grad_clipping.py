import copy
import torch 
import torch.nn as nn 
torch.manual_seed(42)


def grad_clipping_hook(module, input_grad, output_grad):

    new_out_grad = copy.deepcopy(input_grad)
    for i in range(len(input_grad)):
        locations = input_grad[i] > 0.1
        new_out_grad[i][locations] = 0.1
    return new_out_grad

class GradClippingWrapper():
    def __init__(self, optimizer: torch.optim.Optimizer, max_grad:float = 0.2):
        super().__init__()
        self.optimizer = optimizer
        self.max_grad = max_grad

    def step(self):
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                positive_locations = param.grad > self.max_grad
                negative_locations = param.grad < -self.max_grad
                param.grad[positive_locations] = self.max_grad
                param.grad[negative_locations] = -self.max_grad

        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()

model = nn.Sequential(nn.Linear(4, 4,), nn.ReLU(), nn.Linear(4, 1))
optimizer1 = torch.optim.SGD(model.parameters(), lr=0.05)

optimizer = GradClippingWrapper(optimizer1, 0.1)
# model.register_backward_hook(grad_clipping_hook)
[print(i.grad) for i in model.parameters()]
optimizer.zero_grad()
input = torch.rand((2,4))
output = model(input)
gt = torch.rand((2,1))
loss = torch.nn.MSELoss()(output, gt)
loss.backward()
optimizer.step()
[print(i.grad) for i in model.parameters()]

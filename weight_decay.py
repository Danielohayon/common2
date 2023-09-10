import torch
import torch.nn as nn
torch.manual_seed(42)

model = nn.Sequential(nn.Linear(4,4), nn.ReLU(), nn.Linear(4,1))
input = torch.rand((2,4))

class WeightDecayWrapper(nn.Module):
    def __init__(self, loss, alpha, model):
        super().__init__()
        self.loss_class = loss
        self.alpha = alpha
        self.model = model
    
    def get_curr_sum_of_weights(self):
        curr_sum = 0.0
        for name, parameter in dict(self.model.named_parameters()).items():
            curr_sum += torch.sum(parameter**2)
        return curr_sum
    
    def forward(self, outputs, labels):
        data_loss = self.loss_class(outputs, labels)
        weight_loss = self.alpha * 0.5 * self.get_curr_sum_of_weights()
        return data_loss + weight_loss

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        return inputs * (1/(1+torch.exp(-inputs)))
s = Swish()
out = s(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
print(out)

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ouputs, labels):
        return torch.sum(torch.pow(ouputs - labels))

class CrossEntorpyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, labels):
        return labels * torch.log(outputs)
        

# criterion = WeightDecayWrapper(nn.MSELoss(), 0.1, model)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0)
optimizer.zero_grad()
[print(i) for i in model.parameters()]
output = model(input)
labels = torch.rand((2,1))
loss = criterion(output, labels)
loss.backward()

optimizer.step()
[print(i) for i in model.parameters()]

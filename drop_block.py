import torch
import torch.nn as nn

class DropBlock(nn.Module):
    def __init__(self, window_size = 3):
        super().__init__()
        self.window_size = window_size 
        self.zero_window = torch.zeros((window_size, window_size))

    def forward(self, input):
        B, C, d1, d2 = input.shape
        random_row = torch.randint(0, d1 - self.window_size, (1,))
        random_col = torch.randint(0, d2 - self.window_size, (1,))
        input[:, :, random_row:random_row+self.window_size, random_col:random_col+self.window_size] = 0
        return input
input = torch.rand((2, 3, 9, 9))
model = nn.Sequential(nn.Conv2d(3, 6, 3, 1), DropBlock(3), nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(6, 3), nn.LogSoftmax(1))
optim = torch.optim.SGD(model.parameters(), lr=0.1)
out = model(input)
label = torch.rand((2,3))
print(out.shape)
print(out)
loss = nn.CrossEntropyLoss()(out, label)
loss.backward()
optim.step()


# implementing from scratch: DropBlock
# implementing DropBlock from scratch
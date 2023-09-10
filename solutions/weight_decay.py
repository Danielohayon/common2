import torch
import torch.nn as nn
import torch.optim as optim

# Random seed for reproducibility
torch.manual_seed(42)

# Generate some dummy data
x = torch.randn(100, 1) * 10
y = 2.0 * x + torch.randn(100, 1) * 2

# Simple Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)

# Initialize the model
model = LinearRegression()

# Define a custom loss function with L2 regularization
def custom_mse_loss_with_weight_decay(output, target, model, weight_decay):
    mse_loss = nn.MSELoss()(output, target)
    l2_reg = torch.tensor(0.)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss = mse_loss + weight_decay * l2_reg
    return loss

# Hyperparameters
learning_rate = 0.01
weight_decay = 0.01
num_epochs = 100

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
losses = []
for epoch in range(num_epochs):
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(x)
    
    # Compute the loss
    loss = custom_mse_loss_with_weight_decay(outputs, y, model, weight_decay)
    losses.append(loss.item())
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()

# Return the trained model parameters and the loss curve
model_params = list(model.parameters()), losses[-1]
print(model_params)

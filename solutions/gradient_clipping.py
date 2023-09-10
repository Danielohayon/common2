import torch
import torch.nn as nn
import torch.optim as optim
# Simple Linear Regression Model
# Random seed for reproducibility
torch.manual_seed(42)

# Generate some dummy data
x = torch.randn(100, 1) * 10
y = 2.0 * x + torch.randn(100, 1) * 2

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)
# Let's test the custom gradient clipping function
# First, we'll generate dummy gradients for our model
dummy_model = LinearRegression()
dummy_optimizer = optim.SGD(dummy_model.parameters(), lr=0.01)
dummy_loss = nn.MSELoss()(dummy_model(x), y)
dummy_loss.backward()

def custom_gradient_clipping_v2(model, clip_value):
    # Compute the norm of all gradients
    total_norm = torch.sqrt(sum(p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None))
    
    if total_norm > clip_value:
        # Scale gradients if the norm exceeds the clip value
        scale_factor = clip_value / (total_norm + 1e-6)  # Adding a small value to avoid division by zero
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(scale_factor)
    return torch.sqrt(sum(p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None))


# Artificially increase the gradients by a large factor
for p in dummy_model.parameters():
    if p.grad is not None:
        p.grad.data.zero_()
        p.grad.data.add_(100.0)

# Check the norm before clipping
pre_clip_norm_large_v2 = torch.sqrt(sum(p.grad.data.norm(2) ** 2 for p in dummy_model.parameters() if p.grad is not None))

# Clip the gradients
post_clip_norm_large_v2 = custom_gradient_clipping_v2(dummy_model, clip_value=1.0)

print(pre_clip_norm_large_v2, post_clip_norm_large_v2)
import torch
import torch.nn as nn

# Redefining the clip_gradients function and testing it
def clip_gradients(model, max_norm, norm_type=2):
    """
    Clips the gradients of a PyTorch model.
    
    Args:
    - model (torch.nn.Module): The model whose gradients are to be clipped.
    - max_norm (float): The maximum allowable value for the gradient norm.
    - norm_type (float): The type of the p-norm. Default is 2 (L2 norm).
    
    Returns:
    - Total norm of the parameters (viewed as a single vector).
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm

# Test
model = nn.Linear(10, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
dummy_input = torch.randn(5, 10)
loss_fn = nn.MSELoss()

# Forward and backward passes
output = model(dummy_input)
loss = loss_fn(output, torch.randn(5, 10))
loss.backward()

# Clip gradients and get the total norm before clipping
total_norm_before_clipping = clip_gradients(model, max_norm=1.0)

total_norm_before_clipping

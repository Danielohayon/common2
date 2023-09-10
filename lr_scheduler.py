import torch
import torch.nn as nn 
from torch.optim.lr_scheduler import LinearLR
import math
import torch.optim as optim
class lr_scheduler():
    def __init__(self, optimizer: torch.optim.Optimizer, update_every = 10):
        super().__init__()
        self.update_every = update_every
        self.optimizer = optimizer
        self.curr_epoch = 0
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()
    
    def update_lr(self):
        if self.curr_epoch % self.update_every == 0 and self.curr_epoch != 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] /= 2
            # self.optimizer.defaults['lr'] = self.optimizer.defaults['lr'] / 2
        self.curr_epoch += 1 
    
class SolLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer:torch.optim.Optimizer, update_every = 10):
        super().__init__(optimizer)
        self.update_every = update_every
    
    def get_lr(self):
        divider = pow(2.0, (self.last_epoch + 1) // 10)
        return [lr/divider for lr in self.base_lrs]

class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if (self.last_epoch + 1) % 10 == 0:  # +1 because last_epoch starts from -1
            self.base_lrs =  [base_lr / 2.0 for base_lr in self.base_lrs]
        return self.base_lrs
    
class StepDecayLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, decay_factor=0.1, step_size=10, last_epoch=-1):
        self.decay_factor = decay_factor
        self.step_size = step_size
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        return [base_lr * (self.decay_factor ** (self.last_epoch // self.step_size)) for base_lr in self.base_lrs]
import math

# My sol
class CosineAnealingScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epoch):
        self.max_epoch = max_epoch
        super().__init__(optimizer)
        # self.optimizer = optimizer
    
    def get_lr(self):
        return [0.5 * (base_lr) * (1 + math.cos(self.last_epoch / self.max_epoch * torch.pi)) for base_lr in self.base_lrs]
        
# Chat sol
import math
class CosineAnnealingLRScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epoch, last_epoch=-1):
        self.max_epoch = max_epoch
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        return [0.5 * (1 + math.cos(math.pi * self.last_epoch / self.max_epoch)) * base_lr for base_lr in self.base_lrs]


res = []
res2 = []
model = nn.Linear(4,2)
model2 =nn.Linear(4,2) 
optimizer = torch.optim.SGD(model.parameters(), lr=0.8)
optimizer2 =torch.optim.SGD(model2.parameters(), lr=0.8) 
# wrapper_optim = lr_scheduler(optimizer)
scheduler = CosineAnealingScheduler(optimizer, max_epoch=100)
scheduler2 = CosineAnnealingLRScheduler(optimizer2, max_epoch=100)

for i in range(100):
    scheduler.step()
    scheduler2.step()
    res.append(optimizer.param_groups[0]['lr'])
    res2.append(optimizer2.param_groups[0]['lr'])


import matplotlib.pyplot as plt
plt.figure()
plt.plot(res)
plt.plot(res2)
plt.show()
print()


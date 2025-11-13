import math

class MyLRScheduler:
    def __init__(
        self,
        optimizer,
        init_step1 = 5000,
        init_step2 = 50000,
        max_step = 100000,
        init_lr = 1e-3,
        min_lr1 = 1e-4,
        min_lr2 = 1e-5,
    ):
        self.optimizer = optimizer
        self.init_step1 = init_step1
        self.init_step2 = init_step2
        self.max_step = max_step
        self.init_lr = init_lr
        self.min_lr1 = min_lr1
        self.min_lr2 = min_lr2
        self.idx = -1
        for idx, param_group in enumerate(optimizer.param_groups):
            if param_group['name'] == 'encoder':
                self.idx = idx

    def step(self, cur_step):
        if cur_step < self.init_step1 or self.idx < 0:
            return
        elif cur_step < self.init_step2:
            step_lr_schedule(self.optimizer, self.init_step2-cur_step, self.init_step2-self.init_step1, self.init_lr, self.min_lr1, self.idx)
        else:
            cosine_lr_schedule(self.optimizer, self.init_step2, cur_step, self.max_step, self.min_lr1, self.min_lr2, self.idx)
        
            
def cosine_lr_schedule(optimizer, init_step, cur_step, max_step, init_lr, min_lr, idx):
    cur_step -= init_step
    max_step -= init_step
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * cur_step / max_step)
    ) + min_lr
    param_group = optimizer.param_groups[idx]
    param_group["lr"] = lr

def step_lr_schedule(optimizer, step, max_step, init_lr, min_lr, idx):
    lr = min_lr + (init_lr - min_lr) * max(1, step) / max_step
    param_group = optimizer.param_groups[idx]    
    param_group["lr"] = lr

def step_lr_schedule2(optimizer, init_step, cur_step, init_lr, min_lr, decay_rate=0.999, idx=0):
    step = cur_step - init_step
    lr = max(min_lr, init_lr * (decay_rate**step))
    param_group = optimizer.param_groups[idx]    
    param_group["lr"] = lr
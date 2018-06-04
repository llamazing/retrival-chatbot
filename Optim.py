import itertools
import torch


class Optimizer(object):
    def __init__(self, optim_type, params,  learning_rate, max_grad_norm=0):
        if optim_type == "Adam":
            optim = torch.optim.Adam(params, learning_rate)
        else:
            raise NotImplementedError
        self.optimizer = optim
        self.scheduler = None
        self.max_grad_norm = max_grad_norm

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def step(self):
        if self.max_grad_norm > 0:
            params = itertools.chain.from_iterable([group['params'] for group in self.optimizer.param_groups])
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

        self.optimizer.step()

    def update(self, loss):
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss)
        else:
            self.scheduler.step()

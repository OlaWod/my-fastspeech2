import torch
import numpy as np


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_cfg, model_cfg):

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_cfg["optimizer"]["betas"],
            eps=train_cfg["optimizer"]["eps"],
            weight_decay=train_cfg["optimizer"]["weight_decay"],
        )
        
        self.n_warmup_steps = train_cfg["optimizer"]["warm_up_step"]
        self.anneal_steps = train_cfg["optimizer"]["anneal_steps"]
        self.anneal_rate = train_cfg["optimizer"]["anneal_rate"]
        self.current_step = train_cfg["optimizer"]["restore_step"]
        self.init_lr = np.power(model_cfg["net_width"], -0.5)

    def update_lr_and_step(self):
        self.current_step += 1
        self.update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def load_state_dict(self, path):
        self.optimizer.load_state_dict(path)

    def get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def update_learning_rate(self):
        """ Learning rate scheduling per step """
        lr = self.init_lr * self.get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


if __name__ == '__main__':
    optimizer = ScheduledOptim(model, train_cfg, model_cfg)

    for epoch in range(n_epochs):
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.update_lr_and_step()

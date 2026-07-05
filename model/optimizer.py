import torch
import numpy as np


class ScheduledOptimMain:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config, model_config, current_step, extra_filters=None):
        # Discriminator params are excluded from the MAIN optimizer (they have
        # their own ScheduledOptimDisc) but stay TRAINABLE -- do NOT freeze them.
        disc_filters = ['D_s', 'D_t', 'postfilter_d']
        # extra_filters are truly frozen: excluded from the optimizer AND
        # requires_grad=False.
        freeze_filters = list(extra_filters) if extra_filters else []

        trained, frozen, disc_excluded = [], [], []
        for name, param in model.named_parameters():
            if any(f in name for f in freeze_filters):
                param.requires_grad = False     # truly frozen
                frozen.append(name)
            elif any(f in name for f in disc_filters):
                disc_excluded.append(name)       # trainable, but not by THIS optimizer
            else:
                trained.append((name, param))

        print(f"[ScheduledOptimMain] training {len(trained)} param tensors; "
              f"{len(disc_excluded)} disc tensors deferred to optimizer_disc")
        if freeze_filters:
            print(f"[ScheduledOptimMain] FROZE {len(frozen)} param tensors "
                  f"(requires_grad=False) with prefixes {freeze_filters}; first 10:")
            for n in frozen[:10]:
                print(f"  - {n}")

        self._optimizer = torch.optim.Adam(
            [p for _, p in trained],
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)
        # meta_learning_warmup = train_config["step"]["meta_learning_warmup"]
        self.current_step = current_step# if current_step <= meta_learning_warmup else current_step - meta_learning_warmup

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, state_dict):
        state_dict['param_groups'] = self._optimizer.state_dict()['param_groups']
        self._optimizer.load_state_dict(state_dict)

    def _get_lr_scale(self):
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

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


class ScheduledOptimDisc:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config):

        self._optimizer = torch.optim.Adam(
            [param for name, param in model.named_parameters()
                        if any([filtered_name in name
                                for filtered_name in ['D_s', 'D_t', 'postfilter_d']])],
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self.init_lr = train_config["optimizer"]["lr_disc"]
        self._init_learning_rate()

    def step_and_update_lr(self):
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, state_dict):
        state_dict['param_groups'] = self._optimizer.state_dict()['param_groups']
        self._optimizer.load_state_dict(state_dict)

    def _init_learning_rate(self):
        lr = self.init_lr
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


import torch.optim as optim


class CAdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CustomAdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            eps = group['eps']
            beta1, beta2 = group['betas']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if weight_decay != 0:
                    grad.add_(weight_decay, p.data)

                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                state['step'] += 1

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr * bias_correction2.sqrt() / bias_correction1
                p.data.add_(-step_size, grad)

                p.data.mul_(1 - lr * weight_decay)

        return loss

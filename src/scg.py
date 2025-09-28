import torch
from torch.optim.optimizer import Optimizer

class SCG(Optimizer):
    """
    Scaled Conjugate Gradient (SCG) Optimizer for PyTorch
    Implements Møller's SCG algorithm (1993). test1
    """

    def __init__(self, params, sigma=1e-4, lambd=1e-6):
        if sigma <= 0.0:
            raise ValueError("Invalid sigma value: {}".format(sigma))
        if lambd < 0.0:
            raise ValueError("Invalid lambda value: {}".format(lambd))

        defaults = dict(sigma=sigma, lambd=lambd)
        super(SCG, self).__init__(params, defaults)

        # Internal buffers
        self.state['success'] = True
        self.state['k'] = 0

    #@torch.no_grad()
    def step(self, closure):
        """
        Performs a single SCG step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        loss = closure()
        self.zero_grad()
        loss.backward()

        # Flatten gradients and parameters for SCG math
        params = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)
                grads.append(p.grad.view(-1))
        w = torch.cat([p.data.view(-1) for p in params])
        g = torch.cat(grads)

        # State init
        state = self.state
        sigma = self.param_groups[0]['sigma']
        lambd = self.param_groups[0]['lambd']

        if 'w_prev' not in state:
            state['w_prev'] = w.clone()
            state['r'] = -g.clone()
            state['p'] = state['r'].clone()
            state['lambd'] = lambd
            state['success'] = True
            state['delta'] = None

        r = state['r']
        p = state['p']
        lambd = state['lambd']
        success = state['success']

        # Step 2: compute second order info
        if success:
            sigma_k = sigma / p.norm()
            w_sigma = w + sigma_k * p
            # get gradient at w+sigma*p
            self._set_params(params, w_sigma)
            loss_sigma = closure()
            self.zero_grad()
            loss_sigma.backward()
            g_sigma = torch.cat([p.grad.view(-1) for group in self.param_groups
                                 for p in group['params'] if p.grad is not None])
            s_k = (g_sigma - g) / sigma_k
            delta = p.dot(s_k)
            delta += (lambd) * p.dot(p)
            state['delta'] = delta
        else:
            delta = state['delta']

        if delta <= 0:
            lambd = lambd + 2 * (lambd - delta / p.dot(p))
            delta = delta + lambd * p.dot(p)
            state['delta'] = delta

        # Step size
        alpha = r.dot(r) / delta

        # Compute comparison parameter Δ
        w_new = w + alpha * p
        self._set_params(params, w_new)
        loss_new = closure()
        self.zero_grad()
        loss_new.backward()        
        A = 2 * delta * (loss - loss_new) / (alpha ** 2)

        if A >= 0:  # success
            r_new = -torch.cat([p.grad.view(-1) for group in self.param_groups
                                for p in group['params'] if p.grad is not None])
            lambd = 0
            success = True
            if (state['k'] + 1) % len(w) == 0:  # restart
                p_new = r_new.clone()
            else:
                beta = (r_new.norm()**2 - r_new.dot(r)) / r.norm()**2
                p_new = r_new + beta * p
            state['r'] = r_new
            state['p'] = p_new
            w = w_new
            self._set_params(params, w)
            if A > 0.75:
                lambd = 0.25 * lambd
        else:  # failure
            self._set_params(params, w)  # revert
            success = False
            state['r'] = r
            state['p'] = p

        if A < 0.25:
            lambd = lambd + delta * (1 - A) / p.dot(p)

        state['lambd'] = lambd
        state['success'] = success
        state['k'] += 1

        self.zero_grad()
        return loss_new

    def _set_params(self, params, flat):
        """Assign a flattened parameter vector back to model params."""
        offset = 0
        for p in params:
            numel = p.numel()
            with torch.no_grad():
                p.data.copy_(flat[offset:offset + numel].view_as(p))
            offset += numel

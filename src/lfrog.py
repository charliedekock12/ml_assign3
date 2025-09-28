import math
import torch
from torch.optim import Optimizer

class LeapfrogOptimizer(Optimizer):
    """
    Minimal leapfrog (Snyman) optimizer for full-batch training.

    Args:
        params: iterable of parameters to optimize
        time_step (float): Δt
        delta_max (float): δ (cap on step norm)
        tol_grad (float): ε stopping tolerance on gradient norm (for user checks)
        j_threshold (int): j number of consecutive reduced-velocity restarts before zeroing
    """
    def __init__(self, params,
                 time_step=0.5,
                 delta_max=1.0,
                 tol_grad=1e-5,
                 j_threshold=2):
        if time_step <= 0:
            raise ValueError("time_step must be > 0")
        defaults = dict(time_step=time_step,
                        delta_max=delta_max,
                        tol_grad=tol_grad,
                        j_threshold=j_threshold)
        super().__init__(params, defaults)

        # initialize per-parameter velocity and restart counter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['velocity'] = torch.zeros_like(p.data)
                state['i'] = 0

    def grad_norm(self):
        """Compute global gradient norm (useful for convergence check)."""
        total = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                total += p.grad.data.pow(2).sum().item()
        return math.sqrt(total)

    def step(self, closure):
        """
        Perform one leapfrog optimizer step.
        closure: callable that zeroes grads, computes loss, backward, and returns loss.
        Returns loss (torch.Tensor) as standard PyTorch pattern.
        """
        if closure is None:
            raise ValueError("closure must compute full-batch loss and call backward()")

        # 1. Evaluate loss and gradients at x_k
        loss = closure()

        for group in self.param_groups:
            dt = group['time_step']
            delta_max = group['delta_max']
            j_thresh = group['j_threshold']

            # compute step size norm across group
            total_step_norm_sq = 0.0
            steps = {}
            for p in group['params']:
                v = self.state[p]['velocity']
                step = v * dt
                steps[p] = step
                total_step_norm_sq += step.pow(2).sum().item()
            total_step_norm = math.sqrt(total_step_norm_sq)

            # cap step size if needed
            if total_step_norm > delta_max and total_step_norm > 0.0:
                scale = delta_max / (total_step_norm + 1e-16)
                for p in steps:
                    steps[p] = steps[p] * scale

            # update positions
            for p, step in steps.items():
                p.data.add_(step)

        # 2. Recompute gradients at x_{k+1}
        loss = closure()

        # 3. Velocity update + restart logic
        for group in self.param_groups:
            dt = group['time_step']
            j_thresh = group['j_threshold']
            for p in group['params']:
                if p.grad is None:
                    a_new = torch.zeros_like(p.data)
                else:
                    a_new = -p.grad.data

                v_old = self.state[p]['velocity']
                v_new = v_old + a_new * dt

                if v_new.norm() > v_old.norm():
                    self.state[p]['i'] = 0
                else:
                    self.state[p]['i'] += 1
                    if self.state[p]['i'] < j_thresh:
                        v_new = 0.25 * (v_new + v_old)
                    else:
                        v_new = torch.zeros_like(v_old)
                        self.state[p]['i'] = 1

                self.state[p]['velocity'] = v_new

        return loss

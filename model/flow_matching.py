import torch
from torch import nn
from torchdyn.core import NeuralODE

class ConditionedVelocityModel(nn.Module):
    def __init__(self, velocity_model, h,):
        super(ConditionedVelocityModel, self).__init__()
        self.velocity_model = velocity_model
        self.h = h

    def forward(self, t, x, h=None, *args, **kwargs):
        if not h:
            h = self.h
        return self.velocity_model(x, t=t, z=h)

class FlowMatching(nn.Module):
    def __init__(self, velocity_model, sigma=0.,):
        super().__init__()
        self.velocity_model = velocity_model
        self.sigma = sigma

    def get_mu_t(self, x0, x1, t):
        return t * x1 + (1 - t) * x0

    def get_gamma_t(self, t):
        return torch.sqrt(2 * t * (1 - t))

    def sample_xt(self, x0, x1, t, epsilon):
        t = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        mu_t = self.get_mu_t(x0, x1, t)
        if self.sigma != 0.0:
            sigma_t = self.get_gamma_t(t)
            return mu_t + sigma_t * epsilon
        return mu_t

    def compute_loss(
        self,
        x0,
        x1,
        h=None,
        t=None,
    ):
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device).type_as(x0)

        if self.sigma != 0.0:
            eps = torch.randn_like(x0)
        else:
            eps = None
        xt = self.sample_xt(x0, x1, t, eps)
        ut = x1 - x0
        # embed class to add to time embeddings
        vt = self.velocity_model(xt, t=t, z=h)
        return torch.mean((vt - ut) ** 2)

    def predict(
        self,
        x0,
        h=None,
        n_sampling_steps=10,
    ):
        conditional_velocity_model = ConditionedVelocityModel(
            velocity_model=self.velocity_model,
            h=h,
        )
        node = NeuralODE(
            conditional_velocity_model,
            solver="dopri5",
            sensitivity="adjoint",
        )
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, n_sampling_steps),
            )
        return traj[-1]
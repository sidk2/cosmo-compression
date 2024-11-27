'''Implementation of flow matching decoder'''
import torch
from torch import nn
from torchdyn import core as tdyn


class ConditionedVelocityModel(nn.Module):
    def __init__(self, velocity_model, h,):
        super(ConditionedVelocityModel, self).__init__()
        self.velocity_model = velocity_model
        self.register_buffer('h', h)

    def forward(self, t, x, *args, **kwargs):
        return self.velocity_model(x, timestep=t).sample

class FlowMatching(nn.Module):
    '''Implementation of the flow matching loss from Lipman et. al 2023'''
    def __init__(self, velocity_model, sigma=0.,):
        super(FlowMatching, self).__init__()
        self.velocity_model = velocity_model
        self.sigma = sigma

    def get_mu_t(self, x0, x1, t):
        '''Get mean for optimal transport'''
        return t * x1 + (1 - t) * x0

    def get_gamma_t(self, t):
        '''Get variance for OT'''
        return torch.sqrt(2 * t * (1 - t))

    def sample_xt(self, x0, x1, t, epsilon, h=None):
        '''Sample forward pass'''
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
        xt = self.sample_xt(x0, x1, t, eps, h)
        ut = x1 - x0
        # embed class to add to time embeddings
        vt = self.velocity_model(xt, timestep=t).sample
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
        node = tdyn.NeuralODE(
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
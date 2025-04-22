"""Implements conditional flow matching from Lipman et. al 23"""

import torch
from torch import nn
from torchdyn.core import NeuralODE


class ConditionedVelocityModel(nn.Module):
    """Neural net for velocity field prediction"""

    def __init__(
        self,
        velocity_model: torch.nn.Module,
        h: torch.Tensor | None,
        reverse: bool = False,
    ):
        super(ConditionedVelocityModel, self).__init__()
        self.reverse = reverse
        self.velocity_model = velocity_model
        self.h = h

    def forward(
        self,
        t: torch.Tensor | int,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        if not h:
            h = self.h
        return (
            -1 * self.velocity_model(x, t=t, z=h)
            if self.reverse
            else self.velocity_model(x, t=t, z=h)
        )


class FlowMatching(nn.Module):
    """Implements the flow matching loss"""

    def __init__(
        self,
        velocity_model: torch.nn.Module,
        sigma: torch.Tensor | int = 0.0,
        reverse: bool = False,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.sigma = sigma
        self.reverse = reverse

    def get_mu_t(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Sample a mean for the Gaussian conditional probability path from the noise distribution to the training sample at time t.

        Args:
            - x0: The noise sample
            - x1: The training sample
            - t: The time step.
        Returns:
            - mu_t: The mean of a Gaussian at t, along the path. At t=0, will be pure noise, and at t=1, will be the the sample mean
        """
        return t * x1 + (1 - t) * x0

    def get_gamma_t(self, t: torch.Tensor):
        """Get standard deviation of Gaussian for probability path"""
        return torch.sqrt(2 * t * (1 - t))

    def sample_xt(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, epsilon: int
    ) -> torch.Tensor:
        """Sample the predicted sample at x_t."""
        t = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        mu_t = self.get_mu_t(x0, x1, t)
        if self.sigma != 0.0:
            sigma_t = self.get_gamma_t(t)
            return mu_t + sigma_t * epsilon
        return mu_t

    def compute_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        h: torch.Tensor | None = None,
        t: torch.Tensor | None | int = None,
    ) -> torch.Tensor:
        """Given a noise field and a training sample, compute the flow matching loss."""
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
        x0: torch.Tensor,
        h: torch.Tensor | None = None,
        n_sampling_steps: int = 100,
        solver="dopri5",
    ) -> torch.Tensor:
        """Runs inference for flow matching model.

        Args:
            - x0: The noise field if model is not reversed, else the training sample
            - h: The vector to be conditioned on
            - n_sampling_steps: The number of steps to be used when solving the flow matching ODE
            - solver: The differential equation solver. Options are "dopri5", "rk4", and "euler". Default is dopri5, which is the slowest but most accurate.
        Returns:
            - x1: The predicted training sample

        """
        conditional_velocity_model = ConditionedVelocityModel(
            velocity_model=self.velocity_model, h=h, reverse=self.reverse
        )
        node = NeuralODE(
            conditional_velocity_model,
            solver="euler",
            sensitivity="adjoint",
        )
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, n_sampling_steps),
            )
        return traj[-1]

import torch
from torch import nn
import torch.autograd as autograd
from torchdyn.core import NeuralODE
from torchdiffeq import odeint


class ConditionedVelocityModel(nn.Module):
    """Neural net for velocity field prediction, with optional reverse flag"""

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
        if h is None:
            h = self.h
        velocity = self.velocity_model(x, t=t, z=h)
        return -velocity if self.reverse else velocity


class ODEWithLogProb(nn.Module):
    """Wraps a flow‐matching vector field so that we can track divergence and log‐density."""

    def __init__(
        self,
        vector_field: ConditionedVelocityModel,
        divergence_method: str = 'approximate'
    ):
        super().__init__()
        self.vector_field = vector_field
        self.divergence_method = divergence_method

    def divergence(self, f: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.divergence_method == 'exact':
            # Compute ∇·f exactly (slow for high dimensions)
            batch_size, dim = x.shape
            trace = torch.zeros(batch_size, device=x.device)
            for i in range(dim):
                grad_i = autograd.grad(f[:, i].sum(), x, create_graph=True)[0][:, i]
                trace += grad_i
            return trace
        else:
            # Hutchinson’s estimator: E[ εᵀ ∂f/∂x ε ] = ∇·f
            eps = torch.randn_like(x)
            f_eps = (f * eps).sum()
            grad = autograd.grad(f_eps, x, create_graph=True)[0]
            return (grad * eps).sum(dim=1)

    def forward(self, t: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x, logp = states
        x = x.requires_grad_()
        v_t = self.vector_field(t, x)                  # velocity field at time t
        div = self.divergence(v_t, x, t)               # ∇·v_t
        dlogp = -div                                   # d/dt log p = –∇·v
        return v_t, dlogp


class FlowMatching(nn.Module):
    """Flow‐matching module with training loss and inference‐time log‐likelihood."""

    def __init__(
        self,
        velocity_model: torch.nn.Module,
        sigma: float = 0.0,
        reverse: bool = False,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.sigma = sigma
        self.reverse = reverse

    def get_mu_t(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return t * x1 + (1 - t) * x0

    def get_gamma_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(2 * t * (1 - t))

    def sample_xt(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, eps: torch.Tensor | None
    ) -> torch.Tensor:
        t_broadcast = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        mu_t = self.get_mu_t(x0, x1, t_broadcast)
        if self.sigma != 0.0:
            sigma_t = self.get_gamma_t(t_broadcast)
            return mu_t + sigma_t * eps
        return mu_t

    def compute_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        h: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device).type_as(x0)

        eps = torch.randn_like(x0) if self.sigma != 0.0 else None
        xt = self.sample_xt(x0, x1, t, eps)
        ut = x1 - x0
        vt = self.velocity_model(xt, t=t, z=h)
        return torch.mean((vt - ut) ** 2)

    def predict(
        self,
        x0: torch.Tensor,
        h: torch.Tensor | None = None,
        n_sampling_steps: int = 100,
        solver: str = "dopri5",
        full_return: bool = False,
    ) -> torch.Tensor:
        conditional_velocity_model = ConditionedVelocityModel(
            velocity_model=self.velocity_model, h=h, reverse=self.reverse
        )
        node = NeuralODE(
            conditional_velocity_model,
            solver=solver,
            sensitivity="adjoint",
        )
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, n_sampling_steps, device=x0.device),
            )
        return traj[-1] if not full_return else traj

    def estimate_log_likelihood(
        self,
        x_T: torch.Tensor,
        h: torch.Tensor | None = None,
        t0: float = 0.0,
        t1: float = 1.0,
        divergence_method: str = 'approximate',
    ) -> torch.Tensor:
        """
        Estimate log-likelihood of target batch x_T by integrating the flow ODE backward
        from T=1 to T=0 and tracking log-density change along the path.

        Returns a 1D tensor of shape (batch_size,) containing log p_T(x_T) for each sample.
        """
        x_T = x_T.clone().detach().requires_grad_()
        # Wrap the trained velocity_model into a reverse‐flagged conditioned model
        cond_vel_model = ConditionedVelocityModel(
            velocity_model=self.velocity_model, h=h, reverse=True
        )
        ode_func = ODEWithLogProb(cond_vel_model, divergence_method)

        # Initialize log p at t1 to zero (we accumulate d(log p)/dt backward)
        logp_T = torch.zeros(x_T.size(0), device=x_T.device)

        # Pack initial states: (x(t1)=x_T, logp(t1)=0)
        states = (x_T, logp_T)
        t_span = torch.tensor([t1, t0], device=x_T.device)

        
        # Integrate backward: from t1 → t0
        z_traj, logp_traj = odeint(
            ode_func,
            states,
            t_span,
            rtol=1e-5,
            atol=1e-5,
        )

        # After integration, z_traj[-1] is z(0), logp_traj[-1] is ∫ₜ₁→ₜ₀ (–div) dt
        z0, logp0 = z_traj[-1], logp_traj[-1]

        # Assume base distribution p₀(z) = N(0, I); compute its log‐prob at z0
        base_dist = torch.distributions.Normal(0, 1)
        logp_z0 = base_dist.log_prob(z0).sum(dim=1)

        # Total log‐likelihood: log p₀(z0) + (accumulated change)
        return logp_z0 + logp0

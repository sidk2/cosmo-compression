import torch
from torch import nn
import torch.autograd as autograd
from torchdyn.core import NeuralODE
from torchdiffeq import odeint
from torch.nn import functional as F


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
    """Flow-matching module using blurring instead of noise for forward process."""
    
    def __init__(
        self,
        velocity_model: torch.nn.Module,
        max_sigma: float = 255.0,  # Maximum blur sigma
        min_sigma: float = 0.1,  # Minimum blur sigma
        reverse: bool = False,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.reverse = reverse
        
    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get blur sigma as a function of time t ∈ [0, 1]."""
        # Linear interpolation from min_sigma to max_sigma
        return self.min_sigma + t * (self.max_sigma - self.min_sigma)
    
    def gaussian_blur(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # 1) Compute adaptive kernel sizes (odd) for each σ
        ks = (2 * (4 * sigma).ceil().int() + 1)  # e.g. 8σ rule

        # 2) Build one “giant” block‐diag kernel of shape (B*C, 1, Kmax, Kmax)
        Kmax = ks.max().item()
        coords = torch.arange(Kmax, device=device) - (Kmax // 2)
        y, xg = torch.meshgrid(coords, coords, indexing='ij')
        base = y**2 + xg**2

        kernels = []
        for σ, k in zip(sigma, ks):
            σ = σ.item()
            k = k.item()
            # slice out the central k×k of the Kmax×Kmax grid:
            start = (Kmax - k) // 2
            sub = base[start:start+k, start:start+k]
            kern = torch.exp(-sub / (2*σ*σ))
            kern = F.pad(kern, [start]*4)      # pad back to Kmax
            kern = kern / kern.sum()
            kernels.append(kern)
        # stack and repeat for channels
        kernels = torch.stack(kernels)               # (B, Kmax, Kmax)
        kernels = kernels.view(B, 1, Kmax, Kmax)
        kernels = kernels.repeat_interleave(C, 0)    # (B*C,1,Kmax,Kmax)

        # 3) reshape input to (B*C,1,H,W) and apply one conv
        x_ = x.reshape(1, B*C, H, W)
        out = F.conv2d(x_, kernels, padding=Kmax//2, groups=B*C)
        return out.view(B, C, H, W)
    
    def get_mu_t(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Get mean of interpolation between x0 and x1."""
        return t * x1 + (1 - t) * x0
    
    def sample_xt(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """Sample x_t by applying progressive blurring."""
        # For cold diffusion, we blur x1 (the target) towards x0 (fully blurred)
        # At t=0: we want x1 (sharp)
        # At t=1: we want x0 (blurred)
        
        # Get blur sigma for this timestep
        t_broadcast = t.view(t.shape[0], *([1] * (x1.dim() - 1)))
        sigma_t = self.get_sigma_t(t_broadcast.squeeze())
        
        # Apply blur to x1
        x1_blurred = self.gaussian_blur(x1, sigma_t)
        
        # Interpolate between sharp x1 and blurred version
        # At t=0: return x1 (sharp)
        # At t=1: return heavily blurred version
        return self.get_mu_t(x1, x1_blurred, t_broadcast)
    
    def compute_loss(
        self,
        x0: torch.Tensor,  # This will be the "fully blurred" version
        x1: torch.Tensor,  # This is the sharp target
        h: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute flow matching loss with blurring."""
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device).type_as(x0)
        
        # Sample x_t using blurring
        xt = self.sample_xt(x0, x1, t)
        
        # The target velocity should point from blurred to sharp
        ut = x1 - xt  # This is the "deblurring direction"
        
        # Predict velocity
        vt = self.velocity_model(xt, t=t, z=h)
        
        # Compute loss
        loss = torch.mean((vt - ut) ** 2)
        
        # Optional: Add frequency-weighted loss
        if hasattr(self, 'freq_weight') and self.freq_weight > 0:
            # Compute high-frequency loss
            grad_pred = torch.gradient(vt.view(vt.shape[0], -1), dim=1)[0]
            grad_target = torch.gradient(ut.view(ut.shape[0], -1), dim=1)[0]
            freq_loss = torch.mean((grad_pred - grad_target) ** 2)
            loss = loss + self.freq_weight * freq_loss
        
        return loss
    
    def predict(
        self,
        x0: torch.Tensor,  # Starting point (blurred)
        h: torch.Tensor | None = None,
        n_sampling_steps: int = 100,
        solver: str = "dopri5",
        full_return: bool = False,
    ) -> torch.Tensor:
        """Predict by running the ODE from blurred to sharp."""
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

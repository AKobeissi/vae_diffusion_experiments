import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional

class CFGDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device,
                 null_label: int = -1, p_uncond: float = 0.1):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        self.device = device

        self.lambda_min = -20
        self.lambda_max = 20
        self.null_label = null_label
        self.p_uncond = p_uncond



    ### UTILS
    def get_exp_ratio(self, l: torch.Tensor, l_prim: torch.Tensor):
        return torch.exp(l-l_prim)
    
    def get_lambda(self, t: torch.Tensor): 
        # TODO: Write function that returns lambda_t for a specific time t. Do not forget that in the paper, lambda is built using u in [0,1]
        # Note: lambda_t must be of shape (batch_size, 1, 1, 1)
        
        # Normalize t to [0, 1] range
        # In the paper, they sample λ via λ = −2 log tan(au + b) for uniformly distributed u ∈ [0, 1]
        u = t.float() / self.n_steps  # Convert to [0, 1]
        
        # Convert scalar values to tensors with the same device as t
        lambda_min_t = torch.tensor(self.lambda_min, device=t.device)
        lambda_max_t = torch.tensor(self.lambda_max, device=t.device)
        
        # Constants a and b as defined in the paper
        b = torch.arctan(torch.exp(-lambda_max_t / 2))
        a = torch.arctan(torch.exp(-lambda_min_t / 2)) - b
        
        # Calculate lambda_t following the paper's formula: λ = −2 log tan(au + b)
        lambda_t = -2 * torch.log(torch.tan(a * u + b))
        
        # Ensure lambda_t is within bounds
        lambda_t = torch.clamp(lambda_t, min=self.lambda_min, max=self.lambda_max)
        
        # Reshape to (batch_size, 1, 1, 1) as required
        lambda_t = lambda_t.view(-1, 1, 1, 1)
        
        return lambda_t

    
    def alpha_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Alpha(lambda_t) for a specific time t according to (1)
        # From equation (1): α²λ = 1/(1 + e^(-λ))
        var = 1 / (1 + torch.exp(-lambda_t))
        return var.sqrt()

    
    def sigma_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Sigma(lambda_t) for a specific time t according to (1)
        # From equation (1): σ²λ = 1 - α²λ
        alpha_squared = 1 / (1 + torch.exp(-lambda_t))
        var = 1 - alpha_squared
        return var.sqrt()
    
    
    ## Forward sampling
    def q_sample(self, x: torch.Tensor, lambda_t: torch.Tensor, noise: torch.Tensor):
        #TODO: Write function that returns z_lambda of the forward process, for a specific: x, lambda l and N(0,1) noise  according to (1)
        # From equation (1): q(zλ|x) = N(αλx, σ²λI)
        alpha = self.alpha_lambda(lambda_t)
        sigma = self.sigma_lambda(lambda_t)
        z_lambda_t = alpha * x + sigma * noise
        
        return z_lambda_t               
    
    def sigma_q(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l) according to (2)
         # From equation (2): σ²λ|λ' = (1 - e^(λ-λ'))σ²λ
        # First, get σ²λ which is 1 - α²λ
        sigma_squared = 1 - 1/(1 + torch.exp(-lambda_t))
        
        # Then compute the variance according to the equation
        var_q = (1 - torch.exp(lambda_t - lambda_t_prim)) * sigma_squared
        
        return var_q.sqrt()
    
    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l, x) according to (3)
    # From equation (3): σ̃²λ'|λ = (1 - e^(λ-λ'))σ²λ'
        # First, get σ²λ' which is 1 - α²λ'
        alpha_prim_squared = 1 / (1 + torch.exp(-lambda_t_prim))
        sigma_prim_squared = 1 - alpha_prim_squared
        
        # Then compute the variance according to the equation
        var_q_x = (1 - torch.exp(lambda_t - lambda_t_prim)) * sigma_prim_squared
        
        return var_q_x.sqrt()
    
    ### REVERSE SAMPLING
    def mu_p_theta(self, z_lambda_t: torch.Tensor, x: torch.Tensor, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns mean of the forward process transition distribution according to (4)
        # From equation (4) and (3): μ̃λ'|λ(zλ, x) = e^(λ-λ')(αλ'/αλ)zλ + (1-e^(λ-λ'))αλ'x
        alpha = self.alpha_lambda(lambda_t)
        alpha_prim = self.alpha_lambda(lambda_t_prim)
        
        # Compute the two terms in the mean formula
        exponential_factor = torch.exp(lambda_t - lambda_t_prim)
        
        # First term: e^(λ-λ')(αλ'/αλ)zλ
        term1 = exponential_factor * (alpha_prim / alpha) * z_lambda_t
        
        # Second term: (1-e^(λ-λ'))αλ'x
        term2 = (1 - exponential_factor) * alpha_prim * x
        
        # Combine the terms to get the mean
        mu = term1 + term2
        
        return mu


    def var_p_theta(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, v: float=0.3):
        #TODO: Write function that returns var of the forward process transition distribution according to (4)
        # From equation (4): (σ̃²λ'|λ)^(1-v)(σ²λ|λ')^v
        
        # First, calculate σ̃²λ'|λ using sigma_q_x
        sigma_q_x_squared = (1 - torch.exp(lambda_t - lambda_t_prim)) * (1 - 1/(1 + torch.exp(-lambda_t_prim)))
        
        # Then, calculate σ²λ|λ' using sigma_q
        sigma_q_squared = (1 - torch.exp(lambda_t - lambda_t_prim)) * (1 - 1/(1 + torch.exp(-lambda_t)))
        
        # Finally, interpolate between the two variance estimates
        var = sigma_q_x_squared ** (1-v) * sigma_q_squared ** v
        
        return var
    
    def p_sample(self, z_lambda_t: torch.Tensor, lambda_t : torch.Tensor, lambda_t_prim: torch.Tensor,  x_t: torch.Tensor, set_seed=False):
        # TODO: Write a function that sample z_{lambda_t_prim} from p_theta(•|z_lambda_t) according to (4) 
        # Note that x_t correspond to x_theta(z_lambda_t)
        if set_seed:
            torch.manual_seed(42)
    
        # Calculate the mean of the transition distribution
        alpha_t = self.alpha_lambda(lambda_t)
        alpha_t_prim = self.alpha_lambda(lambda_t_prim)
        
        exp_ratio = torch.exp(lambda_t - lambda_t_prim)
        
        # Calculate the mean using equation (4)
        mu = exp_ratio * (alpha_t_prim / alpha_t) * z_lambda_t + (1 - exp_ratio) * alpha_t_prim * x_t
        
        # Calculate the variance
        var = self.var_p_theta(lambda_t, lambda_t_prim)
        
        # Generate noise for sampling
        epsilon = torch.randn_like(z_lambda_t)
        
        # Sample from the transition distribution: z_λ' ~ N(μ, var)
        sample = mu + torch.sqrt(var) * epsilon
        
        return sample
    
    def loss(
        self,
        x0: torch.Tensor,
        labels: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        set_seed: bool = False,
    ) -> torch.Tensor:
        if set_seed:
            torch.manual_seed(42)

        batch_size = x0.shape[0]
        # dims to average over per‑sample (exclude batch dim)
        dim = list(range(1, x0.ndim))

        print(f"Batch size: {batch_size}")
        print(f"Input x0 shape: {x0.shape}")
        print(f"Labels shape: {labels.shape}")

        # 1) pick random timesteps t ∼ Uniform({0,…,n_steps−1})
        t = torch.randint(
            0, self.n_steps, (batch_size,),
            device=x0.device, dtype=torch.long
        )
        print(f"Random timesteps t: {t.shape}")
        print(t)

        # 2) sample noise ε ∼ N(0, I)
        if noise is None:
            noise = torch.randn_like(x0)
        print(f"Noise shape: {noise.shape}")

        # 3) compute λ_t and corrupt x0 → z_λ = α_λ x0 + σ_λ ε
        lambda_t = self.get_lambda(t)      # (B,1,1,1)
        print(f"Lambda_t shape: {lambda_t.shape}")
        print(lambda_t)

        z_lambda_t = self.q_sample(x0, lambda_t, noise)  # (B, C, H, W)
        print(f"z_lambda_t shape: {z_lambda_t.shape}")
        print(z_lambda_t)

        # 4) classifier‑free dropout: with prob p_uncond, swap c→null_label
        drop_mask = torch.rand(batch_size, device=x0.device) < self.p_uncond
        print(f"Drop mask: {drop_mask.shape}")
        print(drop_mask)

        labels_cf = labels.clone()
        labels_cf[drop_mask] = self.null_label
        print(f"Labels_cf shape: {labels_cf.shape}")
        print(labels_cf)

        # 5) **only** pass (z, labels) into the network
        eps_pred = self.eps_model(z_lambda_t, labels_cf)
        print(f"eps_pred shape: {eps_pred.shape}")
        print(eps_pred)

        # 6) per‑sample MSE over all dims except batch, then mean over batch
        per_sample_loss = ((eps_pred - noise) ** 2).sum(dim=dim)  # (B,)
        print(f"Per-sample loss shape: {per_sample_loss.shape}")
        print(per_sample_loss)

        loss = per_sample_loss.mean()                   # scalar
        print(f"Final loss: {loss}")

        return loss

    # def loss(
    #     self,
    #     x0: torch.Tensor,
    #     labels: torch.Tensor,
    #     noise: Optional[torch.Tensor] = None,
    #     set_seed: bool = False,
    # ) -> torch.Tensor:
    #     if set_seed:
    #         torch.manual_seed(42)

    #     batch_size = x0.shape[0]
    #     # dims to average over per‐sample (exclude batch dim)
    #     dim = list(range(1, x0.ndim))

    #     # 1) sample random λ_t
    #     t = torch.randint(0, self.n_steps, (batch_size,),
    #                       device=x0.device, dtype=torch.long)

    #     # 2) sample noise ε ∼ N(0, I)
    #     if noise is None:
    #         noise = torch.randn_like(x0)

    #     # 3) compute forward noised sample z_λ = α_λ x0 + σ_λ ε
    #     lambda_t = self.get_lambda(t)
    #     z_lambda_t = self.q_sample(x0, lambda_t, noise)
    #     print(f"z_lambda_t: {z_lambda_t.shape}")
    #     print(z_lambda_t)

    #     print(f"lambda_t: {lambda_t.shape}")
    #     print(lambda_t)
    #     # 4) classifier‐free dropout: with prob p_uncond, replace label by null_label
    #     drop_mask = torch.rand(batch_size, device=x0.device) < self.p_uncond
    #     labels_cf = labels.clone()
    #     labels_cf[drop_mask] = self.null_label
    #     print(f"labels_cf: {labels_cf.shape}")
    #     print(labels_cf)

    #     # 5) predict ε̂ = ε_model(z_λ, λ_t, c)
    #     eps_pred = self.eps_model(z_lambda_t, lambda_t, labels_cf)

    #     # 6) per‐sample MSE over dims 1..N, then mean over batch
    #     per_sample_loss = ((eps_pred - noise) ** 2).mean(dim=dim)
    #     loss = per_sample_loss.mean()

    #     return loss


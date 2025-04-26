import torch 
from torch import nn 
from typing import Optional, Tuple
import torch.nn.functional as F  


class DenoiseDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta


    ### UTILS
    def gather(self, c: torch.Tensor, t: torch.Tensor):
        c_ = c.gather(-1, t)
        return c_.reshape(-1, 1, 1, 1)

    ### FORWARD SAMPLING
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return mean and variance of q(x_t|x_0)
        # mean = sqrt(alpha_bar) * x0
        # var = (1 - alpha_bar)
        
        alpha_bar = self.gather(self.alpha_bar, t)
        mean = torch.sqrt(alpha_bar) * x0
        var = 1.0 - alpha_bar
        
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        
        # Return x_t sampled from q(x_t|x_0)
        # x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * epsilon
        
        alpha_bar = self.gather(self.alpha_bar, t)
        mean = torch.sqrt(alpha_bar) * x0
        std = torch.sqrt(1.0 - alpha_bar)
        
        sample = mean + std * eps
        
        return sample

    ### REVERSE SAMPLING
    def p_xt_prev_xt(self, xt: torch.Tensor, t: torch.Tensor):
        # Return mean and variance of p_theta(x_{t-1} | x_t)
        # mu_theta = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_theta)
        # var = beta_t
        
        # Predict noise using the model (make sure t is properly formatted)
        eps_theta = self.eps_model(xt, t)
        
        # Gather the right values from alpha, beta, and alpha_bar using the time step t
        alpha = self.gather(self.alpha, t)
        beta = self.gather(self.beta, t)
        alpha_bar = self.gather(self.alpha_bar, t)
        
        # Calculate the coefficient for eps_theta
        eps_coef = beta / torch.sqrt(1.0 - alpha_bar)
        
        # Calculate mu_theta according to the formula
        mu_theta = (1.0 / torch.sqrt(alpha)) * (xt - eps_coef * eps_theta)
        
        # The variance is simply beta
        var = beta
        
        return mu_theta, var

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        
        # Sample x_{t-1} from p_theta(x_{t-1}|x_t)
        # Get mean and variance
        mu_theta, var = self.p_xt_prev_xt(xt, t)
        
        # No noise if t = 0 (final step)
        if t[0] == 0:
            return mu_theta
        
        # Add noise with the variance
        sigma = torch.sqrt(var)
        
        # Generate noise
        noise = torch.randn_like(xt)
        
        # Sample from the distribution
        sample = mu_theta + sigma * noise
        
        return sample

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        """Compute the DDPM loss according to the formula."""
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Calculate noisy sample x_t 
        alpha_bar = self.gather(self.alpha_bar, t)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        
        # Generate x_t from x_0 and noise according to the formula
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        
        # Predict noise
        predicted_noise = self.eps_model(x_t, t)
        
        # Alternative loss calculation - use L2 norm squared directly
        squared_error = torch.sum((noise - predicted_noise) ** 2, dim=list(range(1, noise.ndim)))
        loss = torch.mean(squared_error)
        
        return loss
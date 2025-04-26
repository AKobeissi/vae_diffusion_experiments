"""
Solutions for Question 1 of hwk3.
@author: Shawn Tan and Jae Hyun Lim
"""
import math
import numpy as np
import torch

torch.manual_seed(42)

def log_likelihood_bernoulli(mu, target):
    """ 
    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)

    # compute log_likelihood_bernoulli
    # For Bernoulli distribution: log p(x) = x log(mu) + (1-x) log(1-mu)
    log_prob = target * torch.log(mu + 1e-8) + (1 - target) * torch.log(1 - mu + 1e-8)
    ll_bernoulli = log_prob.sum(dim=1)  # Sum over all dimensions except batch

    return ll_bernoulli


def log_likelihood_normal(mu, logvar, z):
    """ 
    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)
    
    # Dimension of input
    input_size = mu.size(1)
    
    # compute log normal
    # For multivariate normal with diagonal covariance: 
    # log p(z) = -0.5 * (log(2π) + logvar + (z-μ)²/σ²)
    log_prob = -0.5 * (torch.log(2 * torch.tensor(math.pi)) + 
                       logvar + 
                       ((z - mu) ** 2) / torch.exp(logvar))
    
    ll_normal = log_prob.sum(dim=1)  # Sum over all dimensions except batch
    
    return ll_normal


def log_mean_exp(y):
    """ 
    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1)

    # compute log_mean_exp
    # First find the maximum value for each data point (batch)
    a = torch.max(y, dim=1, keepdim=True)[0]
    
    # log(mean(exp(y - a) * exp(a))) = log(mean(exp(y - a))) + a
    # where mean is across the sample dimension
    lme = torch.log(torch.mean(torch.exp(y - a), dim=1)) + a.squeeze(1)
    
    return lme 


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)

    # compute kld
    # KL[N(μ_q, σ²_q) || N(μ_p, σ²_p)] = 
    # 0.5 * (log(σ²_p/σ²_q) + (σ²_q + (μ_q - μ_p)²)/σ²_p - 1)
    
    # Log variance terms: log(σ²_p) - log(σ²_q)
    var_ratio = logvar_p - logvar_q
    
    # Mean squared difference term: (μ_q - μ_p)²/σ²_p
    mean_diff_sq = ((mu_q - mu_p) ** 2) / torch.exp(logvar_p)
    
    # Variance ratio term: σ²_q/σ²_p
    var_term = torch.exp(logvar_q - logvar_p)
    
    # Combine all terms
    kl_per_dim = 0.5 * (var_ratio + var_term + mean_diff_sq - 1)
    
    # Sum over all dimensions except batch
    kl_gg = kl_per_dim.sum(dim=1)
    
    return kl_gg


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

    # compute Monte Carlo estimate of KL divergence
    # KL(q||p) = E_q[log q(z) - log p(z)]
    
    # Sample from q(z|x)
    std_q = torch.exp(0.5 * logvar_q)
    eps = torch.randn_like(std_q)
    z = mu_q + eps * std_q
    
    # Compute log q(z|x) for these samples
    log_q = -0.5 * (math.log(2 * math.pi) + 
                    logvar_q + 
                    ((z - mu_q) ** 2) / torch.exp(logvar_q))
    log_q = log_q.sum(dim=2)  # Sum over dimensions for each sample
    
    # Compute log p(z) for the same samples
    log_p = -0.5 * (math.log(2 * math.pi) + 
                    logvar_p + 
                    ((z - mu_p) ** 2) / torch.exp(logvar_p))
    log_p = log_p.sum(dim=2)  # Sum over dimensions for each sample
    
    # KL divergence for each sample
    kl_samples = log_q - log_p
    
    # Average over samples
    kl_mc = torch.mean(kl_samples, dim=1)
    
    return kl_mc
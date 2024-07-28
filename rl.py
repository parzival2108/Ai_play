import numpy as np
import pandas as p
import torch.distributions as dist
import torch
import pyro
torch.tensor([1,2,3])



def prior(rain):
    return rain


def likelihood(cloud_given_rain):
    return cloud_given_rain


def marginal(priors, likelihoods):
    priors = prior(priors)
    likelihoods = likelihood(likelihoods)
    total_prob = (likelihoods*priors) + ((1-likelihoods) * (1-priors))
    return total_prob


def posterior(priors, likelihoods):
    marginals = marginal(priors, likelihoods)
    posteriors = (likelihood(likelihoods) * prior(priors))/marginals
    return posteriors

probability_of_rain = posterior(0.3, 0.7)*100
print(f"probability of rain: {probability_of_rain: .1f}%")














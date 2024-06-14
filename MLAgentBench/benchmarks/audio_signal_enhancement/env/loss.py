import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import convolve

def sigmoid_loss(z):
    
    """
    The sigmoid loss for binary classification based on empirical risk minimization. See R. Kiryo, G. Niu, 
    M. C. du Plessis, and M. Sugiyama, “Positive-unlabeled learning with non-negative risk estimator,” in 
    Proc. NIPS, CA, USA, Dec. 2017.
    
    Args:
        z: The margin.
    
    Returns:
        The loss value.
    """

    return F.sigmoid(-z)


def sa_loss(yhat, mix_stft, cln_stft):

    """
    The signal approximation loss for speech enhancement.
    
    Args:
        yhat: The output of the convolutional neural network before the non-linear activation function.
        mix_stft: The noisy speech in the short-time Fourier transform domain.
        cln_stft: The clean speech in the short-time Fourier transform domain.
    
    Returns:
        The signal approximation loss.
    """

    return torch.mean((F.sigmoid(yhat) * torch.abs(mix_stft) - torch.abs(cln_stft)) ** 2)


def weighted_pu_loss(y, yhat, mix_stft, beta, gamma = 1.0, prior = 0.5, ell = sigmoid_loss, p = 1, mode = 'nn'):
    
    """
    The weighted loss for learning from positive and unlabelled data (PU learning). See N. Ito and M. Sugiyama, 
    "Audio signal enhancement with learning from positive and unlabelled data," arXiv, 
    https://arxiv.org/abs/2210.15143. 
    
    Args:
        y: A mask indicating whether each time-frequency component is positive (1) or unlabelled (0).
        yhat: The output of the convolutional neural network before the non-linear activation function.
        mix_stft: The noisy speech in the short-time Fourier transform domain.
        beta: The beta parameter in PU learning using non-negative empirical risk.
        gamma: The gamma parameter in PU learning using non-negative empirical risk.
        prior: The class prior for the positive class.
        ell: The loss function for each time-frequency component such as the sigmoid loss.
        p: The exponent for the weight. p = 1.0 corresponds to weighting by the magnitude spectrogram of the 
            input noisy speech. p = 0.0 corresponds to no weighting.
        mode: The type of the empirical risk in PU learning. 'nn' corresponds to the non-negative empirical 
            risk. 'unbiased' corresponds to the unbiased empirical risk.
        
    Returns:
        The weighted loss.
    """
    
    # TODO: Finish the weighted loss function for learning from positive and unlabelled data (PU learning).
    pass

def MixIT_loss(yhat, mix_stft, noise_stft, noisy_stft):
    
    """
    The loss for mixture invariant training (MixIT). 
    
    Args:
        yhat: The output of the convolutional neural network before the non-linear activation function. There are three output
            channels corresponding to an enhanced speech and two noise estimates.
        mix_stft: The mixture of a noisy signal example and a noise example in the short-time Fourier transform domain.
        noise_stft: The noise example in the short-time Fourier transform domain.
        noisy_stft: The noisy speech example in the short-time Fourier transform domain.
        
    Returns:
        The loss.
    """

    loss1 = torch.mean(((F.sigmoid(yhat[:,0,:,:]) + F.sigmoid(yhat[:,1,:,:])) * torch.abs(mix_stft) - torch.abs(noisy_stft)) ** 2)
    loss1 += torch.mean((F.sigmoid(yhat[:,2,:,:])  * torch.abs(mix_stft) - torch.abs(noise_stft)) ** 2)
    loss2 = torch.mean(((F.sigmoid(yhat[:,0,:,:]) + F.sigmoid(yhat[:,2,:,:])) * torch.abs(mix_stft) - torch.abs(noisy_stft)) ** 2)
    loss2 += torch.mean((F.sigmoid(yhat[:,1,:,:])  * torch.abs(mix_stft) - torch.abs(noise_stft)) ** 2)
    
    return torch.minimum(loss1, loss2)

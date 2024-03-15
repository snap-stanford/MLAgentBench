
import ultraimport
weighted_pu_loss = ultraimport('./loss.py', 'weighted_pu_loss')
# from ..env.loss import weighted_pu_loss


import torch
import torch.nn.functional as F
import numpy as np


DELTA = 10 ** -3
ITERATIONS = 5

def golden(y, yhat, mix_stft, prior = 0.5, p = 1):
    
    """
    The weighted loss for learning from positive and unlabelled data (PU learning). 
    
    Args:
        y: A mask indicating whether each time-frequency component is positive (1) or unlabelled (0).
        yhat: The output of the convolutional neural network before the non-linear activation function.
        mix_stft: The noisy speech in the short-time Fourier transform domain.
        prior: The class prior for the positive class.
        p: The exponent for the weight. p = 1.0 corresponds to weighting by the magnitude spectrogram of the 
            input noisy speech. p = 0.0 corresponds to no weighting.
        
    Returns:
        The weighted loss.
    """
    
    epsilon = 10 ** -7
    weight = (torch.abs(mix_stft) + epsilon) ** p
    pos = prior * torch.sum(y * F.sigmoid(-yhat) * weight) / (torch.sum(y) + epsilon)
    neg = torch.maximum(torch.sum((1. - y) * F.sigmoid(yhat) * weight) / (torch.sum(1. - y) + epsilon) - prior * torch.sum(y * F.sigmoid(yhat) * weight) / (torch.sum(y) + epsilon), torch.tensor(0))

    return pos+neg


def test():
    for i in range(ITERATIONS):
        size = (16,1,513,196)
        yhat_low, yhat_high = -0.5, 0.5
        yhat = (yhat_low - yhat_high) * torch.rand(*size) + yhat_high
        y = torch.bernoulli(torch.empty(*size).uniform_(0, 1))
        mix_stft = torch.randn(*size, dtype=torch.cfloat)*1000
        prior = 0.5
        expected = golden(y,yhat,mix_stft,prior)
        actual = weighted_pu_loss(y,yhat,mix_stft,prior, mode = 'unbiased')
        if actual is None:
            raise Exception("Nontype detected, loss function didn't implement.")
        # torch.testing.assert_close(expected, actual)

        if expected - DELTA <= actual <= expected + DELTA:
            # print("Generated loss function pass the test case.")
            continue
        else:
            raise Exception(f"The generated loss function failed the test case. For random generated input, where y = {y}, yhat = {yhat}, mix_stft = {mix_stft}, the expected output was {expected}, but the error output obtained was {actual}. Please revise the loss function.")
    print("The generated loss function passes all the test cases.")
    return None

if __name__ == "__main__":

    test()
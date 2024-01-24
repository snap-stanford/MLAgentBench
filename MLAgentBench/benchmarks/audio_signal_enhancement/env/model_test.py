from model import *

import torch

ITERATIONS = 5

def make_CNN_golden(droprate):
    """
    Implement a CNN  Network using the following specification.
    The learned model is based on 2D convolutional neural network with dropout and ReLU as the activation function. The specific specification is `Conv2d(1, 8, 3)-Conv2d(8, 8, 3)- Conv2d(8, 16, 3)-Conv2d(16, 16, 3)-Conv2d(16, 32, 3)-Conv2d(32, 32, 3)-Conv2d(32, 64, 3)-Conv2d(64, 64, 3)-Conv2d(64, 128, 1)- Conv2d(128, 128, 1)-Conv2d(128, 1, 1)`  Conv2d(Cin, Cout, K) is a two-dimensional convolutional layer with Cin input chatorch.nnels, Cout output chatorch.nnels, a kernel size of K × K, a stride of (1,1), and no padding. All but the last convolutional layer are followed by a rectified linear unit (ReLU) and then a dropout layer with a dropout rate of droprate. The size of the input spectrogram patch is set to that of the receptive field of the network (i.e., 17 × 17) so that the Ctorch.nn output size is 1 × 1.
    """

    layers = [
        torch.nn.Conv2d(1, 8, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(droprate),
        
        torch.nn.Conv2d(8, 8, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(droprate),
        
        torch.nn.Conv2d(8, 16, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(droprate),
        
        torch.nn.Conv2d(16, 16, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(droprate),
        
        torch.nn.Conv2d(16, 32, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(droprate),
        
        torch.nn.Conv2d(32, 32, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(droprate),
        
        torch.nn.Conv2d(32, 64, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(droprate),
        
        torch.nn.Conv2d(64, 64, 3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(droprate),
        
        torch.nn.Conv2d(64, 128, 1, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(droprate),
        
        torch.nn.Conv2d(128, 128, 1, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(droprate),
        
        torch.nn.Conv2d(128, 1, 1, stride=1, padding="same")
    ]
    return torch.nn.Sequential(*layers) 


def test_architecture():
    model1 = make_CNN_golden(0.3)
    model2 = make_CNN(channels = 8, droprate=0.3, method = 'PU', blocks=4, fcblocks=1)
    # get state dictionaries
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()


    # compare keys:
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    try:
        model2.load_state_dict(state_dict1)
    except:
        raise Exception(
            f"The generated model architecture did not pass the test case. "
            f"The generated model ({model2}) does not match the expected architecture. "
            f"Please review and revise the model accordingly."
        )
    if keys1 == keys2:
        pass
    else:
        raise Exception(
            f"The generated model architecture did not pass the test case. "
            f"The generated model ({model2}) does not match the expected architecture. "
            f"Please review and revise the model accordingly."
        )


def test_cases():
    # Set a seed for reproducibility
    torch.manual_seed(42)

    #initialize the models
    golden = make_CNN_golden(0.3)
    golden_parameters = golden.state_dict()

    # set the
    generated_model = make_CNN(channels=8, droprate=0.3, method='PU', blocks=4, fcblocks=1)
    generated_model.load_state_dict(golden_parameters)

    # Set models to evaluation mode
    for model in [golden, generated_model]:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    for i in range(ITERATIONS):
        size = (16, 1, 513, 196)
        x = torch.rand(*size)
        golden_y =golden.forward(x)
        yhat = generated_model.forward(x)

        try:
            torch.testing.assert_close(golden_y,yhat)
        except:
            raise Exception(
                f"The generated model architecture did not pass the test cases. "
                f"The generated model({generated_model}) does not match the expected architecture. "
                f"Please review and adjust the model architecture."
            )


if __name__ == "__main__":
    test_architecture()
    test_cases()
    print("The generated model has successfully passed both the architecture check and the test cases check.")
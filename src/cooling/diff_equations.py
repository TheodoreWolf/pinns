import torch
from functorch import make_functional, grad, vmap
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def grad(outputs, inputs):
    """compute the derivative of outputs associated with inputs.
    Params
    ======
    outputs: (N, 1) tensor
    inputs: (N, D) tensor
    """
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )


def cooling_law(time, Tenv, T0, R):
    T = Tenv + (T0 - Tenv) * np.exp(-R * time)
    return T

from typing import Callable

import torch
from torch import nn
from functorch import make_functional, grad, vmap


class NNApproximator(nn.Module):
    def __init__(
        self,
        num_inputs: int = 1,
        num_outputs: int = 1,
        num_hidden: int = 1,
        dim_hidden: int = 1,
        act: nn.Module = nn.Tanh(),
    ) -> None:
        """Simple neural network with linear layers and non-linear activation function

        This class is used as universal function approximator for the solution of
        partial differential equations using PINNs

        Args:
            num_inputs (int, optional): The number of input dimensions
            num_outputs (int, optional): The number of outputs of the model, in general is 1
            num_hidden (int, optional): The number of hidden layers in the mode
            dim_hidden (int, optional): The number of neurons for each hidden layer
            act (nn.Module, optional): The type of non-linear activation function to be used
        """
        super().__init__()

        self.layer_in = nn.Linear(num_inputs, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, num_outputs)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.layer_in(x))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        return self.layer_out(out)


def make_forward_fn(
    num_inputs: int = 1,
    num_outputs: int = 1,
    num_hidden: int = 1,
    dim_hidden: int = 1,
    act: nn.Module = nn.Tanh(),
    derivative_order: int = 1,
) -> tuple[Callable, tuple[torch.Tensor], dict[int, Callable]]:
    """Make a functional forward pass and gradient functions

    This function creates a functional model using the NNApproximator class
    and then returns a tuple with the functional model itself, the parameters
    initialized randomly and the composable v-mapped version of the forward pass
    and of higher-order derivatives with respect to the inputs as
    specified by the input argument `derivative_order`

    The forward pass is created as a closure

    Args:
        num_inputs (int, optional): The number of input dimensions
        num_outputs (int, optional): The number of outputs of the model, in general is 1
        num_hidden (int, optional): Number of hidden layers in the NN model
        dim_hidden (int, optional): Number of neurons per hidden layer in the NN model
        act (nn.Module, optional): The type of non-linear activation function to be used
        derivative_order (int, optional): Up to which order return functions for computing the
            derivative of the model with respect to the inputs

    Returns:
        tuple: A tuple with the functional model itself, the parameters
            and the a dictionary where values are the composable v-mapped 
            version of the forward pass and higher-order gradient functions 
            for efficient batching over the inputs
    """
    model = NNApproximator(
        num_hidden=num_hidden,
        dim_hidden=dim_hidden,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        act=act,
    )
    fmodel, params = make_functional(model)

    def f(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # only a single element is support thus unsqueeze must be applied
        # for batching multiple inputs, vmap must be used as below
        x_ = x.unsqueeze(0)
        res = fmodel(params, x_).squeeze(0)
        return res

    f_vmap = vmap(f, in_dims=(0, None))

    fns = {}
    fns[0] = f_vmap

    dfunc = f
    for i in range(derivative_order):
        dfunc = grad(dfunc)
        fns[i + 1] = vmap(dfunc, in_dims=(0, None))

    return fmodel, params, fns
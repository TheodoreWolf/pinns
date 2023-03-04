from typing import Callable
import argparse

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import torchopt

from pinn import make_forward_fn


R = 1.0  # rate of maximum population growth parameterizing the equation
X_BOUNDARY = 0.0  # boundary condition coordinate
F_BOUNDARY = 0.5  # boundary condition value


def make_loss_fn(f: Callable, dfdx: Callable) -> Callable:
    """Make a function loss evaluation function

    The loss is computed as sum of the interior MSE loss (the differential equation residual)
    and the MSE of the loss at the boundary

    Args:
        f (Callable): The functional forward pass of the model used a universal function approximator. This
            is a function with signature (x, params) where `x` is the input data and `params` the model
            parameters
        dfdx (Callable): The functional gradient calculation of the universal function approximator. This
            is a function with signature (x, params) where `x` is the input data and `params` the model
            parameters

    Returns:
        Callable: The loss function with signature (params, x) where `x` is the input data and `params` the model
            parameters. Notice that a simple call to `dloss = functorch.grad(loss_fn)` would give the gradient
            of the loss with respect to the model parameters needed by the optimizers
    """

    def loss_fn(params: torch.Tensor, x: torch.Tensor):

        # interior loss
        f_value = f(x, params)
        interior = dfdx(x, params) - R * f_value * (1 - f_value)

        # boundary loss
        x0 = X_BOUNDARY
        f0 = F_BOUNDARY
        x_boundary = torch.tensor([x0])
        f_boundary = torch.tensor([f0])
        boundary = f(x_boundary, params) - f_boundary

        loss = nn.MSELoss()
        loss_value = loss(interior, torch.zeros_like(interior)) + loss(
            boundary, torch.zeros_like(boundary)
        )

        return loss_value

    return loss_fn


if __name__ == "__main__":

    # make it reproducible
    torch.manual_seed(42)

    # parse input from user
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num-hidden", type=int, default=5)
    parser.add_argument("-d", "--dim-hidden", type=int, default=5)
    parser.add_argument("-b", "--batch-size", type=int, default=30)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-1)
    parser.add_argument("-e", "--num-epochs", type=int, default=100)

    args = parser.parse_args()

    # configuration
    num_hidden = args.num_hidden
    dim_hidden = args.dim_hidden
    batch_size = args.batch_size
    num_iter = args.num_epochs
    tolerance = 1e-8
    learning_rate = args.learning_rate
    domain = (-5.0, 5.0)

    # function versions of model forward, gradient and loss
    fmodel, params, funcs = make_forward_fn(
        num_hidden=num_hidden, dim_hidden=dim_hidden, derivative_order=1
    )

    f = funcs[0]
    dfdx = funcs[1]
    loss_fn = make_loss_fn(f, dfdx)

    # choose optimizer with functional API using functorch
    optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))

    # train the model
    loss_evolution = []
    for i in range(num_iter):

        # sample points in the domain randomly for each epoch
        x = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1])

        # update the parameters
        loss = loss_fn(params, x)
        params = optimizer.step(loss, params)

        print(f"Iteration {i} with loss {float(loss)}")
        loss_evolution.append(float(loss))

    # plot solution on the given domain
    x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)
    f_eval = f(x_eval, params)
    analytical_sol_fn = lambda x: 1.0 / (1.0 + (1.0/F_BOUNDARY - 1.0) * np.exp(-R * x))
    x_eval_np = x_eval.detach().numpy()
    x_sample_np = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1]).detach().numpy()

    fig, ax = plt.subplots()

    ax.scatter(x_sample_np, analytical_sol_fn(x_sample_np), color="red", label="Sample training points")
    ax.plot(x_eval_np, f_eval.detach().numpy(), label="PINN final solution")
    ax.plot(
        x_eval_np,
        analytical_sol_fn(x_eval_np),
        label=f"Analytic solution",
        color="green",
        alpha=0.75,
    )
    ax.set(title="Logistic equation solved with NNs", xlabel="t", ylabel="f(t)")
    ax.legend()

    fig, ax = plt.subplots()
    ax.semilogy(loss_evolution)
    ax.set(title="Loss evolution", xlabel="# epochs", ylabel="Loss")
    ax.legend()

    plt.show()
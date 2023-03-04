import torch
from functorch import make_functional, grad, vmap

# from . import network

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MU = 0.5
_g = torch.tensor([[0.0, -9.81]]).to(DEVICE)


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


def loss_pde(model):

    t = torch.linspace(0, 6, steps=100, requires_grad=True).to(DEVICE)[:, None]
    s = model(t)
    s_x, s_y = s[:, 0], s[:, 1]

    v_x = grad(s_x[:, None], t)[0]
    a_x = grad(v_x, t)[0]
    v_y = grad(s_y[:, None], t)[0]
    a_y = grad(v_y, t)[0]

    a = torch.concat([a_x, a_y], axis=1)
    v = torch.concat([v_x, v_y], axis=1)

    v = v.detach()
    v_norm = torch.norm(v, dim=1, keepdim=True)

    loss = torch.nn.MSELoss()(a, -_MU * v * v_norm - _g)
    return loss


def f(x: torch.Tensor, params: torch.Tensor, fmodel) -> torch.Tensor:
    # only a single element is supported thus unsqueeze must be applied
    # for batching multiple inputs, `vmap` must be used as below
    x_ = x.unsqueeze(0)
    res = fmodel(params, x_).squeeze(0)
    return res


def get_gradients():
    # use `vmap` primitive to allow efficient batching of the input
    f_vmap = vmap(f, in_dims=(0, None))

    # return function for computing higher order gradients with respect
    # to input by simply composing `grad` calls and use again `vmap` for
    # efficient batching of the input
    dfdx = vmap(grad(f), in_dims=(0, None))
    d2fdx2 = vmap(grad(grad(f)), in_dims=(0, None))

    R = 1.0  # rate of maximum population growth parameterizing the equation
    X_BOUNDARY = 0.0  # boundary condition coordinate
    F_BOUNDARY = 0.5  # boundary condition value


def loss_fn(params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

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

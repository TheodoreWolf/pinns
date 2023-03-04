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


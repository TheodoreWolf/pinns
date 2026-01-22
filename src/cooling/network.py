import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thdat

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np_to_th(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(n_samples, -1)


class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=30,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=1e-3,
        loss2=None,
        loss2_weight=0.1,
        Tmax = 1000,
        T0 = 100,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.n_units = n_units

        # For scaling input and output of NN model
        self.Tmax = Tmax
        self.T0 = T0

        # Multiple hidden layers is massive overkill here
        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.n_units),
            nn.GELU(),
            nn.Linear(self.n_units, self.n_units),
            nn.GELU(),
            #nn.Linear(self.n_units, self.n_units),
            #nn.Tanh(),
            #nn.Linear(self.n_units, self.n_units),
            #nn.Tanh(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

    def forward(self, x):
        # Scale input and output so of order unity for better training
        h = self.layers(x/self.Tmax)
        out = self.T0*self.out(h)

        return out

    def fit(self, X, y):
        Xt = np_to_th(X)
        yt = np_to_th(y)

        # Setup BFGS optimiser rather than Adam since we're doing full-batch
        optimiser = optim.LBFGS(self.parameters(), lr=self.lr, line_search_fn='strong_wolfe')
        def closure():
            # zero gradients to compute fresh grads for this closure
            try:
                self.optimiser.zero_grad()
            except Exception:
                pass
            self.zero_grad()
            # compute loss
            loss = loss_fn(yt, self(Xt))
            # backward to populate `.grad`
            loss.backward()

            return loss

        # Define loss function
        def loss_fn(yt, outputs):

            if (self.loss2 is None):
                return self.loss(yt, outputs)
            else:
                #print("loss : ", self.loss(yt, outputs).item(), " loss2 : ", self.loss2(self).item())
                return self.loss(yt, outputs) + self.loss2_weight * self.loss2(self)


        self.train()
        losses = []
        for ep in range(self.epochs):
            optimiser.zero_grad()
            #outputs = self.forward(Xt)
            loss = optimiser.step(closure)
            losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")

        return losses

    def predict(self, X):
        self.eval()
        out = self.forward(np_to_th(X))
        return out.detach().cpu().numpy()


class NetDiscovery(Net):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=0.001,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        super().__init__(
            input_dim, output_dim, n_units, epochs, loss, lr, loss2, loss2_weight
        )

        self.r = nn.Parameter(data=torch.tensor([0.]))

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
        loss2,
        n_units=100,
        epochs=100,
        loss=nn.MSELoss(),
        lr=1e-3,
        loss2_weight=0.1,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.lr = lr
        self.n_units = n_units
        self.loss2_weight = loss2_weight

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.n_units),
            nn.GELU(),
            nn.Linear(self.n_units, self.n_units),
            nn.GELU(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)

        return out

    def fit(self, X, y):
        n_samples = len(X)
        Xt = torch.from_numpy(X).to(torch.float).to(DEVICE).reshape(n_samples, -1)
        yt = torch.from_numpy(y).to(torch.float).to(DEVICE).reshape(n_samples, -1)

        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            optimiser.zero_grad()
            outputs = self.forward(Xt)
            loss = self.loss(yt, outputs)
            if self.loss2:
                loss += self.loss2_weight * self.loss2(self)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses

    def predict(self, X):
        self.eval()
        n_samples = len(X)
        out = self.forward(
            torch.from_numpy(X).to(torch.float).to(DEVICE).reshape(n_samples, -1)
        )
        return out.detach().cpu().numpy()

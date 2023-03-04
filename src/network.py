import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thdat
from functorch import make_functional

import diff_equations as de

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(
        self, input_dim, output_dim, pde_loss, n_units=100, epochs=100, loss=nn.MSELoss(), lr=1e-3
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.pde_loss = pde_loss
        self.lr = lr
        self.batch_size = 32
        self.n_units = n_units

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
        X_torch = torch.from_numpy(X).to(torch.float).to(DEVICE).reshape(n_samples, -1)
        y_torch = torch.from_numpy(y).to(torch.float).to(DEVICE).reshape(n_samples, -1)
        train_loader = thdat.DataLoader(
            thdat.TensorDataset(X_torch, y_torch),
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            for batch in train_loader:
                inputs, targets = batch
                optimiser.zero_grad()
                outputs = self.forward(inputs)
                loss = self.loss(targets, outputs) + self.phys_loss(self)
                # print(loss)
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

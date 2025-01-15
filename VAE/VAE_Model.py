import torch
import torch.nn.functional as F

from torch import nn

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        # encoder
        self.img_to_hid = nn.Linear(input_dim, h_dim)
        self.hid_to_mu = nn.Linear(h_dim, z_dim)
        self.hid_to_sigma = nn.Linear(h_dim, z_dim) # using a neural network to calculate the mean and std of the latent variable z

        # decoder
        self.z_to_hid = nn.Linear(z_dim, h_dim)
        self.hid_to_img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()


    def encode(self, x):
        # an encode function (network) which acts as Q_phi(z|x). uses phi as the parameter.
        h = self.relu(self.img_to_hid(x))
        mu, sigma = self.hid_to_mu(h), self.hid_to_sigma(h)

        return mu, sigma

    def decode(self, z):
        # decode function which acts as P_theta(x|z). uses theta as the parameter.
        pass

    def forward(self, x):
        pass

if __name__ == "__main__":
    x = torch.randn(1, 784)
    # vae = VariationalAutoEncoder()
    # print(vae(x).shape)
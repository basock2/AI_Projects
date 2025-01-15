import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from VAE_Model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

DEVICE = torch.device("cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 4

# Dataset Loading
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")

# Training
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader)) # batch data and its index
    for i, (x, _) in loop: # for each batch in epoch

        # Forward Pass
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM) # mini batch shape change
        x_reconstructed, mu, sigma = model(x)

        # Compute Loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        # Backpropagation
        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
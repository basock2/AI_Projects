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
NUM_EPOCHS = 3
BATCH_SIZE = 32
LR_RATE = 0.0005

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

model = model.to("cpu")
def inference(digit, num_example=1):
    # take one example image from the dataset
    for x, y in dataset:
        if y == digit:
            mu, sigma = model.encode(x.view(1,784))
            for example in range(num_example):
                epsilon = torch.rand_like(sigma)
                z = mu + sigma * epsilon
                out = model.decode(z)
                out = out.view(-1, 1, 28, 28)
                save_image(out, f"VAE/VAE_generated/generated_{digit}_ex{example}.png")
            break
    
for idx in range(10):
    inference(idx, num_example=3)
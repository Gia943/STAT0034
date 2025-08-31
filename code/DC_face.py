
import os
import torch
import numpy as np
from torch import nn

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
import torch
from torch.utils.data import Dataset, DataLoader, random_split

torch.manual_seed(42)

Z_DIMS = 32
H_DIMS = 32
N_CHANNELS = 1               # We want to generate a black and white image.
DEVICE = "cuda"               # Make sure to enable gpu.
IMG_DIMS = 64                # Size of the image
LEARNING_RATE = 2e-4         # Learning rate of the Networks

# Additional optimizer parameters.
beta_1 = 0.5
beta_2 = 0.999

class Generator(nn.Module):
    def __init__(self, z_dim: int = Z_DIMS, hidden_dims: int = H_DIMS, image_channels: int = N_CHANNELS):
        super().__init__()

        self.z_dim = z_dim
        self.gen = nn.Sequential(
            # (z) -> (hidden*8) x 4 x 4
            nn.ConvTranspose2d(z_dim, hidden_dims * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dims * 8),
            nn.ReLU(True),

            # 4 -> 8
            nn.ConvTranspose2d(hidden_dims * 8, hidden_dims * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims * 4),
            nn.ReLU(True),

            # 8 -> 16
            nn.ConvTranspose2d(hidden_dims * 4, hidden_dims * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims * 2),
            nn.ReLU(True),

            # 16 -> 32
            nn.ConvTranspose2d(hidden_dims * 2, hidden_dims, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU(True),

            # 32 -> 64
            nn.ConvTranspose2d(hidden_dims, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise):
        noise = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(noise)

def generate_noise(samples: int, z_dim: int = Z_DIMS, device=DEVICE):
    return torch.randn(samples, z_dim, device=device)

base_gen = Generator()
noise = generate_noise(10, device='cpu')
synthetic_images = base_gen(noise)


def show_generations(generations, n_rows, n_cols, figsize=(8, 5), title=None, save_loc=None):
    synthetic_images = generations.view(-1, IMG_DIMS, IMG_DIMS, N_CHANNELS).detach().cpu()

    plt.figure(figsize=figsize)
    plt.suptitle("Synthetic Images" if title is None else title)

    for index in range(n_rows * n_cols):
        plt.subplot(n_rows, n_cols, index+1)
        plt.imshow(synthetic_images[index], cmap='gray')
        plt.axis('off')

    if save_loc is not None:
        if not os.path.exists(os.path.dirname(save_loc)):
            os.makedirs(os.path.dirname(save_loc))
        plt.savefig(save_loc)

    plt.show()

class Discriminator(nn.Module):
    def __init__(self, image_channels: int = N_CHANNELS, hidden_dims: int = 64):
        super().__init__()
        # 64 → 32 → 16 → 8 → 4 → 1
        self.disc = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dims, 4, 2, 1, bias=False),   # 64->32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dims, hidden_dims * 2, 4, 2, 1, bias=False),  # 32->16
            nn.BatchNorm2d(hidden_dims * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dims * 2, hidden_dims * 4, 4, 2, 1, bias=False), # 16->8
            nn.BatchNorm2d(hidden_dims * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dims * 4, hidden_dims * 8, 4, 2, 1, bias=False), # 8->4
            nn.BatchNorm2d(hidden_dims * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dims * 8, 1, 4, 1, 0, bias=False),  # 4->1
        )

    def forward(self, images):
        return self.disc(images).view(len(images), -1)

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

gen_opt = torch.optim.Adam(generator.parameters(), lr = LEARNING_RATE, betas=(beta_1, beta_2))
disc_opt = torch.optim.Adam(discriminator.parameters(), lr = LEARNING_RATE, betas=(beta_1, beta_2))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

generator = generator.apply(weights_init)
discriminator = discriminator.apply(weights_init)

BATCH_SIZE = 8                                      
criterion = nn.BCEWithLogitsLoss()  

# Dataset
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import TensorDataset, DataLoader

faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.images  # (400, 64, 64)
X = torch.from_numpy(X).unsqueeze(1).float()
X = (X - 0.5) / 0.5

y = torch.zeros(len(X))

dataset = TensorDataset(X, y)

BATCH_SIZE = 32  
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

EPOCHS = 100
STEP_SIZE = 1000

for epoch in range(EPOCHS):

    step_size = 0

    for real, _ in tqdm(dataloader):

        curr_batch_size = len(real)
        real = real.to(DEVICE)

        disc_opt.zero_grad()

        noise_samples = generate_noise(samples = curr_batch_size)
        synthetic_images = generator(noise_samples)
        disc_syn_preds = discriminator(synthetic_images.detach())

        disc_real_preds = discriminator(real)

        disc_syn_loss = criterion(disc_syn_preds, torch.zeros_like(disc_syn_preds))
        disc_real_loss = criterion(disc_real_preds, torch.ones_like(disc_real_preds))
        discriminator_loss = (disc_syn_loss + disc_real_loss) / 2

        discriminator_loss.backward(retain_graph=True)
        disc_opt.step()

        gen_opt.zero_grad()

        noise_samples = generate_noise(samples = curr_batch_size)
        synthetic_images = generator(noise_samples)
        disc_syn_preds = discriminator(synthetic_images)

        generator_loss = criterion(disc_syn_preds, torch.ones_like(disc_syn_preds))
        generator_loss.backward()
        gen_opt.step()

        if (step_size % STEP_SIZE == 0):
            noise_vector = generate_noise(10)
            generations = generator(noise_vector)
            show_generations(
                generations, 2, 2,
                figsize=(8, 5),
                title = f"Generations at Step : {step_size} Epochs : {epoch+1}",
                save_loc = f"./Images/training_gen_{step_size}_epoch_{epoch+1}.png"
            )

        step_size += 1

noise = generate_noise(50)
syn_images = generator(noise)

show_generations(
    syn_images, 4, 6,
    figsize=(20, 10),
    title="DCGAN Model Generations",
    save_loc = "./FinalGenerations.png"
)

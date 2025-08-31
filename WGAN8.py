
import os
import torch
import numpy as np
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from torchvision.datasets import MNIST

import matplotlib.pyplot as plt

from torch import nn
from torch.nn import Module, Sequential
from torch.nn import Conv2d, ConvTranspose2d
from torch.nn import BatchNorm2d, ReLU, LeakyReLU, Tanh

!pip install pytorch-msssim
from pytorch_msssim import ssim


DEVICE = "cuda"
IMG_DIMS = 28


ZDIM = 32
HDIM = 32
IMG_CHANNELS = 1


LR = 2e-4
EPOCHS = 20                           
beta_1 = 0.5
C_LAMBDA = 10
beta_2 = 0.999
BATCH_SIZE = 16
CRITIC_STEPS = 5
DISPLAY_STEP = 500

np.random.seed(42)
torch.manual_seed(42)

class Generator(Module):

    def __init__(self, zdims: int = ZDIM, hdims: int = HDIM, img_channels: int = IMG_CHANNELS):

        super(Generator, self).__init__()

        self.zdims = zdims
        self.generator = Sequential(
            self.generator_block(ZDIM, HDIM),
            self.generator_block(HDIM, HDIM * 2, kernel_size=4, stride=1),
            self.generator_block(HDIM * 2, HDIM * 4),
            self.generator_block(HDIM * 4, img_channels, kernel_size=4, output_layer=True),
        )

    def generator_block(self, input_dims: int, output_dims: int, kernel_size: int = 3, stride: int = 2, output_layer: bool = False):


        if output_layer:
            return Sequential(
                ConvTranspose2d(input_dims, output_dims, kernel_size, stride),
                Tanh()
            )
        else:
            return Sequential(
                ConvTranspose2d(input_dims, output_dims, kernel_size, stride),
                BatchNorm2d(output_dims),
                ReLU(inplace=True)
            )

    def forward(self, noise):


        noise = noise.view(len(noise), self.zdims, 1, 1)
        synthesized_images = self.generator(noise)

        return synthesized_images

def generate_noise(n_samples: int, z_dims: int = ZDIM, device: str = DEVICE) -> torch.Tensor:
    return torch.randn(n_samples, z_dims, device=device)

def show_generations(generations, n_rows=1, n_cols=1, figsize=(8, 5), title=None, save_loc=None):

    synthetic_images = generations.view(-1, IMG_DIMS, IMG_DIMS, IMG_CHANNELS).detach().cpu()

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

noise = generate_noise(10, device='cpu')
synthetic_images = Generator()(noise)

show_generations(
    synthetic_images, 2, 4,
    figsize=(15, 5),
    title="Generator Synthetic Images",
    save_loc = "./GeneratorImages/base_generations.png"
)

class Critic(Module):
    def __init__(self, hdims:int =HDIM, img_channels:int=IMG_CHANNELS):

        super(Critic, self).__init__()

        self.critic = Sequential(
            self.critic_block(img_channels, hdims),
            self.critic_block(hdims, hdims * 2),
            self.critic_block(hdims * 2, 1, output_layer=True),
        )

    def critic_block(self, input_dims, output_dims, kernel_size=4, stride=2, output_layer=False):

        if output_layer:
            return Sequential(
                Conv2d(input_dims, output_dims, kernel_size, stride)
            )
        else:
            return Sequential(
                Conv2d(input_dims, output_dims, kernel_size, stride),
                BatchNorm2d(output_dims),
                LeakyReLU(0.2, inplace=True)
            )

    def forward(self, image):

        critic_pred = self.critic(image)
        return critic_pred.view(len(critic_pred), -1)


generator = Generator().to(DEVICE)

generator_optim = torch.optim.Adam(params = generator.parameters(), lr = LR, betas = (beta_1, beta_2))

critic = Critic().to(DEVICE)

critic_optim = torch.optim.Adam(params = critic.parameters(), lr = LR, betas = (beta_1, beta_2))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

generator = generator.apply(weights_init)
critic = critic.apply(weights_init)



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataset = MNIST(".", download=True, train=True, transform=transform)

target_digit = 8
indices = [i for i, label in enumerate(dataset.targets) if label == target_digit]

digit8_dataset = Subset(dataset, indices)

dataloader = DataLoader(
    digit8_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

def extract_gradients(critic: Module, real: torch.Tensor, fake: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:


    mixed_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = critic(mixed_images)

    gradients = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    return gradients

def gradient_penalty(gradients: torch.Tensor) -> torch.Tensor:

    gradients = gradients.view(len(gradients), -1)

    gradient_norm = gradients.norm(2, dim=1)

    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty

def compute_generator_loss(critic_fake_pred):

    return -1. * torch.mean(critic_fake_pred)

def compute_critic_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):

    return torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp





EPOCHS = 20


dir = './GANLearningCurve'
if not os.path.exists(dir):
    os.mkdir(dir)


for epoch in range(1, EPOCHS+ 1):

    CRITIC_LOSSES = []
    GENERATOR_LOSSES = []
    SSIM_SCORES = []

    step = 0
    for real, _ in tqdm(dataloader):

        curr_batch_size = len(real)
        real_images = real.to(DEVICE)

        mean_iter_critic_loss = 0

        for _ in range(CRITIC_STEPS):

            critic_optim.zero_grad()

            fake_images = generator(generate_noise(curr_batch_size))
            critic_fake_preds = critic(fake_images.detach())
            critic_real_preds = critic(real_images)

            epsilon = torch.rand(curr_batch_size, 1, 1, 1, device=DEVICE, requires_grad=True)
            gp = gradient_penalty(extract_gradients(critic, real_images, fake_images.detach(), epsilon))

            critic_loss = compute_critic_loss(critic_fake_preds, critic_real_preds, gp, C_LAMBDA)

            mean_iter_critic_loss += critic_loss.item() / CRITIC_STEPS

            critic_loss.backward(retain_graph=True)
            critic_optim.step()

        CRITIC_LOSSES += [mean_iter_critic_loss]

        generator_optim.zero_grad()
        fake_images = generator(generate_noise(curr_batch_size))
        critic_fake_pred = critic(fake_images)

        generator_loss = compute_generator_loss(critic_fake_pred)
        generator_loss.backward()
        generator_optim.step()

        GENERATOR_LOSSES += [generator_loss.item()]

        if (step % 100 == 0):
            score = ssim(real_images.detach(), fake_images.detach())
            SSIM_SCORES.append(score.detach().cpu())

        if (step % DISPLAY_STEP == 0):

            noise = generate_noise(25)
            synthetic_images = generator(noise)

            show_generations(
                synthetic_images, 2, 2,
                figsize=(8, 5),
                title=f"Generated Images\nStep: {step} Epoch: {epoch}",
                save_loc = f"./GeneratorImages/Epoch_{epoch}/synthetic_image_{step}.png"
            )

        step += 1

    plt.title(f"Learning Curve\nEpoch: {epoch}")
    plt.plot(CRITIC_LOSSES, label='Critic')
    plt.plot(GENERATOR_LOSSES, label='Generator')
    plt.grid()
    plt.legend()
    plt.savefig(f"./{dir}/LC_Epoch{epoch}.png")
    plt.show()

    plt.title("SSIM Every 100th Step")
    plt.plot(SSIM_SCORES)
    plt.grid()
    plt.savefig(f"./{dir}/SSIM_Score_{epoch}.png")
    plt.show()

noise = generate_noise(50)
syn_images = generator(noise)

show_generations(
    syn_images, 4, 6,
    figsize=(20, 10),
    title="WGAN Model Generations",
    save_loc = "./FinalGenerations.png"
)
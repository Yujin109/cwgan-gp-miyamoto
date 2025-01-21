import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from utils import save_coords, save_loss, to_cpu
from wgan_gp.models import Discriminator, Generator

cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    # 学習設定
    parser.add_argument("--n_epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--done_epoch", type=int, default=0, help="Number of epochs already completed")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--output_prefix", type=str, default="", help="Prefix for output files")
    # データセット設定
    parser.add_argument("--coords_path", type=str, default="dataset/standardized_coords.npz", help="Path to dataset")
    parser.add_argument("--perfs_path", type=str, default="dataset/standardized_perfs.npz", help="Path to dataset")
    parser.add_argument("--max_cl", type=float, default=1.58, help="Maximum CL value")
    parser.add_argument("--coord_size", type=int, default=496, help="Size of each coordinate")
    parser.add_argument("--n_classes", type=int, default=1, help="Number of classes in the dataset")
    # 最適化設定
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--b1", type=float, default=0, help="Adam beta1")
    parser.add_argument("--b2", type=float, default=0.9, help="Adam beta2")
    # モデル設定
    parser.add_argument("--latent_dim", type=int, default=3, help="Latent space dimensionality")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels")
    parser.add_argument("--n_critic", type=int, default=5, help="Discriminator steps per generator step")
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="Gradient penalty weight")

    return parser.parse_args()


# Function to configure dataset
def load_dataset(coords_path, perfs_path):
    coords_npz = np.load(coords_path)
    coords = coords_npz["coords"]
    coord_mean = coords_npz["mean"]
    coord_std = coords_npz["std"]

    perfs_npz = np.load(perfs_path)
    perfs = perfs_npz["perfs"]
    perf_mean = perfs_npz["mean"]
    perf_std = perfs_npz["std"]

    dataset = TensorDataset(torch.tensor(coords), torch.tensor(perfs))
    coord_stats = {"mean": coord_mean, "std": coord_std}
    perf_stats = {"mean": perf_mean, "std": perf_std}

    return dataset, coord_stats, perf_stats


# Function to compute gradient penalty
def compute_gradient_penalty(discriminator, real_samples, fake_samples, labels):
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1)))
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates, labels)
    fake = Variable(FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


# Function to generate and save samples
def generate_and_save_samples(generator, coord_stats, prefix, epoch=None, num_samples=15):
    z = Variable(FloatTensor(np.random.normal(0, 1, (num_samples, opt.latent_dim))))
    labels = Variable(FloatTensor(opt.max_cl * np.random.random_sample(size=(num_samples, opt.n_classes))))
    gen_coords = to_cpu(generator(z, labels)).detach().numpy()
    gen_coords = gen_coords * coord_stats["std"] + coord_stats["mean"]

    if epoch is not None:
        save_coords(gen_coords, labels, f"wgan_gp/results/{prefix}samples_epoch_{str(epoch).zfill(3)}.png")
    else:
        np.savez(f"wgan_gp/results/{prefix}final_samples", labels=labels, coords=gen_coords)
        save_coords(gen_coords, labels, f"wgan_gp/results/{prefix}final_samples.png")


# Main training function
def train_model(opt):

    dataset, coord_stats, perfs_stats = load_dataset(opt.coords_path, opt.perfs_path)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    generator = Generator(opt.latent_dim)
    discriminator = Discriminator()

    if opt.done_epoch > 0:
        generator.load_state_dict(
            torch.load(f"wgan_gp/results/{opt.output_prefix}generator_params_{opt.done_epoch}.pth", weights_only=True)
        )
        discriminator.load_state_dict(
            torch.load(
                f"wgan_gp/results/{opt.output_prefix}discriminator_params_{opt.done_epoch}.pth", weights_only=True
            )
        )

    if cuda:
        generator.cuda()
        discriminator.cuda()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    D_losses, G_losses = [], []
    start_time = time.time()

    for epoch in range(opt.n_epochs):
        epoch += opt.done_epoch
        for i, (coords, labels) in enumerate(dataloader):
            batch_size = coords.size(0)
            coords = coords.view(batch_size, 1, opt.coord_size).type(FloatTensor)
            labels = labels.view(batch_size, opt.n_classes).type(FloatTensor)

            optimizer_D.zero_grad()

            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            fake_coords = generator(z, labels)

            validity_real = discriminator(coords, labels)
            validity_fake = discriminator(fake_coords, labels)

            gradient_penalty = compute_gradient_penalty(discriminator, coords, fake_coords, labels)
            d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + opt.lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            if i % opt.n_critic == 0:
                optimizer_G.zero_grad()
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
                gen_coords = generator(z, labels)
                validity = discriminator(gen_coords, labels)
                g_loss = -torch.mean(validity)
                g_loss.backward()
                optimizer_G.step()

            if i == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"[Epoch {epoch}/{opt.n_epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] [Time: {elapsed_time:.2f}s]"
                )
                D_losses.append(d_loss.item())
                G_losses.append(g_loss.item())

        if epoch % 500 == 0 or epoch == opt.n_epochs - 1:
            torch.save(generator.state_dict(), f"wgan_gp/results/{opt.output_prefix}generator_params_{epoch}.pth")
            torch.save(
                discriminator.state_dict(), f"wgan_gp/results/{opt.output_prefix}discriminator_params_{epoch}.pth"
            )
            generate_and_save_samples(generator, coord_stats, opt.output_prefix, epoch=epoch)

    torch.save(generator.state_dict(), f"wgan_gp/results/{opt.output_prefix}generator_params_final.pth")
    torch.save(discriminator.state_dict(), f"wgan_gp/results/{opt.output_prefix}discriminator_params_final.pth")
    save_loss(G_losses, D_losses, path=f"wgan_gp/results/{opt.output_prefix}loss.png")


if __name__ == "__main__":
    opt = parse_args()
    train_model(opt)

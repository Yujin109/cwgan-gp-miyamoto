import argparse
import os
import sys
import time

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from diffusion.models import Diffuser, UNetCond
from utils import save_coords, save_loss, to_cpu

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
    # 最適化設定
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--b1", type=float, default=0, help="Adam beta1")
    parser.add_argument("--b2", type=float, default=0.9, help="Adam beta2")
    # モデル設定
    # parser.add_argument("--latent_dim", type=int, default=3, help="Latent space dimensionality")
    # parser.add_argument("--channels", type=int, default=1, help="Number of channels")
    # parser.add_argument("--n_critic", type=int, default=5, help="Discriminator steps per generator step")
    # parser.add_argument("--lambda_gp", type=float, default=10.0, help="Gradient penalty weight")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of timesteps in diffusion")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Starting beta value for diffusion")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Ending beta value for diffusion")
    parser.add_argument("--in_ch", type=int, default=2, help="Number of input channels")
    parser.add_argument("--time_embed_dim", type=int, default=100, help="Time embedding dimension")
    parser.add_argument("--label_dim", type=int, default=1, help="Number of labels for conditional model")

    return parser.parse_args()


# Function to configure dataset
def load_dataset(coords_path, perfs_path):
    coords_npz = np.load(coords_path)
    coords = coords_npz["coords"]
    coords = coords.reshape(-1, 2, 248)  # ADDED
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


# Function to generate and save samples
def generate_and_save_samples(diffuser, model, coord_stats, perfs_stats, prefix, epoch=None, num_samples=20):
    labels = FloatTensor(opt.max_cl * np.random.random_sample(size=(num_samples, opt.label_dim)))
    # labelsの標準化
    labels_standardized = (labels - perfs_stats["mean"]) / perfs_stats["std"]
    gen_coords = diffuser.generate_from_labels(model, labels_standardized, coord_shape=(2, 248))
    gen_coords_flatten = to_cpu(gen_coords.view(gen_coords.size(0), -1)).detach().numpy()
    gen_coords_flatten = gen_coords_flatten * coord_stats["std"] + coord_stats["mean"]

    if epoch is not None:
        save_coords(gen_coords_flatten, labels, f"diffusion/results/{prefix}samples_epoch_{str(epoch).zfill(3)}.png")
    else:
        np.savez(f"diffusion/results/{prefix}final_samples", labels=labels, coords=gen_coords)
        save_coords(gen_coords_flatten, labels, f"diffusion/results/{prefix}final_samples.png")


# Main training function
def train_model(opt):

    dataset, coord_stats, perfs_stats = load_dataset(opt.coords_path, opt.perfs_path)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    diffuser = Diffuser(
        num_timesteps=opt.num_timesteps, beta_start=opt.beta_start, beta_end=opt.beta_end, device="cpu"
    )
    model = UNetCond(in_ch=opt.in_ch, time_embed_dim=opt.time_embed_dim, label_dim=opt.label_dim)

    if opt.done_epoch > 0:
        model.load_state_dict(
            torch.load(f"diffusion/results/{opt.output_prefix}params_{opt.done_epoch}.pth", weights_only=True)
        )

    if cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    losses = []
    start_time = time.time()

    for epoch in tqdm(range(opt.n_epochs), desc="Training", unit="epoch"):
        epoch += opt.done_epoch
        loss_sum = 0.0
        cnt = 0
        for i, (coords, labels) in tqdm(enumerate(dataloader)):
            batch_size = coords.size(0)
            coords = coords.view(batch_size, 2, 248).type(FloatTensor)
            labels = labels.view(batch_size, opt.label_dim).type(FloatTensor)
            timesteps = torch.randint(1, opt.num_timesteps + 1, (batch_size,))

            optimizer.zero_grad()

            coords_noisy, noise = diffuser.add_noise(coords, timesteps)

            noise_pred = model(coords_noisy, timesteps, labels)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1

        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        elapsed_time = time.time() - start_time
        print(f"[Epoch {epoch}/{opt.n_epochs}] [loss: {loss_avg:.4f}] [Time: {elapsed_time:.2f}s]")

        if epoch % 200 == 0 or epoch == opt.n_epochs - 1:
            torch.save(model.state_dict(), f"diffusion/results/{opt.output_prefix}params_{epoch}.pth")
            generate_and_save_samples(diffuser, model, coord_stats, perfs_stats, opt.output_prefix, epoch=epoch)

    torch.save(model.state_dict(), f"diffusion/results/{opt.output_prefix}params_final.pth")
    save_loss(losses, path=f"diffusion/results/{opt.output_prefix}loss.png")


if __name__ == "__main__":
    opt = parse_args()
    train_model(opt)

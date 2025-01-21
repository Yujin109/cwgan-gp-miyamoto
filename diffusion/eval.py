if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from calc_cl import get_cls
from utils import to_cpu
from wgan_gp.archives.models import Generator

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Evaluator:
    def __init__(self, model_path, coords_npz, perfs_npz, output_prefix, model_name):
        state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        self.generator = Generator(3)
        self.generator.load_state_dict(state_dict)
        self.generator.eval()
        self.latent_dim = 3

        self.coords = {
            "standardized_data": (coords_npz["coords"]),
            "mean": coords_npz["mean"],
            "std": coords_npz["std"],
        }

        self.coords["data"] = self.rev_standardize(self.coords["standardized_data"])

        self.cls = {
            "standardized_data": perfs_npz["perfs"],
            "mean": perfs_npz["mean"],
            "std": perfs_npz["std"],
        }

        # self.cls["data"] = self.rev_standardize_cl(self.cls["standardized_data"])
        self.cls["data"] = self.cls["standardized_data"]  # TODO: remove this line

        self.output_prefix = output_prefix
        self.model_name = model_name

    def rev_standardize(self, coords):
        return coords * self.coords["std"] + self.coords["mean"]

    def standardize_cl(self, cl):
        return (cl - self.cls["mean"]) / self.cls["std"]

    def rev_standardize_cl(self, cl):
        return cl * self.cls["std"] + self.cls["mean"]

    def calculate_v(self, generated_coords, valid_mask=None):
        """
        Calculate the maximum of the minimum Euclidean distances between each generated wing shape
        and the training data. If valid_mask is provided, only consider valid generated coordinates.
        """
        max_min_dist = 0
        gen_idx, train_idx = -1, -1

        if valid_mask is not None:
            generated_coords = generated_coords[valid_mask]

        for i, gen_coord in enumerate(generated_coords):
            min_dist = float("inf")
            min_train_idx = -1
            for j, train_coord in enumerate(self.coords["data"]):
                dist = np.linalg.norm(gen_coord - train_coord)
                if dist < min_dist:
                    min_dist = dist
                    min_train_idx = j

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                gen_idx = i
                train_idx = min_train_idx

        return max_min_dist, gen_idx, train_idx

    def plot_overlap(self, gen_coord, train_coord, cl_label, cl_gen, gen_idx, train_idx, suffix=""):
        plt.figure()
        x_gen, y_gen = gen_coord.reshape(2, -1)
        x_train, y_train = train_coord.reshape(2, -1)

        plt.plot(x_gen, y_gen, color="purple", label="Generated")
        plt.plot(x_train, y_train, color="cyan", label="Training")
        plt.title(f"CL={cl_label}, Generated CL={cl_gen}, Gen idx={gen_idx}, Train idx={train_idx}")
        plt.legend()
        plt.savefig(f"wgan_gp/results/{self.output_prefix}{self.model_name}_overlap_{gen_idx}_{train_idx}{suffix}.png")
        plt.close()

    def plot_samples(self, gen_coords, cl_r, cl_label):
        num_samples = len(gen_coords)
        cols = 5
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        fig.suptitle(f"CL={cl_label}")

        for i, (coord, cl) in enumerate(zip(gen_coords, cl_r)):
            ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
            x, y = coord.reshape(2, -1)
            color = "red" if np.isnan(cl) else "blue"
            ax.plot(x, y, color=color)
            title = "nan" if np.isnan(cl) else f"CL={cl:.4f}"
            ax.set_title(title)

        # Hide unused subplots
        for i in range(num_samples, rows * cols):
            ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"wgan_gp/results/{self.output_prefix}{self.model_name}_samples_CL_{cl_label}.png")
        plt.close()

    def generate_and_evaluate(self, cl_range, samples_per_label=10):
        cl_c_list, cl_r_list, gen_coords_list = [], [], []
        convergence_rates, mse_list, diversity_l2_list, diversity_l1_list = [], [], [], []

        max_min_dist, max_gen_idx, max_train_idx = 0, -1, -1
        max_min_dist_valid, max_gen_idx_valid, max_train_idx_valid = 0, -1, -1

        for cl_c in tqdm(cl_range, desc="Evaluating Labels"):
            # Standardize CL for model input
            # cl_c_std = self.standardize_cl(cl_c)
            cl_c_std = cl_c.copy()  # TODO: remove this line
            # Generate wing shapes
            z = Variable(FloatTensor(np.random.normal(0, 1, (samples_per_label, self.latent_dim))))
            labels = Variable(torch.full((samples_per_label, 1), cl_c_std, dtype=torch.float32).type(FloatTensor))
            gen_coords = self.rev_standardize(to_cpu(self.generator(z, labels)).detach().numpy())

            # Evaluate CL values
            cl_r = get_cls(gen_coords)
            valid_mask = ~np.isnan(cl_r)

            # Calculate metrics
            convergence_rate = valid_mask.mean()
            mse = np.mean((cl_r[valid_mask] - cl_c) ** 2) if valid_mask.any() else np.nan
            diversity_l2 = (
                np.mean(np.linalg.norm(gen_coords[valid_mask] - gen_coords[valid_mask].mean(axis=0), axis=2))
                if valid_mask.any()
                else np.nan
            )
            diversity_l1 = (
                np.mean(np.sum(np.abs(gen_coords[valid_mask] - gen_coords[valid_mask].mean(axis=0)), axis=2))
                if valid_mask.any()
                else np.nan
            )

            # Save results for this label
            cl_c_list.append([cl_c] * samples_per_label)
            cl_r_list.append(cl_r)
            gen_coords_list.append(gen_coords)
            convergence_rates.append(convergence_rate)
            mse_list.append(mse)
            diversity_l2_list.append(diversity_l2)
            diversity_l1_list.append(diversity_l1)

            # Plot specific CL values
            if cl_c in [0.01, 0.5, 1.0, 1.5]:
                self.plot_samples(gen_coords, cl_r, cl_c)

        # Flatten lists for saving
        cl_c_array = np.concatenate(cl_c_list)
        cl_r_array = np.concatenate(cl_r_list)
        gen_coords_array = np.vstack(gen_coords_list)

        # Calculate v (maximum minimum distance)
        max_min_dist, max_gen_idx, max_train_idx = self.calculate_v(gen_coords_array)
        self.plot_overlap(
            gen_coords_array[max_gen_idx],
            self.coords["data"][max_train_idx],
            cl_c_array[max_gen_idx],
            cl_r_array[max_gen_idx],
            max_gen_idx,
            max_train_idx,
        )

        # Calculate v for valid CLs only
        max_min_dist_valid, max_gen_idx_valid, max_train_idx_valid = self.calculate_v(
            gen_coords_array, valid_mask=~np.isnan(cl_r_array)
        )
        self.plot_overlap(
            gen_coords_array[max_gen_idx_valid],
            self.coords["data"][max_train_idx_valid],
            cl_c_array[max_gen_idx_valid],
            cl_r_array[max_gen_idx_valid],
            max_gen_idx_valid,
            max_train_idx_valid,
            suffix="_valid",
        )

        # Save generated data
        np.savez(
            f"wgan_gp/results/{self.output_prefix}{self.model_name}_generated_data.npz",
            cl_c=cl_c_array,
            cl_r=cl_r_array,
            gen_coords=gen_coords_array,
        )

        # Save evaluation metrics
        total_convergence_rate = np.mean(convergence_rates)
        total_mse = np.mean([m for m in mse_list if not np.isnan(m)])
        total_diversity_l2 = np.mean([d for d in diversity_l2_list if not np.isnan(d)])
        total_diversity_l1 = np.mean([d for d in diversity_l1_list if not np.isnan(d)])

        print(f"total_convergence_rate: {total_convergence_rate}")
        print(f"total_mse: {total_mse}")
        print(f"total_diversity_l2: {total_diversity_l2}")
        print(f"total_diversity_l1: {total_diversity_l1}")
        print(f"max_min_dist: {max_min_dist}")
        print(f"max_min_dist_valid: {max_min_dist_valid}")

        with open(
            f"wgan_gp/results/{self.output_prefix}{self.model_name}_evaluation_results.csv", "w", newline=""
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Label", "Convergence Rate", "MSE", "Diversity L2", "Diversity L1"])
            for cl_c, cr, mse, div_l2, div_l1 in zip(
                cl_range, convergence_rates, mse_list, diversity_l2_list, diversity_l1_list
            ):
                writer.writerow([cl_c, cr, mse, div_l2, div_l1])

            writer.writerow([])
            writer.writerow(["Overall", total_convergence_rate, total_mse, total_diversity_l2, total_diversity_l1])
            writer.writerow(["v (Max Min Distance)", max_min_dist])
            writer.writerow(["v (Max Min Distance, Valid Only)", max_min_dist_valid])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the generator model file")
    parser.add_argument("--coords_path", type=str, required=True, help="Path to the standardized coordinates npz file")
    parser.add_argument("--perfs_path", type=str, required=True, help="Path to the standardized performance npz file")
    parser.add_argument("--output_prefix", type=str, default="", help="Prefix for output files")
    parser.add_argument("--model_name", type=str, default="wgan_gp", help="Model name for output files")
    parser.add_argument("--samples_per_label", type=int, default=10, help="Number of samples per label")

    args = parser.parse_args()

    coords_npz = np.load(args.coords_path)
    perfs_npz = np.load(args.perfs_path)
    evaluator = Evaluator(args.model_path, coords_npz, perfs_npz, args.output_prefix, args.model_name)

    cl_range = np.round(np.linspace(0.01, 1.50, 150), 2)
    evaluator.generate_and_evaluate(cl_range, args.samples_per_label)

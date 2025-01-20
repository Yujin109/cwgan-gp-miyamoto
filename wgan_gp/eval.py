if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from calc_cl import get_cl, get_cls
from util import save_coords_by_cl, to_cpu, to_cuda
from wgan_gp.models import Generator

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Eval:
    def __init__(self, G_PATH, coords_npz):
        state_dict = torch.load(G_PATH, map_location=torch.device("cpu"), weights_only=True)
        self.G = Generator(3)
        self.G.load_state_dict(state_dict)
        self.G.eval()
        self.latent_dim = 3
        self.coords = {
            "data": coords_npz[coords_npz.files[0]],
            "mean": coords_npz[coords_npz.files[1]],
            "std": coords_npz[coords_npz.files[2]],
        }

    def rev_standardize(self, coords):
        return coords * self.coords["std"] + self.coords["mean"]

    def create_coords_by_cl(self, cl_c, data_num=20):
        z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, self.latent_dim))))
        labels = np.array([cl_c] * data_num)
        labels = Variable(torch.reshape(FloatTensor(np.array([labels])), (data_num, 1)))
        gen_coords = self.rev_standardize(to_cpu(self.G(z, labels)).detach().numpy())
        return gen_coords

    def create_successive_coords(self):
        """0.01から1.50まで151個のC_L^cと翼形状を生成"""
        cl_r = []
        cl_c = []
        gen_coords = []
        for cl in tqdm(range(151)):
            cl /= 100
            cl_c.append(cl)
            labels = Variable(torch.reshape(FloatTensor(np.array([cl])), (1, 1)))
            while True:
                z = Variable(FloatTensor(np.random.normal(0, 1, (1, self.latent_dim))))
                gen_coord = self.rev_standardize(to_cpu(self.G(z, labels)).detach().numpy())
                cl = get_cl(gen_coord)
                # cl = 0.1
                if not np.isnan(cl):
                    cl_r.append(cl)
                    gen_coords.append(gen_coord)
                    break

        np.savez("wgan_gp/results/successive_label", cl_c, cl_r, gen_coords)

    def save_coords(self, gen_coords, labels, path):
        data_size = gen_coords.shape[0]
        fig, ax = plt.subplots(4, min(5, data_size // 4), sharex=True, sharey=True)
        plt.subplots_adjust(hspace=0.6)
        for i in range(min(20, data_size)):
            coord = gen_coords[i]
            label = labels[i]
            x, y = coord.reshape(2, -1)
            ax[i % 4, i // 4].plot(x, y)
            cl = round(label.item(), 4)
            title = "CL={0}".format(str(cl))
            ax[i % 4, i // 4].set_title(title)

        fig.savefig(path)

    def successive(self):
        coords_npz = np.load("wgan_gp/results/successive_label.npz")
        cl_c = coords_npz[coords_npz.files[0]]
        cl_r = coords_npz[coords_npz.files[1]]
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.set_xlim([0, 1.5])
        x = np.linspace(0, 1.5, 10)
        ax.plot(x, x, color="black")
        ax.scatter(cl_c, cl_r)
        ax.set_xlabel("Specified label")
        ax.set_ylabel("Recalculated label")
        # plt.show()
        fig.savefig("wgan_gp/results/successive_label.png")

    def sample_data(self, data_num=100):
        z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, 3))))
        labels = 1.558 * np.random.random_sample(size=(data_num, 1))
        labels = Variable(FloatTensor(labels))
        gen_coords = to_cpu(self.G(z, labels)).detach().numpy()
        labels = to_cpu(labels).detach().numpy()
        np.savez("wgan_gp/results/final", labels, self.rev_standardize(gen_coords))

    def euclid_dist(self, coords):
        """バリエーションがどれぐらいあるか"""
        mean = np.mean(coords, axis=0)
        mu_d = np.linalg.norm(coords - mean) / len(coords)
        return mu_d

    def _dist_from_dataset(self, coord):
        """データセットからの距離の最小値"""
        min_dist = 100
        idx = -1
        for i, data in enumerate(self.rev_standardize(self.coords["data"])):
            dist = np.linalg.norm(coord - data)
            if dist < min_dist:
                min_dist = dist
                idx = i

        return min_dist, idx

    def calc_dist_from_dataset(self, coords, clr):
        data_idx = -1
        generate_idx = -1
        max_dist = 0
        for i, c in enumerate(coords):
            cl = clr[i]
            if not np.isnan(cl):
                dist, didx = self._dist_from_dataset(c)
                if dist > max_dist:
                    max_dist = dist
                    data_idx = didx
                    generate_idx = i
        return max_dist, data_idx, generate_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=0, help="nuber of epoch trained")  # 0
    parser.add_argument("--cl_c", type=float, default=0.789, help="specified cl_c")  # 0.789
    opt = parser.parse_args()

    # データセットの読み込み
    coords_npz = np.load("dataset/standardized_upsampling_coords.npz")
    perfs = np.load("dataset/upsampling_perfs.npy")

    # モデルの読み込み
    G_PATH = f"wgan_gp/results/generator_params_{opt.n_epochs}"
    evl = Eval(G_PATH, coords_npz)

    ### 指定されたCL_cに関する評価 ###

    # 指定されたCL_cの翼形状を生成
    coords = evl.create_coords_by_cl(opt.cl_c)
    coords = coords.reshape(coords.shape[0], -1)

    # 生成された翼形状群内の平均距離を計算
    mu_d = evl.euclid_dist(coords)
    print(f"mu_d:{mu_d}")

    # 生成された翼形状群のCL_rを計算
    clr = get_cls(coords)
    print(f"cl_r:\n{clr}")

    # データセット内の翼形状との距離の最大値を計算
    max_dist, d_idx, g_idx = evl.calc_dist_from_dataset(coords, clr)
    print(f"max_dist:{max_dist}")

    # 生成された翼形状群内の最大距離を持つ翼形状とデータセット内の翼形状を保存
    d_coord = evl.rev_standardize(evl.coords["data"][d_idx])
    d_cl = perfs[d_idx]
    g_coord = coords[g_idx]
    g_cl = clr[g_idx]

    # print(opt.cl_c, d_cl, g_cl)
    print(f"Input CL_c:{opt.cl_c} → MAX_DISTANCE_PAIR: CL_r of Raw Data:{d_cl}, CL_r of Generated Data:{g_cl}")

    cls = np.array([opt.cl_c, d_cl, g_cl])
    np.savez("dist_{0}".format(opt.cl_c), d_coord, g_coord, cls, max_dist)

    #########################################

    evl.create_successive_coords()
    evl.successive()

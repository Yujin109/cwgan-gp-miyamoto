if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def to_cuda(tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor


def to_cpu(tensor):
    return tensor.cpu() if torch.cuda.is_available() else tensor


def postprocess(X):  # TODO: リファクタ
    X = np.squeeze(X)
    return X


def save_coords_by_cl(gen_coords, cl_c, path):  # TODO: リファクタ
    from calc_cl import get_cl

    data_size = gen_coords.shape[0]
    fig, ax = plt.subplots(4, min(5, data_size // 4), sharex=True, sharey=True)
    fig.suptitle("CL={0}".format(cl_c))
    plt.subplots_adjust(hspace=0.6)
    for i in range(min(20, data_size)):
        coord = gen_coords[i]
        cl_r = get_cl(coord)
        x, y = coord.reshape(2, -1)
        if not np.isnan(cl_r):
            cl = round(cl_r, 4)
            title = str(cl)
            ax[i % 4, i // 4].set_title(title)
            ax[i % 4, i // 4].plot(x, y)
        else:
            title = "nan"
            ax[i % 4, i // 4].plot(x, y, color="r")
            ax[i % 4, i // 4].set_title(title)

    # plt.show()
    fig.savefig(path)


def save_coords(gen_coords, labels, path, rows=4, cols=5):
    try:
        fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(cols * 3, rows * 3))
        fig.subplots_adjust(hspace=0.6)
        for i in range(min(rows * cols, len(gen_coords))):
            coord = gen_coords[i]
            label = labels[i].item()
            x, y = coord.reshape(2, -1)
            ax[i // cols, i % cols].plot(x, y)
            ax[i // cols, i % cols].set_title(f"CL={label:.4f}")
        plt.savefig(path)
        plt.close(fig)
    except Exception as e:
        print(f"Error saving coordinates: {e}")


def save_loss(losses, path="results/loss.png"):
    try:
        plt.figure(figsize=(10, 5))
        plt.title("MSE Loss During Training")
        plt.plot(losses, label="MSE Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Error saving loss plot: {e}")

import torch
import torch.nn as nn

coord_shape = (1, 496)  # TODO: 汎用性のある形に変更（モデルはflatで不可変だが、データセットの保存時の形式が複数ある）


def create_block(in_features, out_features, normalize=True, dropout=False):
    layers = [nn.Linear(in_features, out_features)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_features, 0.8))
    if dropout:
        layers.append(nn.Dropout(0.4))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim=496):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            *create_block(latent_dim + 1, 64, normalize=False),
            *create_block(64, 128),
            *create_block(128, 256),
            *create_block(256, 512),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((labels, noise), dim=-1)
        coords = self.model(gen_input)
        coords = coords.view(coords.shape[0], *coord_shape)
        return coords


class Discriminator(nn.Module):
    def __init__(self, input_dim=496):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            *create_block(input_dim + 1, 512, normalize=False, dropout=False),
            *create_block(512, 256, normalize=False, dropout=False),
            nn.Linear(256, 1)
        )

    def forward(self, coords, labels):
        x = torch.cat((coords.view(coords.size(0), -1), labels), dim=-1)
        x = x.view(x.shape[0], -1)
        return self.model(x)

import math

import torch
import torch.nn as nn
from tqdm import tqdm


def pos_encoding(timesteps, output_dim, device="cpu"):
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        t = timesteps[i]
        for dim in range(output_dim):
            if dim % 2 == 0:
                v[i, dim] = math.sin(t / (10000 ** (dim / output_dim)))
            else:
                v[i, dim] = math.cos(t / (10000 ** ((dim - 1) / output_dim)))
    return v


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super(UNetConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu3 = nn.LeakyReLU(negative_slope=0.01)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_channels), nn.ReLU(), nn.Linear(in_channels, in_channels)
        )

    def forward(self, x, t_emb):
        # t_emb = self.time_mlp(t_emb).unsqueeze(-1)
        t_emb = self.time_mlp(t_emb).unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = self.conv1(x + t_emb)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return self.pool(x), x  # Downsampled output, skip connection


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_embed_dim):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_channels), nn.ReLU(), nn.Linear(in_channels, in_channels)
        )

    def forward(self, x, skip_connection, t_emb):
        # t_emb = self.time_mlp(t_emb).unsqueeze(-1)
        x = self.upconv(x)
        t_emb = self.time_mlp(t_emb).unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv1(x + t_emb)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class UNetCond(nn.Module):
    def __init__(self, in_ch=2, time_embed_dim=100, label_dim=1):
        """
        Args:
            in_ch (int): 入力チャネル数
            time_embed_dim (int): 時間埋め込みの次元数
            label_dim (int): ラベル入力の次元数（デフォルトは1）
        """
        super(UNetCond, self).__init__()
        self.time_embed_dim = time_embed_dim

        # Encoder
        self.encoder1 = UNetConvBlock(in_ch, 64, time_embed_dim)
        self.encoder2 = UNetConvBlock(64, 128, time_embed_dim)
        self.encoder3 = UNetConvBlock(128, 256, time_embed_dim)
        self.encoder4 = UNetConvBlock(256, 512, time_embed_dim)
        # self.encoder5 = UNetConvBlock(512, 1024, time_embed_dim)

        # Decoder
        # self.decoder4 = DecoderBlock(1024, 512, 512, time_embed_dim)
        self.decoder3 = DecoderBlock(512, 256, 256, time_embed_dim)
        self.decoder2 = DecoderBlock(256, 128, 128, time_embed_dim)
        self.decoder1 = DecoderBlock(128, 64, 64, time_embed_dim)

        # Output layer
        self.final_conv = nn.Conv1d(64, in_ch, kernel_size=1)

        # Label embedding using MLP for continuous labels with variable input dimension
        self.label_mlp = nn.Sequential(
            nn.Linear(label_dim, time_embed_dim),  # label_dim を可変に
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, x, timesteps, labels=None):
        t_emb = pos_encoding(timesteps, self.time_embed_dim, x.device)

        if labels is not None:
            labels = labels.view(labels.size(0), -1).float()  # Ensure labels have shape (batch_size, label_dim)
            label_emb = self.label_mlp(labels)
            t_emb += label_emb

        # Encoding
        x1, skip1 = self.encoder1(x, t_emb)
        x2, skip2 = self.encoder2(x1, t_emb)
        x3, skip3 = self.encoder3(x2, t_emb)
        x4, skip4 = self.encoder4(x3, t_emb)  # skip4:[64,512,31]
        # x5, skip5 = self.encoder5(x4, t_emb)  # x5:[64,512,14], skip5:[64,1024,7]

        # Decoding
        # x = self.decoder4(skip5, skip4, t_emb)
        x = self.decoder3(skip4, skip3, t_emb)
        x = self.decoder2(x, skip2, t_emb)
        x = self.decoder1(x, skip1, t_emb)

        # Output
        return self.final_conv(x)


class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alpha_bars[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx]  # (N,)
        alpha_bar = alpha_bar.view(alpha_bar.size(0), 1, 1)  # (N, 1, 1)
        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t, labels):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alphas[0] is for t=1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx - 1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t, labels)  # add lable embedding
        model.train()
        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0  # no noise at t=1

        mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        return mu + noise * std

    def generate_from_labels(
        self,
        model,
        labels,
        coord_shape=(2, 248),
    ):
        batch_size = len(labels)
        x = torch.randn((batch_size, coord_shape[0], coord_shape[1]), device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t, labels)
        return x

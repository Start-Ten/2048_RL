"""DQN network modules: SE-ResNet backbone, NoisyLinear, DQN_V4."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedNoisyLinear(nn.Module):
    """Factorized Gaussian Noise linear layer (Rainbow DQN)."""

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        eps_in = self._f(torch.randn(self.in_features, device=self.weight_mu.device))
        eps_out = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    @staticmethod
    def _f(x): return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation residual block."""

    def __init__(self, in_c, out_c, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(out_c, max(1, out_c // reduction)), nn.ReLU(inplace=True),
            nn.Linear(max(1, out_c // reduction), out_c), nn.Sigmoid())
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        r = self.shortcut(x)
        o = F.relu(self.bn1(self.conv1(x)), inplace=True)
        o = self.bn2(self.conv2(o))
        o = o * self.se(o).view(o.size(0), -1, 1, 1) + r
        return F.relu(o, inplace=True)


class DQN_V4(nn.Module):
    """SE-ResNet + Dueling + NoisyNet (~6.5M params — balanced for 2048)."""

    def __init__(self, input_channels=8, output_size=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.s1 = nn.Sequential(SEBlock(64, 128), SEBlock(128, 128))
        self.s2 = nn.Sequential(SEBlock(128, 256), SEBlock(256, 256), SEBlock(256, 256))
        self.s3 = nn.Sequential(SEBlock(256, 256), SEBlock(256, 256))
        self.val_conv = nn.Conv2d(256, 16, 1)
        self.val_fc = nn.Sequential(nn.ReLU(inplace=True), nn.Flatten(),
                                     nn.Linear(256, 128), nn.ReLU(inplace=True))
        self.val_noisy = FactorizedNoisyLinear(128, 1)
        self.adv_conv = nn.Conv2d(256, 64, 1)
        self.adv_fc = nn.Sequential(nn.ReLU(inplace=True), nn.Flatten(),
                                     nn.Linear(1024, 128), nn.ReLU(inplace=True))
        self.adv_noisy = FactorizedNoisyLinear(128, output_size)

    def forward(self, x):
        x = self.stem(x); x = self.s1(x); x = self.s2(x); x = self.s3(x)
        v = self.val_noisy(self.val_fc(self.val_conv(x)))
        a = self.adv_noisy(self.adv_fc(self.adv_conv(x)))
        return v + a - a.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, FactorizedNoisyLinear): m.reset_noise()

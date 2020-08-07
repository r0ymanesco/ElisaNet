import torch
import torch.nn as nn
import numpy as np
import ipdb


class ElisaNet(nn.Module):
    def __init__(self, c_feat):
        super(ElisaNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=c_feat,
                kernel_size=9,
                stride=4,
                padding=4),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=c_feat,
                out_channels=c_feat,
                kernel_size=5,
                stride=2,
                padding=2),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=c_feat,
                out_channels=c_feat,
                kernel_size=5,
                stride=2,
                padding=2),

            nn.ReLU(),

            nn.Conv2d(
               in_channels=c_feat,
               out_channels=c_feat,
               kernel_size=5,
               stride=2,
               padding=2),

            nn.ReLU(),

            nn.Conv2d(
               in_channels=c_feat,
               out_channels=c_feat,
               kernel_size=5,
               stride=2,
               padding=2),
        )

        self.mlp_layers = nn.Sequential(
            nn.Linear(
                in_features=c_feat*43*43,
                out_features=3000
            ),

            nn.ReLU(),

            nn.Linear(
                in_features=3000,
                out_features=300
            ),

            nn.ReLU(),

            nn.Linear(
                in_features=300,
                out_features=30
            ),

            nn.ReLU(),

            nn.Linear(
                in_features=30,
                out_features=1
            ),
        )

    def forward(self, x):
        # modules = [module for k, module in self.layers._modules.items()]
        # out = checkpoint.checkpoint_sequential(modules, 2, x)
        conv_out = self.conv_layers(x)
        out = self.mlp_layers(conv_out.view(x.size(0), -1))
        return out


import torch.nn as nn
import torch
import random
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, args, arch='256_256'):
        super(MLP, self).__init__()
        self.args = args
        if arch.startswith('mlp'):
            arch = arch.replace('mlp', '')

        # ====== backbone ======
        module_list = []
        for i, hidden_size in enumerate(arch.split('_')):
            hidden_size = int(hidden_size)

            module_list.append(nn.Sequential(
                nn.Linear(in_dim, hidden_size),
                nn.BatchNorm1d(hidden_size, affine=True),
                nn.ReLU(),
            ))
            in_dim = hidden_size

        self.backbone = nn.Sequential(*module_list)

        self.fc = nn.Linear(in_dim, out_dim, bias=args.bias=='t')

    def forward(self, state, ret='o'):
        feat = self.backbone(state)
        out = self.fc(feat)
        if ret == 'of':
            return out, feat
        else:
            return out
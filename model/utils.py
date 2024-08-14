import torch
import torch.nn as nn
import numpy as np


def get_centroids(feats_, labels_):
    centroids = []
    for i in np.unique(labels_):
        centroids.append(np.mean(feats_[labels_ == i], axis=0))
    return np.stack(centroids)


class BalancedBatchNorm2d(nn.Module):
    # num_features: the number of output channels for a convolutional layer.
    def __init__(self, num_features, affine=True):
        super().__init__()
        shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(shape))
            self.beta = nn.Parameter(torch.zeros(shape))
        else:
            self.gamma = torch.ones(shape)
            self.beta = torch.zeros(shape)
        # moving_mean and moving_var are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X, label):
        # If X is not on the main memory, copy moving_mean and moving_var to the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        if self.gamma.device != X.device:
            self.gamma = self.gamma.to(X.device)
            self.beta = self.beta.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, label, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-6, momentum=0.1
        )
        return Y


def batch_norm(X, label, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional conv layer, calculate mean and variance on channel dimension (axis=1).
            batch_size, C, H, W = X.shape
            sum_ = torch.zeros((torch.max(label).item()+1, C, H, W), dtype=X.dtype).to(X.device)   # [B, C, H, W], sum over Batch
            sum_.index_add_(dim=0, index=label, source=X)    # [K, C, H, W]
            cnt_ = torch.bincount(label)
            avg_feat = sum_[cnt_>0]/cnt_[cnt_>0][:, None, None, None]    # [K, C, H, W]  class-wise mean feat
            mean = avg_feat.mean(dim=(0, 2, 3), keepdim=True)            # channel mean (equal weight for all classes)

            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data


def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam


def to_one_hot(inp, num_classes):
    if inp is not None:
        y_onehot = torch.FloatTensor(inp.size(0), num_classes)
        y_onehot.zero_()
        y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
        return torch.autograd.Variable(y_onehot.cuda(), requires_grad=False)
    else:
        return None


def mixup_process(out, target, lam):
    indices = np.random.permutation(out.size(0))
    out = out * lam + out[indices] * (1 - lam)
    target = target * lam + target[indices] * (1 - lam)
    return out, target

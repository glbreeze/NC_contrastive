import torch
mean0 = 0.0
mean2 = 5.0
std = 1.0  # Standard deviation

label = torch.tensor([0, 2, 0, 2])

input = torch.concatenate((torch.normal(mean0, std, (1, 3)),
                           torch.normal(mean2, std, (1, 3)),
                           torch.normal(mean0, std, (1, 3)),
                           torch.normal(mean2, std, (1, 3)),
), dim=0)


n_samples=20;  # sample per sub-cls
n_features=2  # int, default=2 The number of features for each sample.
center_box=(-10.0, 10.0)  # generate cluster centers within box
cls_dist=[5, 6]
sub_cls_dist=[2,4]
cls_std=2.0
sub_cls_std=1.0
n_cls=2
n_sub_cls=2
shuffle=True
random_state=None

import math
import numpy as np
def get_vol(d, r):
    vol = r**d * (np.pi)**(d/2) / math.gamma(d/2+1)
    return vol

from matplotlib import pyplot as plt
plt.plot(np.arange(1, 10), [get_vol(d, 1) for d in np.arange(1, 10)])

d = 10
v = (get_vol(d, 1) - get_vol(d, 1-0.05))/get_vol(d, 1)

print(f'{d}: {v}')


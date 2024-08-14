import os
import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt

# Plot for w_cos

def plot_nc(nc_dt):
    k_lst = ['w_norm', 'h_norm', 'w_cos', 'h_cos', 'wh_cos', 'nc1_cls']

    # print(nc_dt['nc1_cls'])
    fig, axes = plt.subplots(nrows=3, ncols=2)
    k = 0
    for key in k_lst:
        cos_matrix = nc_dt[key]  # [K, K]

        if key in ['w_cos', 'h_cos', 'wh_cos']:
            im = axes[int(k//2), int(k %2)].imshow(cos_matrix, cmap='RdBu')
            plt.colorbar(im, ax=axes[int(k//2), int(k %2)])
            im.set_clim(vmin=-1, vmax=1)
            axes[int(k//2), int(k %2)].set_title(key)
        else:
            axes[int( k//2), int( k %2)].bar(np.arange(len(cos_matrix)), cos_matrix)
            axes[int( k//2), int( k %2)].set_title(key)

        k += 1

    return fig

import os
import pickle
import argparse
import torch
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils.measure_nc import analysis_feat

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from test_dino import get_args_parser, test_dino

parser = argparse.ArgumentParser('DINO_analysis', parents=[get_args_parser()])
args = parser.parse_args([])

args.output_dir = "result/dino/cifar100/mres32_s15_dim512_lr1e-3"

all_feats = {}
all_labels = {}
logit_un = {}
for k in range(0, 480+1, 20):
    ckpt_file = f'checkpoint{k:04}.pth'
    all_feats[k], all_labels[k], logit_un[k] = test_dino(args, ckpt_file=ckpt_file, eval_cls=False)

with open(os.path.join(args.output_dir, 'analysis.pkl'), 'wb') as f:
    pickle.dump([all_feats, all_labels, logit_un], f)
print(f'save the result to pickle file')

if False:
    pred_un = logit_un.argmax(axis=-1)
    plt.hist(pred_un, bins=512, alpha=0.7)
    print(f"number of virtual classes with occurrence {len(np.unique(pred_un))}")
    virtual_cls, virtual_cnt = np.unique(pred_un, return_counts=True)


    unique_pred_un = np.unique(pred_un)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_pred_un)}
    remapped_pred_un = np.array([label_mapping[label] for label in pred_un])
    unique_labels, counts = np.unique(remapped_pred_un, return_counts=True)
    plt.bar(unique_labels, counts)


    nc_dict = analysis_feat(torch.tensor(remapped_pred_un), torch.tensor(feats).float(), num_classes=len(unique_pred_un), W=None)


    cm = confusion_matrix(labels, pred_un)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

import os
import pickle
import torch
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils.measure_nc import analysis_feat

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


output_dir = "result/dino/cf100_cp1.5_dim512"
fpath = os.path.join(output_dir,'analysis.pkl')

with open(fpath, 'rb') as f:
    feats, labels, logit_sp, logit_un = pickle.load(f)

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
import os.path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pickle
from PIL import Image

from torchvision import transforms


class Cifar100Imbalance(Dataset):
    def __init__(self, imbalance_rate=0.1, imbalance_type='exp', file_path="data/cifar-100-python/",
                 num_cls=100, transform=None, train=True):
        self.transform = transform
        assert 0.0 < imbalance_rate <= 1, "imbalance_rate must 0.0 < p <= 1"
        self.num_cls = num_cls
        self.file_path = file_path
        self.imbalance_rate = imbalance_rate
        self.imbalance_type = imbalance_type

        if train is True:
            self.data = self.produce_imbalance_data(self.imbalance_rate)
        else:
            self.data = self.produce_test_data()
        self.x = self.data['x']
        self.y = self.data['y'] if isinstance(self.data['y'], list) else self.data['y'].tolist()
        self.targets = self.data['y'] if isinstance(self.data['y'], list) else self.data['y'].tolist()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        x = Image.fromarray(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def get_per_class_num(self):
        return self.per_class_num

    def produce_test_data(self):
        with open(os.path.join(self.file_path,"test"), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x_test = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            y_test = dict[b'fine_labels']
        dataset = {
            "x": x_test,
            "y": y_test,
        }
        return dataset

    def get_img_num_per_cls(self, img_max, imbalance_type='step', imbalance_rate=1.0):
        img_num_per_cls = []
        if imbalance_type == 'exp':
            for cls_idx in range(self.num_cls):
                num = img_max * (imbalance_rate ** (cls_idx / (self.num_cls - 1.0)))
                img_num_per_cls.append(int(num))
        elif imbalance_type == 'step':
            for cls_idx in range(self.num_cls // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(self.num_cls // 2):
                img_num_per_cls.append(int(img_max * imbalance_rate))
        else:
            img_num_per_cls.extend([int(img_max)] * self.num_cls)
        return img_num_per_cls

    def produce_imbalance_data(self, imbalance_rate):

        with open(os.path.join(self.file_path,"train"), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x_train = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            y_train = dict[b'fine_labels']

        y_train = np.array(y_train)
        data_x = None
        data_y = None

        data_num = int(x_train.shape[0] / self.num_cls)
        data_percent = self.get_img_num_per_cls(img_max=data_num, imbalance_type=self.imbalance_type, imbalance_rate=imbalance_rate)

        self.per_class_num = data_percent
        print("imbalance ratio is {}".format(data_percent[0] / data_percent[-1]))
        print("per class num：{}".format(data_percent))

        for i in range(1, self.num_cls + 1):
            a1 = y_train >= i - 1
            a2 = y_train < i
            index = a1 & a2

            task_train_x = x_train[index]
            label = y_train[index]
            data_num = task_train_x.shape[0]
            index = np.random.choice(data_num, data_percent[i - 1],replace=False)
            tem_data = task_train_x[index]
            tem_label = label[index]

            if data_x is None:
                data_x = tem_data
                data_y = tem_label
            else:
                data_x = np.concatenate([data_x, tem_data], axis=0)
                data_y = np.concatenate([data_y, tem_label], axis=0)

        dataset = {
            "x": data_x,
            "y": data_y,
        }

        return dataset

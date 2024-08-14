import torchvision
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

class Cifar10Imbalance(Dataset):
    def __init__(self, imbalance_rate, imbalance_type='exp', num_cls=10, file_path="data/",
                 train=True, transform=None, label_align=True, ):
        self.transform = transform
        self.label_align = label_align
        assert 0.0 < imbalance_rate <= 1, "imbalance_rate must 0.0 < imbalance_rate <= 1"
        self.imbalance_rate = imbalance_rate
        self.imbalance_type = imbalance_type

        self.num_cls = num_cls
        self.data = self.produce_imbalance_data(file_path=file_path, train=train, imbalance_rate=self.imbalance_rate)
        self.x = self.data['x']
        self.targets = self.data['y'].tolist()
        self.y = self.data['y'].tolist()

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

    def produce_imbalance_data(self, imbalance_rate, file_path="/data", train=True):

        train_data = torchvision.datasets.CIFAR10(root=file_path, train=train, download=True,)
        x_train = train_data.data
        y_train = train_data.targets
        y_train = np.array(y_train)

        data_num = int(x_train.shape[0] / self.num_cls)
        if train:
            data_percent = self.get_img_num_per_cls(img_max=data_num, imbalance_rate=imbalance_rate, imbalance_type=self.imbalance_type)
        else:
            data_percent = [int(data_num)] * self.num_cls

        self.per_class_num = data_percent
        if train:
            print("imbalance_ration is {}".format(data_percent[0] / data_percent[-1]))
            print("per class num: {}".format(data_percent))

        rehearsal_data = None
        rehearsal_label = None
        for i in range(1, self.num_cls + 1):
            index = (y_train >= i - 1) & (y_train < i)
            task_train_x = x_train[index]
            label = y_train[index]

            data_num = task_train_x.shape[0]
            index = np.random.choice(data_num, data_percent[i - 1],replace=False)   # chosen index for class i
            tem_data = task_train_x[index]
            tem_label = label[index]
            if rehearsal_data is None:
                rehearsal_data = tem_data
                rehearsal_label = tem_label
            else:
                rehearsal_data = np.concatenate([rehearsal_data, tem_data], axis=0)
                rehearsal_label = np.concatenate([rehearsal_label, tem_label], axis=0)

        task_split = {
            "x": rehearsal_data,
            "y": rehearsal_label,
        }
        # split the data into majority class and minority class

        return task_split

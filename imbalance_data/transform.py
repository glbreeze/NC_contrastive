import shutil
from torch.utils import data
import PIL
import math
import warnings
from imbalance_data.cifar100Imbanlance import *
from imbalance_data.cifar10Imbanlance import *
from imbalance_data.dataset_lt_data import *
import utils.moco_loader as moco_loader


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def _is_numpy_image(img):
    return img.ndim in {2, 3}


def cutout(img, i, j, h, w, v, inplace=False):
    """ Erase the CV Image with given value.
    Args:
        img (Tensor Image): Tensor image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v: Erasing value.
        inplace(bool, optional): For in-place operations. By default is set False.
    Returns:
        CV Image: Cutout image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    if not inplace:
        img = img.copy()

    img[i:i + h, j:j + w, :] = v
    return img


# Define Cutout transform
class Cutout(object):
    """Random erase the given CV Image.

    Arguments:
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        scale: range of proportion of erased area against input image.
        ratio: range of aspect ratio of erased area.
        pixel_level (bool): filling one number or not. Default value is False
    """
    def __init__(self, p=0.5, scale=(0.02, 0.4), ratio=(0.4, 1 / 0.4), value=(0, 255), pixel_level=False, inplace=False):

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.pixel_level = pixel_level
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio):
        if type(img) == np.ndarray:
            img_h, img_w, img_c = img.shape
        else:
            img_h, img_w = img.size
            img_c = len(img.getbands())

        s = random.uniform(*scale)
        # if you img_h != img_w you may need this.
        # r_1 = max(r_1, (img_h*s)/img_w)
        # r_2 = min(r_2, img_h / (img_w*s))
        r = random.uniform(*ratio)
        s = s * img_h * img_w
        w = int(math.sqrt(s / r))
        h = int(math.sqrt(s * r))
        left = random.randint(0, img_w - w)
        top = random.randint(0, img_h - h)

        return left, top, h, w, img_c

    def __call__(self, img):
        if random.random() < self.p:
            left, top, h, w, ch = self.get_params(img, self.scale, self.ratio)

            if self.pixel_level:
                c = np.random.randint(*self.value, size=(h, w, ch), dtype='uint8')
            else:
                c = random.randint(*self.value)

            if type(img) == np.ndarray:
                return cutout(img, top, left, h, w, c, self.inplace)
            else:
                if self.pixel_level:
                    c = PIL.Image.fromarray(c)
                img.paste(c, (left, top, left + w, top + h))
                return img
        return img


def get_transform(dataset, aug=None):
    # cifar10/cifar100: 32x32, stl10: 96x96, fmnist: 28x28, TinyImageNet 64x64

    if dataset == "cifar10" or dataset == "cifar100":
        if dataset == "cifar10":
            mean = (0.49139968, 0.48215827, 0.44653124)
            std = (0.24703233, 0.24348505, 0.26158768)
        elif dataset == "cifar100":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        if (aug is None) or aug == 'null':
            transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        elif aug == 'flip' or aug == 'cutmix' or aug=='cm':
            transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
        elif aug == 'pc':  # padded crop
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),])
        elif aug == 'rs':  # resized crop
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std), ])
        elif aug == 'cj':  # color jittering
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std), ])
        elif aug == 'co':  # cutout
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                Cutout(p=0.75, scale=(0.02, 0.4), ratio=(0.4, 1 / 0.4), value=(0, 255), pixel_level=False),
                transforms.ToTensor(),
                transforms.Normalize(mean, std), ])
        elif aug == 'comp':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        transform_val = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
        return transform_train, transform_val

    if dataset == 'stl10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2434, 0.2615])
        transform_train = transforms.Compose([# transforms.RandomCrop(96, padding=4), # for stl10
                                              transforms.ToTensor(),
                                              normalize])
        transform_val = transforms.Compose([transforms.ToTensor(), normalize])
        return transform_train, transform_val

    elif dataset == "fmnist":
        fashion_mnist = torchvision.datasets.FashionMNIST(download=True, train=True, root="data").train_data.float()
        transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((fashion_mnist.mean() / 255,), (fashion_mnist.std() / 255,))
                                            ])
        transform_val = transforms.Compose([transforms.Resize((32, 32)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((fashion_mnist.mean() / 255,), (fashion_mnist.std() / 255,))
                                            ])
        return transform_train, transform_val

    elif dataset == 'tinyi':  # image_size:64 x 64
        if aug is None or aug == 'null':
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                  ])
            transform_val = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                               ])
        return transform_train, transform_val

    elif dataset == "ImageNet-LT":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45),img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

        transform_train = transforms.Compose(augmentation_sim)

        return transform_train, transform_val

    elif dataset == "iNaturelist2018":
        normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )

        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

        transform_train = transforms.Compose(augmentation_sim)

        return transform_train, transform_val
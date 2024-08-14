import shutil
from torch.utils import data
import copy
import os
from imbalance_data.cifar100Imbanlance import *
from imbalance_data.cifar10Imbanlance import *
from imbalance_data.dataset_lt_data import *
import utils.moco_loader as moco_loader
from utils.randaugment import rand_augment_transform


class Graph_Vars:
    def __init__(self):
        self.epoch = []
        self.acc = []
        self.loss = []

        self.nc1 = []

        self.nc2_h = []
        self.nc2_w =[]
        self.nc21_h = []
        self.nc22_h = []
        self.nc21_w = []
        self.nc22_w = []
        
        self.nc1_cls = []
        self.w_norm = []
        self.h_norm = []
        self.w_mnorm = []
        self.h_mnorm = []
        self.w_cos = []
        self.w_cos_avg = []
        self.h_cos = []
        self.h_cos_avg = []
        self.wh_cos = []

        self.nc3 = []
        self.nc3_d = []
        
        self.w_mnorm1 = []
        self.w_mnorm2 = []
        self.w_mnorm3 = []
        self.h_mnorm1 = []
        self.h_mnorm2 = []
        self.h_mnorm3 = []
        self.w_cos1 = []
        self.w_cos2 = []
        self.w_cos3 = []
        self.h_cos1 = []
        self.h_cos2 = []
        self.h_cos3 = []
  
        self.lr = []
        self.test_acc = []

    def load_dt(self, nc_dt, epoch, lr=None):
        self.epoch.append(epoch)
        if lr:
            self.lr.append(lr)
        for key in nc_dt:
            try:
                self.__getattribute__(key).append(nc_dt[key])
            except:
                print('{} is not attribute of Graph var'.format(key))

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.ceil(W * cut_rat).astype(int)
    cut_h = np.ceil(H * cut_rat).astype(int)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_transform(dataset):
    if dataset == "cifar10":
        mean = (0.49139968, 0.48215827, 0.44653124)
        std = (0.24703233, 0.24348505, 0.26158768)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return transform_train,transform_val

    if dataset == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return transform_train, transform_val

    if dataset == "ImageNet-LT":
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

    if dataset == "iNaturelist2018":
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

def prepare_folders(args):
    folder = os.path.join(args.root_model, args.store_name)
    if not os.path.exists(folder):
        print('creating folder ' + folder)
        os.makedirs(folder)

def save_checkpoint(args, state, is_best, epoch):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
    # if epoch % 20 == 0:
    #     filename = '%s/%s/%s_ckpt.pth.tar' % (args.root_model, args.store_name, str(epoch))
    #     torch.save(state, filename)

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def GLMC_mixed(org1, org2, invs1, invs2, label_org, label_invs, label_org_w, label_invs_w, alpha=1):
    lam = np.random.beta(alpha, alpha)

    # mixup
    mixup_x = lam * org1 + (1 - lam) * invs1
    mixup_y = lam * label_org + (1 - lam) * label_invs
    mixup_y_w = lam * label_org_w + (1 - lam) * label_invs_w

    # cutmix
    bbx1, bby1, bbx2, bby2 = rand_bbox(org2.size(), lam)
    org2[:, :, bbx1:bbx2, bby1:bby2] = invs2[:, :, bbx1:bbx2, bby1:bby2]

    lam_cutmix = lam
    cutmix_y = lam_cutmix * label_org + (1 - lam_cutmix) * label_invs
    cutmix_y_w = lam_cutmix * label_org_w + (1 - lam_cutmix) * label_invs_w

    return mixup_x, org2, mixup_y, cutmix_y, mixup_y_w, cutmix_y_w

def print_model_param_nums(model=None):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    return total
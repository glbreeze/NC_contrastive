
import argparse
import os
import wandb
import sys
import pickle
import math
import json
import torchvision
from pathlib import Path
from model.ResNet_new import mresnet32

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import dino_utils
import vision_transformer as vits
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='mresnet32', type=str)
        # choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] + torchvision_archs + torch.hub.list("facebookresearch/xcit:main")
    parser.add_argument('--patch_size', default=16, type=int, )
    parser.add_argument('--out_dim', default=512, type=int, help="""Dimensionality of the DINO head output. """)
    parser.add_argument('--norm_last_layer', default=True, type=dino_utils.bool_flag, help="""Whether or not to weight normalize """)
    parser.add_argument('--use_bn_in_head', default=False, type=dino_utils.bool_flag, help="Whether to use BN in projection head (Default: False)")

    # fine-tune
    parser.add_argument('--batch_size_per_gpu', default=1024, type=int, help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--cls_epochs', type=int, default=10)
    parser.add_argument('--cls_lr', type=float, default=5e-2)
    parser.add_argument('--cls_weight_decay', type=float, default=1e-4)

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,)
    parser.add_argument('--output_dir', default="result/dino/cf100_cp1.5_dim512", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers per GPU.')
    return parser


def test_dino(args):
    dino_utils.fix_random_seeds(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["WANDB_API_KEY"] = "0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee"
    os.environ["WANDB_MODE"] = "online"  # "dryrun"
    os.environ["WANDB_CACHE_DIR"] = "/scratch/lg154/sseg/.cache/wandb"
    os.environ["WANDB_CONFIG_DIR"] = "/scratch/lg154/sseg/.config/wandb"
    wandb.login(key='0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee')
    wandb.init(project="NC_cls", name=args.output_dir.split('/')[-1])
    wandb.config.update(args)

    # ============ preparing data ... ============
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR100(root='../dataset', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='../dataset', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, sampler=None, batch_size=args.batch_size_per_gpu, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_per_gpu, shuffle=False,
                                              num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    print(f"Data loaded: there are {len(trainset)} images.")

    # ============ building student and teacher networks ... ============
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    elif args.arch == 'mresnet32':
        student = mresnet32(num_classes=100)
        embed_dim = student.feat_dim
    else:
        print(f"Unknow architecture: {args.arch}")
    student.fc, student.head = nn.Identity(), nn.Identity()

    dino_head = DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer,)

    student, dino_head = student.to(device), dino_head.to(device)
    ckpt_path = os.path.join(args.output_dir, 'checkpoint.pth')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint['student']
    else:
        print("Can not find checkpoint at {}".format(ckpt_path))
    student.load_state_dict({key.replace('backbone.', '') : v for key, v in state_dict.items() if 'backbone' in key})
    dino_head.load_state_dict({key.replace('head.', '') : v for key, v in state_dict.items() if 'head' in key})

    # ================= train classifier head  =================
    classifer = nn.Linear(student.feat_dim, args.num_classes, bias=True)
    classifer = classifer.to(device)
    classifer = train_classifier(student, classifer, train_loader, total_epochs=args.cls_epochs)
    # test_acc = evaluate_backbone(student.backbone, classifer, test_loader)

    # ================= train data  =================
    student.eval()
    classifer.eval()
    dino_head.eval()
    all_feats, all_labels, logit_sp, logit_un = [], [], [], []
    for it, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            feats = student(images)
            logit_sp_ = classifer(feats)
            logit_un_ = dino_head(feats)
        all_feats.append(feats)
        all_labels.append(labels)
        logit_sp.append(logit_sp_)
        logit_un.append(logit_un_)
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels)
    logit_sp = torch.cat(logit_sp, dim=0)
    logit_un = torch.cat(logit_un, dim=0)

    with open(os.path.join(args.output_dir, 'analysis.pkl'), 'wb') as f:
        pickle.dump([all_feats, all_labels, logit_sp, logit_un], f)
    print(f'save the result to pickle file')


def train_classifier(backbone, classifer, train_loader, total_epochs=1):
    backbone.eval()
    classifer.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifer.parameters(), lr=args.cls_lr, momentum=0.9, weight_decay=args.cls_weight_decay)
    for epoch in range(total_epochs):
        for it, (images, labels) in enumerate(train_loader):
            prog = (epoch * len(train_loader) + it) / (total_epochs * len(train_loader))
            optimizer.param_groups[0]['lr'] = args.cls_lr * 0.2**(prog//0.333) 
            
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            with torch.no_grad():
                feats = backbone(images)
            logits = classifer(feats)
            loss = criterion(logits, labels)
            train_acc = (logits.argmax(dim=-1) == labels).float().mean().item()

            # ==== gradient update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # ==== print
            if it%10 == 0 :
                print(f'----classifier acc:{train_acc:.3f}, lr:{optimizer.param_groups[0]["lr"]:.4f}')
    return classifer


def evaluate_backbone(backbone, classifer, test_loader):
    backbone.eval()
    classifer.eval()
    all_labels, all_preds = [], []
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.no_grad():
            logits = classifer(backbone(images))
        all_preds.append(logits.argmax(dim=-1))
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    acc = torch.sum(all_preds == all_labels).item()/len(all_labels)
    return acc


def train_one_epoch(student, teacher, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = dino_utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = dino_utils.clip_gradients(student, args.clip_grad)
            dino_utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = dino_utils.clip_gradients(student, args.clip_grad)
            dino_utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    # logging
    metric = dict(loss=loss.item(), lr=optimizer.param_groups[0]["lr"], wd=optimizer.param_groups[0]["weight_decay"])
    print(", ".join(f"{k}: {v:.4f}" for k, v in metric.items()))
    return metric


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(0.1),
            dino_utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args([])
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(args)
    test_dino(args)

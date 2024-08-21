from imbalance_data.cifar100_coarse2fine import fine_id_coarse_id
import wandb
import torch.nn as nn
from torchvision.transforms import v2

from utils.util import *
from utils.plot import plot_nc
from utils.measure_nc import analysis
from model.utils import get_centroids
from model.classifier import NCC_Classifier, LinearClassifier
from model.loss import CrossEntropyLabelSmooth, SupConLoss


def _get_polynomial_decay(lr, end_lr, decay_epochs, from_epoch=0, power=1.0):
    # Note: epochs are zero indexed by pytorch
    end_epoch = float(from_epoch + decay_epochs)

    def lr_lambda(epoch):
        if epoch < from_epoch:
            return 1.0
        epoch = min(epoch, end_epoch)
        new_lr = ((lr - end_lr) * (1. - epoch / end_epoch) ** power + end_lr)
        return new_lr / lr  # LambdaLR expects returning a factor

    return lr_lambda


class Trainer(object):
    def __init__(self, args, model=None, train_loader=None, val_loader=None, train_loader_base=None, log=None):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loader_base = train_loader_base

        if self.args.loss == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction='mean')  # train fc_bc
        elif self.args.loss == 'ls':
            self.criterion = CrossEntropyLabelSmooth(self.args.num_classes, epsilon=self.args.eps)
        elif self.args.loss == 'scon':
            self.criterion = SupConLoss(temperature=self.args.temp)

        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
        self.set_scheduler()
        self.log = log

    def set_scheduler(self,):
        if self.args.scheduler in ['cos', 'cosine']:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        elif self.args.scheduler in ['ms', 'multi_step']:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[70, 140], gamma=0.1)

    def train_one_epoch(self):
        # switch to train mode
        self.model.train()
        losses = AverageMeter('Loss', ':.4e')
        train_acc = AverageMeter('Train_acc', ':.4e')

        for i, (inputs, targets) in enumerate(self.train_loader):

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # ==== update loss and acc
            output, h = self.model(inputs, ret='of')
            loss = self.criterion(output, targets)
            losses.update(loss.item(), targets.size(0))
            train_acc.update((output.argmax(dim=-1) == targets).float().mean().item(), targets.size(0))

            # ==== gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return losses, train_acc

    def train_one_epoch_contrast(self):
        self.model.train()
        losses = AverageMeter('Loss', ':.4e')

        for i, (inputs, labels) in enumerate(self.train_loader):
            if isinstance(inputs,list): inputs = torch.cat([inputs[0], inputs[1]], dim=0)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            bsz = labels.shape[0]

            feat = self.model(inputs)
            f1, f2 = torch.split(feat, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if self.args.loss == 'scon':
                loss = self.criterion(features, labels)
            elif self.args.loss == 'simc':
                loss = self.criterion(features)
            losses.update(loss.item(), bsz)

            # ==== gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return losses

    def train_base(self):
        best_acc1 = 0

        # tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(self.model, self.criterion, log="all", log_freq=20)

        for epoch in range(self.args.epochs):
            if self.args.loss == 'ce':
                losses, train_acc = self.train_one_epoch()
            elif self.args.loss in ['scon', 'simc']:
                losses = self.train_one_epoch_contrast()
            wandb.log({'train/train_loss': losses.avg,'train/lr': self.optimizer.param_groups[0]['lr']}, step=epoch)
            
            # ========= evaluate on validation set =========
            if self.args.loss in ['scon', 'simc']:
                self.set_classifier()
                
            train_acc = self.validate(loader=self.train_loader_base)
            val_acc = self.validate(loader=self.val_loader, fine2coarse=self.args.coarse=='fc')
            
            if self.args.coarse[0] == 'f':
                wandb.log({'train/acc_fine': train_acc,}, step=epoch)
            elif self.args.coarse[0] == 'c': 
                wandb.log({'train/acc_coarse': train_acc,}, step=epoch)
            if self.args.coarse[1] == 'f':
                wandb.log({'val/acc_fine': val_acc,}, step=epoch)
            elif self.args.coarse[1] == 'c': 
                wandb.log({'val/acc_coarse': val_acc,}, step=epoch)
                 
            # ========= measure NC =========
            if (epoch + 1) % self.args.debug == 0 and self.args.debug > 0:
                train_nc = analysis(self.model, self.train_loader_base, self.args)
                self.log.info('>>>>Epoch:{}, Train Loss:{:.3f}, Acc:{:.2f}, NC1:{:.3f}, NC3:{:.3f}'.format(
                    epoch, train_nc['loss'], train_nc['acc'], train_nc['nc1'], train_nc['nc3']))
                wandb.log({
                    'train_nc/nc1': train_nc['nc1'],
                    'train_nc/nc2h': train_nc['nc2_h'],
                    'train_nc/nc2w': train_nc['nc2_w'],
                    'train_nc/nc3': train_nc['nc3'],
                    'train_nc/nc3_d': train_nc['nc3_d'],
                    'train_nc2/nc21_h': train_nc['nc21_h'],
                    'train_nc2/nc22_h': train_nc['nc22_h'],
                    'train_nc2/nc21_w': train_nc['nc21_w'],
                    'train_nc2/nc22_w': train_nc['nc22_w'],
                }, step=epoch)

                test_nc = analysis(self.model, self.val_loader, self.args)
                wandb.log({
                    'test_nc/nc1': test_nc['nc1'],
                    'test_nc/nc2h': test_nc['nc2_h'],
                    'test_nc/nc2w': test_nc['nc2_w'],
                    'test_nc/nc3': test_nc['nc3'],
                    'test_nc/nc3_d': test_nc['nc3_d'],
                    'test_nc2/nc21_h': test_nc['nc21_h'],
                    'test_nc2/nc22_h': test_nc['nc22_h'],
                    'test_nc2/nc21_w': test_nc['nc21_w'],
                    'test_nc2/nc22_w': test_nc['nc22_w'],
                }, step=epoch)

                if (epoch + 1) % (self.args.debug * 5) == 0:
                    fig = plot_nc(train_nc)
                    wandb.log({"chart": fig}, step=epoch + 1)

            self.lr_scheduler.step()
            self.model.train()

        self.log.info('Best Testing Prec@1: {:.3f}\n'.format(best_acc1))

    def set_classifier(self):
        if self.args.cls_type in ['ncc']:
            cfeats = self.get_ncc_centroids()
            self.classifier = NCC_Classifier(feat_dim=self.model.encoder.feat_dim, num_classes=self.args.num_classes,
                                             feat_type='cl2n', dist_type='l2')
            self.classifier.update(cfeats)
        elif self.args.cls_type in ['linear']:
            self.classifier = LinearClassifier(feat_dim=self.model.encoder.feat_dim, num_classes=self.args.num_classes)
            self.update_linear_classifier()

    def validate(self, loader=None, fine2coarse=False):
        torch.cuda.empty_cache()
        self.model.eval()
        all_preds, all_targets, all_feats = [], [], []

        with torch.no_grad():
            for i, (input, target) in enumerate(loader):
                input = input.to(self.device)
                target = target.to(self.device)

                if self.args.loss in ['ce']:
                    output = self.model(input, ret='o')
                elif self.args.loss in ['scon', 'simc'] and self.args.cls_type == 'ncc':
                    output = self.classifier(self.model.encoder(input))
                _, pred = torch.max(output, 1)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        if fine2coarse:
            vectorized_map = np.vectorize(fine_id_coarse_id.get)
            preds = vectorized_map(np.array(all_preds))
        else:
            preds = np.array(all_preds)
        targets = np.array(all_targets)

        acc = np.sum(preds == targets)/len(targets)
        return acc

    def update_linear_classifier(self, epochs=1):
        self.model.eval()
        self.classifier.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)

        for epoch in range(epochs):
            losses, top1 = AverageMeter(), AverageMeter()
            for idx, (inputs, labels) in enumerate(self.train_loader):
                if isinstance(inputs, list):
                    inputs = torch.cat([inputs[0], inputs[1]], dim=0); labels = torch.cat([labels, labels], dim=0)
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                bsz = labels.shape[0]

                # compute loss
                with torch.no_grad():
                    features = self.model.encoder(inputs)
                output = self.classifier(features.detach())
                loss = criterion(output, labels)

                # update metric
                losses.update(loss.item(), bsz)
                acc = (output.argmax(dim=-1) == labels).sum().item()/len(labels)
                top1.update(acc, bsz)

                # SGD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # print info
            self.log.info('Classifier acc: {:.3f}\n'.format(top1.avg))

    def get_ncc_centroids(self):
        # print('===> Calculating centroids.')
        feats, labels = self.extract_feat(self.train_loader)
        feats, labels = feats.cpu().numpy(), labels.cpu().numpy()
        featmean = feats.mean(axis=0)

        # Get unnormalized centorids
        un_centers = get_centroids(feats, labels)

        # Get l2n centorids
        l2n_feats = torch.Tensor(feats.copy())
        norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
        l2n_feats = l2n_feats / norm_l2n
        l2n_centers = get_centroids(l2n_feats.numpy(), labels)

        # Get cl2n centorids
        cl2n_feats = torch.Tensor(feats.copy())
        cl2n_feats = cl2n_feats - torch.Tensor(featmean)
        norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
        cl2n_feats = cl2n_feats / norm_cl2n
        cl2n_centers = get_centroids(cl2n_feats.numpy(), labels)

        return {'mean': featmean,
                'uncs': un_centers,
                'l2ncs': l2n_centers,
                'cl2ncs': cl2n_centers,
                }
        
    def extract_feat(self, data_loader):
        torch.cuda.empty_cache()
        self.model.eval() 
        feats_all, labels_all = [], []
        with torch.set_grad_enabled(False):
            for i, (inputs, labels) in enumerate(data_loader):
                if isinstance(inputs, list):
                    inputs = torch.cat(inputs, dim=0)
                    labels = torch.cat((labels, labels), dim=0)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                feats = self.model.encoder(inputs)
                feats_all.append(feats)
                labels_all.append(labels)
            feats = torch.cat(feats_all, dim=0)
            labels = torch.cat(labels_all, dim=0)
        return feats, labels
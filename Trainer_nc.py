from imbalance_data.cifar100_coarse2fine import fine_id_coarse_id
import wandb
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from utils.util import *
from utils.plot import plot_nc
from utils.measure_nc import analysis, analysis_feat
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
        self.log = log
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

        if self.args.loss == 'ce' and self.args.coarse == 'fc' and self.args.cs_loss:
            self.coarse_classifier = nn.Linear(self.model.encoder.feat_dim, self.args.num_coarse).to(self.device)
            self.optimizer = torch.optim.SGD(list(self.model.parameters()) + list(self.coarse_classifier.parameters()),
                                             momentum=0.9, lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=args.lr,
                                             weight_decay=args.weight_decay)
        self.set_scheduler()

    def set_scheduler(self, ):
        if self.args.scheduler in ['cos', 'cosine']:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        elif self.args.scheduler in ['ms', 'multi_step']:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.args.epochs//3, 2*(self.args.epochs//3)], gamma=0.1)

    def train_one_epoch(self):
        # switch to train mode
        self.model.train()
        losses = AverageMeter('Loss', ':.4e')
        train_acc = AverageMeter('Train_acc', ':.4e')

        for i, (inputs, labels) in enumerate(self.train_loader):

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # ==== update loss and acc
            output, h = self.model(inputs, ret='of')
            loss = self.criterion(output, labels)
            losses.update(loss.item(), labels.size(0))
            train_acc.update((output.argmax(dim=-1) == labels).float().mean().item(), labels.size(0))

            if self.args.cs_loss and self.args.coarse == 'fc':
                output_cs = self.coarse_classifier(h)
                vectorized_map = np.vectorize(fine_id_coarse_id.get)
                coarse_labels = torch.from_numpy(vectorized_map(np.array(labels.cpu().numpy()))).to(self.device)
                loss += self.criterion(output_cs, coarse_labels) * self.args.cs_wt

            # ==== gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return losses, train_acc

    def train_one_epoch_contrast(self):
        self.model.train()
        losses = AverageMeter('Loss', ':.4e')

        for i, (inputs, labels) in enumerate(self.train_loader):
            if isinstance(inputs, list): inputs = torch.cat([inputs[0], inputs[1]], dim=0)
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

            if self.args.cs_loss:
                vectorized_map = np.vectorize(fine_id_coarse_id.get)
                coarse_labels = torch.from_numpy(vectorized_map(np.array(labels.cpu().numpy()))).to(self.device)
                loss += self.criterion(features, coarse_labels) * self.args.cs_wt

            # ==== gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return losses

    def train_base(self):

        # tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(self.model, self.criterion, log="all", log_freq=20)

        for epoch in range(self.args.epochs):
            if self.args.loss == 'ce':
                losses, train_acc = self.train_one_epoch()
            elif self.args.loss in ['scon', 'simc']:
                losses = self.train_one_epoch_contrast()
            
            if epoch == 0 or (epoch+1) % self.args.log_freq == 0:
                wandb.log({'train/train_loss': losses.avg, 'train/lr': self.optimizer.param_groups[0]['lr']}, step=epoch)

                # ========= evaluate on validation set =========
                if self.args.loss in ['scon', 'simc']:
                    self.set_classifier()

                train_dt = self.validate(loader=self.train_loader_base)
                val_dt = self.validate(loader=self.val_loader, fine2coarse=self.args.coarse == 'fc')

                if self.args.coarse[0] == 'f':
                    wandb.log({'train/acc_fine': train_dt['acc'], }, step=epoch)
                elif self.args.coarse[0] == 'c':
                    wandb.log({'train/acc_coarse': train_dt['acc'], }, step=epoch)
                if self.args.coarse[1] == 'f':
                    wandb.log({'val/acc_fine': val_dt['acc'], }, step=epoch)
                elif self.args.coarse[1] == 'c':
                    wandb.log({'val/acc_coarse': val_dt['acc'], }, step=epoch)

                if self.args.coarse == 'fc' and self.args.cs_loss:
                    wandb.log({'train/acc_coarse_cs': train_dt['acc_cs'], 'val/acc_coarse_cs': val_dt['acc_cs']},
                            step=epoch)

            # ========= measure NC =========
            if (epoch + 1) % self.args.nc_freq == 0 and self.args.nc_freq > 0:
                train_nc = analysis_feat(train_dt['labels'], train_dt['feats'], self.args)
                val_nc = analysis_feat(val_dt['labels'], val_dt['feats'], self.args)
                wandb.log({
                    'train_nc/nc1': train_nc['nc1'],
                    'train_nc/nc2h': train_nc['nc2h'],
                    'train_nc/h_norm': train_nc['h_norm'],
                    'val_nc/nc1': val_nc['nc1'],
                    'val_nc/nc2h': val_nc['nc2h'],
                    'val_nc/h_norm': val_nc['h_norm'],
                }, step=epoch)

                if (epoch + 1) % (self.args.debug * 5) == 0:
                    acc_cls = self.classwise_acc(val_dt['labels'].cpu().numpy(), val_dt['preds'].cpu().numpy())
                    data = [[label, val_tr, val_va, acc] for (label, val_tr, val_va, acc) in
                            zip(np.arange(self.args.num_classes), train_nc['nc1_cls'], val_nc['nc1_cls'], acc_cls)]
                    table = wandb.Table(data=data, columns=["label", "nc1_train", 'nc1_val', 'acc'])
                    wandb.log({"per class nc1 train": wandb.plot.bar(table, "label", "nc1_train", title="train nc1")},
                              step=epoch)
                    wandb.log({"per class nc1 val": wandb.plot.bar(table, "label", "nc1_val", title="val nc1")},
                              step=epoch)
                    wandb.log({"per class acc val": wandb.plot.bar(table, "label", "acc", title="val acc")}, step=epoch)

                try:
                    train_nc_all.load_dt(train_nc)
                    val_nc_all.load_dt(val_nc)
                except:
                    train_nc_all = Graph_Vars(train_nc)
                    val_nc_all = Graph_Vars(val_nc)
                    train_nc_all.load_dt(train_nc, epoch=epoch)
                    val_nc_all.load_dt(val_nc, epoch=epoch)

            self.lr_scheduler.step()
            self.model.train()

            if (epoch + 1 == self.args.epochs // 2) or (epoch + 1 == self.args.epochs):
                with open(os.path.join(self.args.root_model + self.args.store_name, 'nc_dt.pkl'), 'wb') as f:
                    pickle.dump([train_nc_all, val_nc_all], f)

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
        all_preds, all_labels, all_preds_cs, all_feats = [], [], [], []

        with torch.no_grad():
            for i, (input, label) in enumerate(loader):
                input = input.to(self.device)
                label = label.to(self.device)

                if self.args.loss in ['ce']:
                    output, feat = self.model(input, ret='of')
                elif self.args.loss in ['scon', 'simc'] and self.args.cls_type == 'ncc':
                    feat = self.model.encoder(input)
                    output = self.classifier(feat)

                if self.args.cs_loss and self.args.loss == 'ce':
                    output_cs = self.coarse_classifier(feat)
                    all_preds_cs.append(output_cs.argmax(dim=-1))

                all_preds.append(output.argmax(dim=-1))
                all_labels.append(label)
                all_feats.append(feat)

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_feats = torch.cat(all_feats, dim=0)

        if fine2coarse:
            vectorized_map = np.vectorize(fine_id_coarse_id.get)
            all_preds = torch.from_numpy(vectorized_map(all_preds.cpu().numpy())).to(self.device)
        acc = (all_preds == all_labels).float().mean().item()

        if self.args.cs_loss and self.args.coarse == 'fc' and self.args.loss == 'ce':
            all_preds_cs = torch.cat(all_preds_cs, dim=0)
            acc_cs = (all_preds_cs == all_labels).float().mean().item()
            acc_dt = {'acc': acc, 'acc_cs': acc_cs, 'feats': all_feats, 'labels': all_labels, 'preds': all_preds}
        else:
            acc_dt = {'acc': acc, 'feats': all_feats, 'labels': all_labels, 'preds': all_preds}

        return acc_dt

    def update_linear_classifier(self, epochs=1):
        self.model.eval()
        self.classifier.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.args.lr, momentum=0.9,
                                    weight_decay=self.args.weight_decay)

        for epoch in range(epochs):
            losses, top1 = AverageMeter(), AverageMeter()
            for idx, (inputs, labels) in enumerate(self.train_loader):
                if isinstance(inputs, list):
                    inputs = torch.cat([inputs[0], inputs[1]], dim=0);
                    labels = torch.cat([labels, labels], dim=0)
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
                acc = (output.argmax(dim=-1) == labels).sum().item() / len(labels)
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

    def classwise_acc(self, targets, preds):
        eps = np.finfo(np.float64).eps
        cf = confusion_matrix(targets, preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        return cls_acc
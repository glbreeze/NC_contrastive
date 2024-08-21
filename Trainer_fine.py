from imbalance_data.cifar100_coarse2fine import fine_id_coarse_id, fine_id_sub_id, coarse_id_fine_id
import wandb
import torch.nn as nn

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

        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=args.lr,
                                         weight_decay=args.weight_decay)
        self.set_scheduler()
        self.log = log

    def set_scheduler(self, ):
        if self.args.scheduler in ['cos', 'cosine']:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        elif self.args.scheduler in ['ms', 'multi_step']:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[70, 140], gamma=0.1)

    def train_one_epoch(self):
        # switch to train mode
        self.model.train()
        coarse_losses, sub_losses = AverageMeter('Loss', ':.4e'), AverageMeter('Loss', ':.4e')
        train_acc = AverageMeter('Train_acc', ':.4e')

        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            if self.args.coarse == 'f':
                vectorized_map = np.vectorize(fine_id_coarse_id.get)
                coarse_labels = vectorized_map(np.array(labels.cpu().numpy()))
                vectorized_map = np.vectorize(fine_id_sub_id.get)
                sub_labels = vectorized_map(np.array(labels.cpu().numpy()))
                coarse_labels, sub_labels = torch.from_numpy(coarse_labels).to(self.device), torch.from_numpy(sub_labels).to(self.device)
            elif self.args.coarse == 'c': 
                coarse_labels = labels.to(self.device)
            
            logits, h = self.model(inputs, ret='of')
            coarse_loss = self.criterion(logits[:, :self.args.num_coarse], coarse_labels)
            sub_loss = 0
            if self.args.coarse == 'f': 
                for idx in range(self.args.num_coarse):
                    selected_idx = coarse_labels == idx
                    selected_logits = logits[selected_idx]
                    selected_sub_labels = sub_labels[selected_idx]
                    within_loss = self.criterion(selected_logits[:, self.args.num_coarse+idx*self.args.num_sub: self.args.num_coarse+(idx+1)*self.args.num_sub],
                                                 selected_sub_labels)
                    sub_loss += within_loss * len(selected_logits)
                    
                sub_loss = sub_loss / labels.size(0)
                loss = coarse_loss + sub_loss * self.args.sub_wt
            else: 
                loss = coarse_loss
                sub_loss = torch.tensor(0.0)
                
            coarse_acc = (logits[:, :self.args.num_coarse].argmax(dim=-1) == coarse_labels).float().mean()
            coarse_losses.update(coarse_loss.item(), labels.size(0))
            sub_losses.update(sub_loss.item(), labels.size(0))
            train_acc.update(coarse_acc.item(), labels.size(0))
            # ==== gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return coarse_losses, sub_losses, train_acc

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
                coarse_losses, sub_losses, coarse_acc = self.train_one_epoch()
            elif self.args.loss in ['scon', 'simc']:
                losses = self.train_one_epoch_contrast()

            wandb.log({'train/coarse_loss': coarse_losses.avg, 
                       'train/sub_loss': sub_losses.avg, 
                       'train/acc_cs_run': coarse_acc.avg,
                       'train/lr': self.optimizer.param_groups[0]['lr']}, step=epoch)

            # ========= evaluate on validation set =========
            if self.args.loss in ['scon', 'simc']:
                self.set_classifier()
                
            acc_cs_tr, acc_fn_tr = self.validate(loader=self.train_loader_base)
            acc_cs_va, acc_fn_va = self.validate(loader=self.val_loader)
            wandb.log({
                'train/acc_coarse': acc_cs_tr, 'train/acc_fine': acc_fn_tr,
                'val/acc_coarse': acc_cs_va, 'val/acc_fine': acc_fn_va,
                       }, step=epoch)

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

    def logits_to_label(self, logits):
        coarse_logits = logits[:, :self.args.num_coarse]
        coarse_id = coarse_logits.argmax(dim=-1)

        sub_logits = logits[:, self.args.num_coarse:]
        reshaped_sub_logits = sub_logits.view(sub_logits.size(0), -1, self.args.num_sub)
        sub_id = reshaped_sub_logits.argmax(dim=-1)

        fine_id = np.array([coarse_id_fine_id[cs][sub[cs]] for cs, sub in zip(coarse_id.cpu().numpy(), sub_id.cpu().numpy())])
        selected_sub_id = sub_id[torch.arange(sub_id.size(0)), coarse_id]
        return coarse_id, torch.from_numpy(fine_id).to(coarse_id.device), selected_sub_id

    def validate(self, loader=None):
        torch.cuda.empty_cache()
        self.model.eval()
        all_preds_cs, all_preds_fn, all_preds_sub, all_labels_cs, all_labels_fn, all_labels_sub = [], [], [], [], [], []

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if self.args.loss in ['ce']:
                    logits = self.model(inputs, ret='o')
                elif self.args.loss in ['scon', 'simc'] and self.args.cls_type == 'ncc':
                    logits = self.classifier(self.model.encoder(inputs))
                    
                if self.args.coarse == 'f':
                    coarse_pred, fine_pred, sub_pred = self.logits_to_label(logits)
                    vectorized_map = np.vectorize(fine_id_coarse_id.get)
                    coarse_labels = vectorized_map(np.array(labels.cpu().numpy()))
                    vectorized_map = np.vectorize(fine_id_sub_id.get)
                    sub_labels = vectorized_map(np.array(labels.cpu().numpy()))
                    coarse_labels, sub_labels = torch.from_numpy(coarse_labels).to(self.device), torch.from_numpy(sub_labels).to(self.device)
                    
                    all_preds_fn.append(fine_pred)
                    all_labels_fn.append(labels)
                    
                elif self.args.coarse == 'c': 
                    coarse_pred = logits.argmax(dim=-1)
                    coarse_labels = labels
                    
                all_preds_cs.append(coarse_pred)
                all_labels_cs.append(coarse_labels)
                
        all_labels_cs = torch.cat(all_labels_cs, dim=0)
        all_preds_cs = torch.cat(all_preds_cs, dim=0)
        acc_cs = (all_labels_cs == all_preds_cs).float().mean().item()
        
        if self.args.coarse == 'f':
            all_labels_fn = torch.cat(all_labels_fn, dim=0)
            all_preds_fn = torch.cat(all_preds_fn, dim=0)
            acc_fn = (all_labels_fn == all_preds_fn).float().mean().item()
        else: 
            acc_fn = 0.0 
        
        return acc_cs, acc_fn

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
                feats_all.append(feats.cpu().numpy())
                labels_all.append(labels.cpu().numpy())
            feats = torch.cat(feats_all, dim=0)
            labels = torch.cat(labels_all, dim=0)
        return feats, labels
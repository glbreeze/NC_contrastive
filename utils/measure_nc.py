# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
from scipy.sparse.linalg import svds
import scipy.linalg


def compute_ETF(W, device):  # W [K, 512]
    K = W.shape[0]
    # W = W - torch.mean(W, dim=0, keepdim=True)
    WWT = torch.mm(W, W.T)            # [K, 512] [512, K] -> [K, K]
    WWT /= torch.norm(WWT, p='fro')   # [K, K]

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device) / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_W_H_relation(W, H, device):  # W:[K, 512] H:[512, K]
    """ H is already normalized"""
    K = W.shape[0]

    # W = W - torch.mean(W, dim=0, keepdim=True)
    WH = torch.mm(W, H.to(device))   # [K, 512] [512, K]
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device)

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item()


def analysis(model, loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = [0 for _ in range(args.num_classes)]  # within class sample size
    mean = [0 for _ in range(args.num_classes)]
    Sw_cls = [0 for _ in range(args.num_classes)]

    # get the logit, label, feats
    model.eval()
    logits_list = []
    labels_list = []
    feats_list = []
    with torch.no_grad():
        for data, target in loader:
            if isinstance(data, list): 
                data = torch.cat(data, dim=0)
                target = torch.cat((target, target), dim=0)
            data, target = data.to(device), target.to(device)
            if args.arch.startswith('mres'):
                logits, feats = model(data, target, ret='of')
            else:
                logits, feats = model(data, ret='of')
            logits_list.append(logits)
            labels_list.append(target)
            feats_list.append(feats)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        feats = torch.cat(feats_list).to(device)
    if args.loss in ['arcf', 'arcm'] and args.bias == 'g':
        feats = F.normalize(feats-model.fc.mu.detach())

    loss = torch.nn.CrossEntropyLoss(reduction='mean')(logits, labels).item()
    acc = (logits.argmax(dim=-1) == labels).sum().item()/len(labels)

    # ====== compute mean and var for each class
    for c in range(args.num_classes):
        idxs = (labels == c).nonzero(as_tuple=True)[0]
        h_c = feats[idxs, :]  # [B, 512]

        N[c] = h_c.shape[0]
        mean[c] = torch.sum(h_c, dim=0)/h_c.shape[0]  #  CHW

        # update within-class cov
        z = h_c - mean[c].unsqueeze(0)  # [B, 512]
        cov_c = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1))   # [B 512 1] [B 1 512] -> [B, 512, 512]
        Sw_cls[c] = torch.sum(cov_c, dim=0)  # [512, 512]

    # global mean
    M = torch.stack(mean).T
    muG = torch.mean(M, dim=1, keepdim=True)  # [512, C]
    Sw = sum(Sw_cls) / sum(N)
    for c in range(args.num_classes):
            Sw_cls[c] = Sw_cls[c] / N[c]

    # between-class covariance
    M_ = M - muG  # [512, C]
    Sb = torch.matmul(M_, M_.T) / args.num_classes

    # ============ NC1: tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=args.num_classes - 1)
    inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    nc1 = np.trace(Sw @ inv_Sb)
    nc1_cls = [np.trace(Sw_cls1.cpu().numpy() @ inv_Sb) for Sw_cls1 in Sw_cls]
    nc1_cls = np.array(nc1_cls)

    # ========== NC2.1 and NC2.2
    has_fc_cb = any(name == "fc_cb" for name, _ in model.named_modules())
    if has_fc_cb:
        W = model.fc_cb.weight.detach().T  # [512, C]
    else: 
        W = model.fc.weight.detach().T
    if args.loss in ['arcf', 'arcm'] and args.bias == 'g':
        W = F.normalize(W, p=2, dim=0)

    M_norms = torch.norm(M_, dim=0)  # [C]
    W_norms = torch.norm(W , dim=0)  # [C]

    # == NC2.1 Norm
    norm_M_CoV = (torch.std(M_norms) / torch.mean(M_norms)).item()
    norm_W_CoV = (torch.std(W_norms) / torch.mean(W_norms)).item()

    # == NC2.2
    def coherence(V):
        G = V.T @ V  # [C, D] [D, C]
        G += torch.ones((args.num_classes, args.num_classes), device=device) / (args.num_classes - 1)
        G -= torch.diag(torch.diag(G))  # [C, C]
        return torch.norm(G, 1).item() / (args.num_classes * (args.num_classes - 1))

    cos_M = coherence(M_ / M_norms)  # [D, C]
    cos_W = coherence(W / W_norms)

    # angle between W
    W_nomarlized = W / W_norms  # [512, C]
    w_cos = (W_nomarlized.T @ W_nomarlized).cpu().numpy()  # [C, D] [D, C] -> [C, C]
    w_cos_avg = (w_cos.sum(1) - np.diag(w_cos)) / (w_cos.shape[1] - 1) # [C]

    # angle between H
    M_normalized = M_ / M_norms  # [512, C]
    h_cos = (M_normalized.T @ M_normalized).cpu().numpy()
    h_cos_avg = (h_cos.sum(1) - np.diag(h_cos)) / (h_cos.shape[1] - 1)

    # angle between W and H
    wh_cos = (W_nomarlized.T @ M_normalized).cpu().numpy()
    wh_cos_avg = np.mean(np.diag(wh_cos))

    # =========== NC2
    nc2_h = compute_ETF(M_.T, device)
    nc2_w = compute_ETF(W.T, device)

    # =========== NC3  ||W^T - M_||
    normalized_M = M_ / torch.norm(M_, 'fro')
    normalized_W = W / torch.norm(W, 'fro')
    W_M_dist = (torch.norm(normalized_W - normalized_M) ** 2).item()

    # =========== NC3 (all losses are equal paper)
    nc3 = compute_W_H_relation(W.T, M_, device)
    
    # =========== for unbalanced dataset: 
    if args.imbalance_type == 'step': 
        w_mnorm1 = np.mean(W_norms[:args.num_classes//2].cpu().numpy())
        w_mnorm2 = np.mean(W_norms[args.num_classes//2:].cpu().numpy())
        h_mnorm1 = np.mean(M_norms[:args.num_classes//2].cpu().numpy())
        h_mnorm2 = np.mean(M_norms[args.num_classes//2:].cpu().numpy())
        
        w_cos1_ = w_cos[:args.num_classes//2, :args.num_classes//2]
        w_cos1  = (w_cos1_.sum(1) - np.diag(w_cos1_)) / (w_cos1_.shape[1] - 1)
        w_cos2_ = w_cos[args.num_classes//2:, args.num_classes//2:]
        w_cos2  = (w_cos2_.sum(1) - np.diag(w_cos2_)) / (w_cos2_.shape[1] - 1)
        w_cos3_ = w_cos[:args.num_classes//2, args.num_classes//2:]
        w_cos3  = w_cos3_.sum(1) / w_cos3_.shape[1] 
        
        h_cos1_ = h_cos[:args.num_classes//2, :args.num_classes//2]
        h_cos1  = (h_cos1_.sum(1) - np.diag(h_cos1_)) / (h_cos1_.shape[1] - 1)
        h_cos2_ = h_cos[args.num_classes//2:, args.num_classes//2:]
        h_cos2  = (h_cos2_.sum(1) - np.diag(h_cos2_)) / (h_cos2_.shape[1] - 1)
        h_cos3_ = h_cos[:args.num_classes//2, args.num_classes//2:]
        h_cos3  = h_cos3_.sum(1) / h_cos3_.shape[1] 
     

    return {
        "loss": loss,
        "acc": acc,
        "nc1": nc1,
        "nc1_cls": nc1_cls,
        "w_norm": W_norms.cpu().numpy(),
        "h_norm": M_norms.cpu().numpy(),
        "w_mnorm": np.mean(W_norms.cpu().numpy()),
        "h_mnorm": np.mean(M_norms.cpu().numpy()),
        "w_cos": w_cos, 
        "w_cos_avg": w_cos_avg,
        "h_cos": h_cos,
        "h_cos_avg": h_cos_avg,
        "wh_cos": wh_cos, 
        "wh_cos_avg": wh_cos_avg, 
        "nc21_h": norm_M_CoV,
        "nc21_w": norm_W_CoV,
        "nc22_h": cos_M,
        "nc22_w": cos_W,
        "nc2_h": nc2_h,
        "nc2_w": nc2_w,
        "nc3": nc3,
        "nc3_d": W_M_dist,
        "w_mnorm1": w_mnorm1 if args.imbalance_type == 'step' else 0, 
        "w_mnorm2": w_mnorm2 if args.imbalance_type == 'step' else 0, 
        "h_mnorm1": h_mnorm1 if args.imbalance_type == 'step' else 0, 
        "h_mnorm2": h_mnorm2 if args.imbalance_type == 'step' else 0,
        "w_cos1": np.mean(w_cos1) if args.imbalance_type == 'step' else 0,
        "w_cos2": np.mean(w_cos2) if args.imbalance_type == 'step' else 0,
        "w_cos3": np.mean(w_cos3) if args.imbalance_type == 'step' else 0,
        "h_cos1": np.mean(h_cos1) if args.imbalance_type == 'step' else 0,
        "h_cos2": np.mean(h_cos2) if args.imbalance_type == 'step' else 0,
        "h_cos3": np.mean(h_cos3) if args.imbalance_type == 'step' else 0,
    }


def analysis_feat(labels, feats, num_classes, W=None):
    # analysis without extracting features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_cls = [0 for _ in range(num_classes)]  # within class sample size
    mean_cls = [0 for _ in range(num_classes)]
    cov_cls = [0 for _ in range(num_classes)]

    # ====== compute mean and var for each class
    for c in range(num_classes):

        feats_c = feats[labels == c]   # [N, 512]

        num_cls[c] = len(feats_c)
        mean_cls[c] = torch.mean(feats_c, dim=0)

        # update within-class cov
        X = feats_c - mean_cls[c].unsqueeze(0)   # [N, 512]
        cov_cls[c] = X.T @ X / num_cls[c]        # [512, 512]

    # global mean
    M = torch.stack(mean_cls)        # [K, 512]
    mean_all = torch.mean(M, dim=0)  # [512]

    Sigma_b = (M - mean_all.unsqueeze(0)).T @ (M - mean_all.unsqueeze(0)) / num_classes
    Sigma_w = torch.stack([cov * num for cov, num in zip(cov_cls, num_cls)]).sum(dim=0) / sum(num_cls)
    Sigma_t = (feats - mean_all.unsqueeze(0)).T @ (feats - mean_all.unsqueeze(0)) / len(feats)

    Sigma_b = Sigma_b.cpu().numpy()
    Sigma_w = Sigma_w.cpu().numpy()
    nc1 = np.trace(Sigma_w @ scipy.linalg.pinv(Sigma_b))
    nc1_cls = [np.trace(cov.cpu().numpy() @ scipy.linalg.pinv(Sigma_b)) for cov in cov_cls]
    nc1_cls = np.array(nc1_cls)

    # =========== NC2
    nc2h = compute_ETF(M - mean_all.unsqueeze(0), device)
    h_norm = torch.norm(M - mean_all.unsqueeze(0), dim=-1).mean().item()
    if W is not None:
        nc2w = compute_ETF(W, device)
        w_norm = torch.norm(W, dim=-1).mean().item()
        nc2 = compute_W_H_relation(W, (M - mean_all.unsqueeze(0)).T, device)

        # =========== NC3 (all losses are equal paper)
        normalized_M = (M - mean_all.unsqueeze(0)) / torch.norm(M - mean_all.unsqueeze(0), 'fro')
        normalized_W = W / torch.norm(W, 'fro')
        nc3d = (torch.norm(normalized_W - normalized_M) ** 2).item()
        # nc3 = compute_W_H_relation(W, M - mean_all.unsqueeze(0), device)

    nc_dt = {
        'nc1': nc1,
        'nc1_cls': nc1_cls,
        'nc2h': nc2h,
        'h_norm': h_norm,
    }

    if W is not None:
        nc_dt.update({'nc2w': nc2w, 'w_norm':w_norm, 'nc2':nc2, 'nc3d':nc3d})

    return nc_dt
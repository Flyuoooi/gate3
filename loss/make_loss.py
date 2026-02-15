# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss



def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATA.SAMPLER
    feat_dim = 1024
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

        # ---------------- Proxy-Anchor cached config (minimal-invasion) ----------------
    text_cfg = getattr(cfg.MODEL, "TEXT", None)



    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATA.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, model=None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                # ---------------- CE / ID loss (保持你原逻辑不动) ----------------
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                # ---------------- Triplet loss (保持你原逻辑不动) ----------------
                if isinstance(feat, list):
                    TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                    TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                    TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    feat_main = feat[0]
                else:
                    TRI_LOSS = triplet(feat, target)[0]
                    feat_main = feat

                total = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                text_cfg = getattr(cfg.MODEL, "TEXT", None)

                # ---------------- Proxy-Similarity matching (Design B, optional) ----------------
                SIM_LOSS = None
                sim_on = (text_cfg is not None) and bool(getattr(text_cfg, "PROXY_SIM_ON", False))
                sim_w = float(getattr(text_cfg, "PROXY_SIM_W", 0.0)) if sim_on else 0.0
                lam_now = 0.0
                if sim_on and sim_w > 0.0 and (model is not None):
                    m2 = model.module if hasattr(model, "module") else model
                    if not hasattr(m2, "head") or not hasattr(m2.head, "weight"):
                        raise AttributeError("Proxy-Sim expects model.head.weight to exist.")
                    head_w2 = m2.head.weight
                    px2 = head_w2.index_select(0, target.to(head_w2.device)).to(feat_main.device)  # [B, feat_dim]

                    f2 = F.normalize(feat_main, dim=1)
                    p2 = F.normalize(px2, dim=1)
                    Sf = f2 @ f2.t()
                    Sp = p2 @ p2.t()
                    B2 = Sf.size(0)
                    if B2 > 1:
                        mask2 = ~torch.eye(B2, dtype=torch.bool, device=Sf.device)
                        SIM_LOSS = F.mse_loss(Sf[mask2], Sp[mask2])
                        # SIM_LOSS = F.smooth_l1_loss(Sf[mask2], Sp[mask2], beta=1.0)   # 换l1看效果
                    else:
                        SIM_LOSS = Sf.new_tensor(0.0)

                    lam_now = float(m2.get_proxy_lambda()) if hasattr(m2, "get_proxy_lambda") else 1.0
                    sim_lam_capped = min(lam_now, 0.5)
                    total = total + (sim_w * sim_lam_capped) * SIM_LOSS

                # For processor.py logging
                loss_func._last = {
                    "id": float(ID_LOSS.detach().item()),
                    "tri": float(TRI_LOSS.detach().item()),
                    "sim": float(SIM_LOSS.detach().item()) if SIM_LOSS is not None else 0.0,
                    "lam": float(lam_now),
                }

                return total
               
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                    'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    # elif cfg.DATA.SAMPLER == 'softmax_triplet':
    #     def loss_func(score, feat, target, target_cam):
    #         if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
    #             if cfg.MODEL.IF_LABELSMOOTH == 'on':
    #                 if isinstance(score, list):
    #                     ID_LOSS = [xent(scor, target) for scor in score[1:]]
    #                     ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
    #                     ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
    #                 else:
    #                     ID_LOSS = xent(score, target)

    #                 if isinstance(feat, list):
    #                         TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
    #                         TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
    #                         TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
    #                 else:
    #                         TRI_LOSS = triplet(feat, target)[0]

    #                 return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
    #                            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
    #             else:
    #                 if isinstance(score, list):
    #                     ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
    #                     ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
    #                     ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
    #                 else:
    #                     ID_LOSS = F.cross_entropy(score, target)

    #                 if isinstance(feat, list):
    #                         TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
    #                         TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
    #                         TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
    #                 else:
    #                         TRI_LOSS = triplet(feat, target)[0]

    #                 return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
    #                            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
    #         else:
    #             print('expected METRIC_LOSS_TYPE should be triplet'
    #                   'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion



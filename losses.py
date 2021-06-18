from typing import List
import torch
from torch import nn
from torch import Tensor, einsum
import math
from utils import simplex, probs2one_hot
from sdf import compute_sdf


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        # print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type(torch.float32)

        loss = - einsum("bcwh,bcwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class DiceLoss():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]

    def __call__(self, probs: Tensor, targets: Tensor) -> Tensor:
        dices = (2 * einsum("bcwh->bc", probs[:, self.idc, ...] * targets[:, self.idc, ...].float()) + 1e-10) \
                / ((einsum("bcwh->bc", probs[:, self.idc, ...]) + einsum("bcwh->bc",
                                                                         targets[:, self.idc, ...].float())) + 1e-10)
        dices = 1 - dices
        loss = einsum("bc->", dices)
        return loss


class reconLoss():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]

    def __call__(self, probs: Tensor, targets: Tensor) -> Tensor:
        criterion = nn.MSELoss()
        loss = criterion(probs[:, self.idc, ...], targets)
        return loss


class LabeledPenalty():
    """
    implements the labeled penalty : crossEntropy + NaivePenalty
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.max_epc = 100
        self.sdf_weight = kwargs['weights'][0]
        self.rec_weight = kwargs['weights'][1]

    def __call__(self, image, probs: Tensor, targets: Tensor, sdf_probs: Tensor, rec_probs: Tensor,
                 ema_mask: Tensor, ema_sdf: Tensor, ema_recon: Tensor, ema_seg_uncertainty, ema_rec_uncertainty,
                 ema_sdf_uncertainty, epc) -> Tensor:
        rec_criterion = reconLoss(**self.kwargs)
        seg_criterion = DiceLoss(**self.kwargs)

        b, c, w, h = probs.size()
        sdf = compute_sdf(targets[:, 1, ...].detach().cpu().numpy(), (b, w, h)) + 1
        sdf = torch.from_numpy(sdf).cuda().float()
        sdf_probs = sdf_probs * 2

        rec_loss = rec_criterion(rec_probs, image * targets.float())
        sdf_loss = rec_criterion(sdf_probs, sdf)
        seg_loss = seg_criterion(probs, targets)

        sdf_consistency = torch.mean(torch.pow(sdf_probs - ema_sdf, 2) * torch.exp(-ema_sdf_uncertainty))
        rec_consistency = torch.mean(torch.pow(rec_probs - ema_recon, 2) * torch.exp(-ema_rec_uncertainty))
        seg_consistency = torch.pow(probs - ema_mask, 2) * (1 - ema_seg_uncertainty)
        seg_consistency = torch.mean(seg_consistency)

        consistency_weight = 0.1 * math.exp(-5 * math.pow((1 - epc / self.max_epc), 2))
        loss = seg_loss + self.sdf_weight * sdf_loss + self.rec_weight * rec_loss + \
               consistency_weight * (sdf_consistency + seg_consistency + rec_consistency)
        return loss


class UnLabeledPenalty():
    """
        implements the unlabeled penalty : crossEntropy + NaivePenalty
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.max_epc = 100
        self.rec_seg_weight = kwargs['weights'][0]
        self.unlabeled_weight = kwargs["unlabled_weight"]

    def __call__(self, image, probs: Tensor, _: Tensor, sdf_probs: Tensor, rec_probs: Tensor,
                 ema_mask: Tensor, ema_sdf: Tensor, ema_recon: Tensor, ema_seg_uncertainty,
                 ema_rec_uncertainty, ema_sdf_uncertainty, epc) -> Tensor:
        rec_criterion = reconLoss(**self.kwargs)
        predicted_mask: Tensor = probs2one_hot(probs.detach())  # Used only for dice computation

        b, c, w, h = probs.size()
        sdf_fake = compute_sdf(predicted_mask[:, 1, ...].detach().cpu().numpy(), (b, w, h)) + 1
        sdf = torch.from_numpy(sdf_fake).cuda().float()
        sdf_consistency = torch.mean(torch.pow(sdf_probs-ema_sdf, 2) * torch.exp(-ema_sdf_uncertainty))
        seg_consistency = torch.pow(probs - ema_mask, 2) * (1 - ema_seg_uncertainty)
        seg_consistency = torch.mean(seg_consistency)
        rec_consistency = torch.mean(torch.pow(rec_probs-ema_recon, 2) * torch.exp(-ema_rec_uncertainty))

        sdf_probs = sdf_probs * 2
        one_hot_probs = probs2one_hot(probs.detach()).float()
        sdf_seg_loss = rec_criterion(sdf_probs, sdf)
        rec_seg_loss = rec_criterion(rec_probs, image * one_hot_probs)
        unlabeled_weight = 0.1 * math.exp(-5 * math.pow((1 - epc / self.max_epc), 2))
        loss = unlabeled_weight * (sdf_consistency + seg_consistency + rec_consistency) \
               + (sdf_seg_loss + rec_seg_loss) * 0.1
        return loss

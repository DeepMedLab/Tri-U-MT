#!/usr/bin/env python3.6
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import warnings
from pathlib import Path
from functools import reduce
from operator import add, itemgetter
from shutil import copytree, rmtree
from typing import Any, Callable, Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from networks import weights_init
from dataloader import get_loaders
from utils import map_
from utils import dice_coef, dice_batch, save_images, tqdm_, haussdorf, iIoU
from utils import inter_sum, union_sum
from utils import probs2one_hot, probs2class
from utils import depth

from time import *

iter_step = 0
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def setup(args, n_class: int) -> Tuple[Any, Any, Any, List[List[Callable]], List[List[float]], Callable]:
    print("\n>>> Setting up")
    gpu_id: str = args.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    cpu: bool = args.cpu or not torch.cuda.is_available()
    device = torch.device("cpu") if cpu else torch.device("cuda")

    if args.weights:
        if cpu:
            net = torch.load(args.weights, map_location='cpu')
        else:
            net = torch.load(args.weights)
        print(f">> Restored weights from {args.weights} successfully.")
    else:
        def create_model(ema=False):
            UNet = getattr(__import__('my_network'), "SDFNet")
            net = UNet(args.modalities, n_class, args.l_rate, has_dropout=ema)
            model = net.cuda()
            if ema:
                for param in model.parameters():
                    param.detach_()
            return model

        student_net = create_model().to(device)
        teacher_net = create_model(True).to(device)
        student_net.apply(weights_init)
        teacher_net.apply(weights_init)

    student_net = student_net.cuda()
    teacher_net = teacher_net.cuda()

    # print(args.losses)
    list_losses = eval(args.losses)
    if depth(list_losses) == 1:  # For compatibility reasons, avoid changing all the previous configuration files
        list_losses = [list_losses]

    loss_fns: List[List[Callable]] = []
    for i, losses in enumerate(list_losses):
        print(f">> {i}th list of losses: {losses}")
        tmp: List[Callable] = []
        for loss_name, loss_params, _, _, fn, _ in losses:
            loss_class = getattr(__import__('losses'), loss_name)
            tmp.append(loss_class(**loss_params, fn=fn))
        loss_fns.append(tmp)

    loss_weights: List[List[float]] = [map_(itemgetter(5), losses) for losses in list_losses]

    scheduler = getattr(__import__('scheduler'), args.scheduler)(**eval(args.scheduler_params))

    return teacher_net, student_net, device, loss_fns, loss_weights, scheduler


def do_epoch(mode: str, teacher_net: Any, student_net: Any, device: Any, loaders: List[DataLoader], epc: int,
             list_loss_fns: List[List[Callable]], list_loss_weights: List[List[float]], C: int,
             savedir: str = "",
             metric_axis: List[int] = [1], compute_haussdorf: bool = False, compute_miou: bool = False,
             temperature: float = 1) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[None, Tensor]]:
    assert mode in ["train", "val"]

    if mode == "train":
        student_net.train()
        teacher_net.train()
        desc = f">> Training   ({epc})"
    elif mode == "val":
        student_net.eval()
        teacher_net.eval()
        desc = f">> Validation ({epc})"

    total_iteration: int = sum(len(loader) for loader in loaders)  # U
    total_images: int = sum(len(loader.dataset) for loader in loaders)  # D
    n_loss: int = max(map(len, list_loss_fns))

    all_dices: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    batch_dices: Tensor = torch.zeros((total_iteration, C), dtype=torch.float32, device=device)
    loss_log: Tensor = torch.zeros((total_iteration, n_loss), dtype=torch.float32, device=device)
    haussdorf_log: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    iiou_log: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    intersections: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    unions: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)

    few_axis: bool = len(metric_axis) <= 3

    done_img: int = 0
    done_batch: int = 0
    tq_iter = tqdm_(total=total_iteration, desc=desc)
    for i, (loader, loss_fns, loss_weights) in enumerate(zip(loaders, list_loss_fns, list_loss_weights)):
        L: int = len(loss_fns)

        for data in loader:
            # pro_begin_time = time()
            data[1:] = [e.to(device) for e in data[1:]]  # Move all tensors to device
            filenames, image, target = data[:3]
            assert not target.requires_grad
            labels = data[3:3 + L]
            volume_batch_r = image.repeat(1, 1, 1, 1)

            B = len(image)

            # Reset gradients
            if mode == "train":
                image = image + torch.clamp(torch.randn_like(image) * 0.02, -0.05, 0.05)
                student_net.zero_grad()
            _, pred_logits, sdf_probs, recon_probs = student_net(image)
            pred_probs: Tensor = F.softmax(temperature * pred_logits, dim=1)
            predicted_mask: Tensor = probs2one_hot(pred_probs.detach())  # Used only for dice computation
            assert not predicted_mask.requires_grad
            if mode == "train":
                K = 8
                ema_preds, ema_sdf_probs, ema_recon_probs = torch.zeros([3, K, B, 2, 256, 256]).cuda()
                for k in range(K):
                    # 在输入数据上加噪声
                    ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.02, -0.05, 0.05)
                    with torch.no_grad():
                        _, ema_preds[k], ema_sdf_probs[k], ema_recon_probs[k] = teacher_net(ema_inputs)

                # dimension [K, B, C, W, H]
                ema_preds = F.softmax(ema_preds, dim=2)
                uncertainty = -1.0 * torch.sum(ema_preds * torch.log2(ema_preds + 1e-6), dim=2,
                                               keepdim=True)
                weights = F.softmax(1 - uncertainty, dim=0)
                ema_probs = torch.sum(ema_preds * weights, dim=0)
                ema_mask: Tensor = probs2one_hot(ema_probs.detach())
                ema_seg_uncertainty = -1.0 * torch.sum(ema_probs * torch.log2(ema_probs + 1e-6), dim=1, keepdim=True)

                ema_sdf = ema_sdf_probs.mean(dim=0)
                ema_sdf_uncertainty = ema_sdf_probs.var(dim=0)

                ema_rec = ema_recon_probs.mean(dim=0)
                ema_rec_uncertainty = ema_recon_probs.std(dim=0)
            else:
                ema_mask = torch.zeros_like(predicted_mask).float()
                ema_sdf = torch.zeros_like(predicted_mask).float()
                ema_rec = torch.zeros_like(predicted_mask).float()
                ema_probs = torch.zeros_like(pred_probs).float()
                ema_seg_uncertainty = torch.zeros_like(predicted_mask).float()
                ema_sdf_uncertainty = torch.zeros_like(predicted_mask).float()
                ema_rec_uncertainty = torch.zeros_like(predicted_mask).float()

            assert len(loss_fns) == len(loss_weights) == len(labels)
            ziped = zip(loss_fns, labels, loss_weights)

            losses = [w * loss_fn(image, pred_probs, label, sdf_probs, recon_probs, ema_probs, ema_sdf, ema_rec,
                                  ema_seg_uncertainty, ema_rec_uncertainty, ema_sdf_uncertainty, epc) for loss_fn, label, w in ziped]
            loss = reduce(add, losses)
            assert loss.shape == (), loss.shape
            
            global iter_step
            # Backward
            if mode=="train":
                loss.backward()
                student_net.optimize()
                update_ema_variables(student_net, teacher_net, 0.99, iter_step)
                iter_step += 1

            # Compute and log metrics
            # loss_log[done_batch] = loss.detach()
            for j in range(len(loss_fns)):
                loss_log[done_batch, j] = losses[j].detach()

            sm_slice = slice(done_img, done_img + B)  # Values only for current batch

            dices: Tensor = dice_coef(predicted_mask, target)
            assert dices.shape == (B, C), (dices.shape, B, C)
            all_dices[sm_slice, ...] = dices

            if B > 1 and mode == "val":
                batch_dice: Tensor = dice_batch(predicted_mask, target)
                assert batch_dice.shape == (C,), (batch_dice.shape, B, C)
                batch_dices[done_batch] = batch_dice

            if compute_haussdorf:
                haussdorf_res: Tensor = haussdorf(predicted_mask, target)
                assert haussdorf_res.shape == (B, C)
                haussdorf_log[sm_slice] = haussdorf_res
            if compute_miou:
                IoUs: Tensor = iIoU(predicted_mask, target)
                assert IoUs.shape == (B, C), IoUs.shape
                iiou_log[sm_slice] = IoUs
                intersections[sm_slice] = inter_sum(predicted_mask, target)
                unions[sm_slice] = union_sum(predicted_mask, target)

            # Save images
            if savedir:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    predicted_class: Tensor = probs2class(pred_probs)
                    save_images(predicted_class, filenames, savedir, mode, epc)

            # Logging
            big_slice = slice(0, done_img + B)  # Value for current and previous batches

            dsc_dict = {f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis} if few_axis else {}

            hauss_dict = {f"HD{n}": haussdorf_log[big_slice, n].mean() for n in metric_axis} \
                if compute_haussdorf and few_axis else {}

            batch_dict = {f"bDSC{n}": batch_dices[:done_batch, n].mean() for n in metric_axis} \
                if B > 1 and mode == "val" and few_axis else {}

            miou_dict = {f"iIoU": iiou_log[big_slice, metric_axis].mean(),
                         f"mIoU": (intersections.sum(dim=0) / (unions.sum(dim=0) + 1e-10)).mean()} \
                if compute_miou else {}

            if len(metric_axis) > 1:
                mean_dict = {"DSC": all_dices[big_slice, metric_axis].mean()}
                if compute_haussdorf:
                    mean_dict["HD"] = haussdorf_log[big_slice, metric_axis].mean()
            else:
                mean_dict = {}

            stat_dict = {**miou_dict, **dsc_dict, **hauss_dict, **mean_dict, **batch_dict,
                         "loss": loss_log[:done_batch].mean()}
            nice_dict = {K: f"{v:.3f}" for (K, v) in stat_dict.items()}

            done_img += B
            done_batch += 1
            tq_iter.set_postfix({**nice_dict, "loader": str(i)})
            tq_iter.update(1)
    tq_iter.close()
    print(f"{desc} " + ', '.join(f"{K}={v}" for (K, v) in nice_dict.items()))

    if compute_miou:
        mIoUs: Tensor = (intersections.sum(dim=0) / (unions.sum(dim=0) + 1e-10))
        assert mIoUs.shape == (C,), mIoUs.shape
    else:
        mIoUs = None

    if not few_axis and False:
        print(f"DSC: {[f'{all_dices[:, n].mean():.3f}' for n in metric_axis]}")
        print(f"iIoU: {[f'{iiou_log[:, n].mean():.3f}' for n in metric_axis]}")
        if mIoUs:
            print(f"mIoU: {[f'{mIoUs[n]:.3f}' for n in metric_axis]}")

    return loss_log, all_dices, batch_dices, haussdorf_log, mIoUs


def run(args: argparse.Namespace) -> Dict[str, Tensor]:
    # vis_env = args.vis_env
    # vis = Visualizer(env=vis_env)
    vis = None
    n_class: int = args.n_class
    lr: float = args.l_rate
    savedir: str = args.workdir
    n_epoch: int = args.n_epoch
    val_f: int = args.val_loader_id

    loss_fns: List[List[Callable]]
    loss_weights: List[List[float]]
    teacher_net, student_net, device, loss_fns, loss_weights, scheduler = setup(args, n_class)
    train_loaders: List[DataLoader]
    val_loaders: List[DataLoader]
    train_loaders, val_loaders = get_loaders(args, args.dataset,
                                             args.batch_size, n_class,
                                             args.debug, args.in_memory)

    n_tra: int = sum(len(tr_lo.dataset) for tr_lo in train_loaders)  # Number of images in dataset
    l_tra: int = sum(len(tr_lo) for tr_lo in train_loaders)  # Number of iteration per epc: different if batch_size > 1
    n_val: int = sum(len(vl_lo.dataset) for vl_lo in val_loaders)
    l_val: int = sum(len(vl_lo) for vl_lo in val_loaders)
    n_loss: int = max(map(len, loss_fns))

    best_dice: Tensor = torch.zeros(1).to(device).type(torch.float32)
    best_epoch: int = 0
    metrics = {"val_dice": torch.zeros((n_epoch, n_val, n_class), device=device).type(torch.float32),
               "val_batch_dice": torch.zeros((n_epoch, l_val, n_class), device=device).type(torch.float32),
               "val_loss": torch.zeros((n_epoch, l_val, len(loss_fns[val_f])), device=device).type(torch.float32),
               "tra_dice": torch.zeros((n_epoch, n_tra, n_class), device=device).type(torch.float32),
               "tra_batch_dice": torch.zeros((n_epoch, l_tra, n_class), device=device).type(torch.float32),
               "tra_loss": torch.zeros((n_epoch, l_tra, n_loss), device=device).type(torch.float32)}
    if args.compute_haussdorf:
        metrics["val_haussdorf"] = torch.zeros((n_epoch, n_val, n_class), device=device).type(torch.float32)
    if args.compute_miou:
        metrics["val_mIoUs"] = torch.zeros((n_epoch, n_class), device=device).type(torch.float32)
        metrics["tra_mIoUs"] = torch.zeros((n_epoch, n_class), device=device).type(torch.float32)

    print("\n>>> Starting the training")
    for i in range(n_epoch):
        # Do training and validation loops
        tra_loss, tra_dice, tra_batch_dice, _, tra_mIoUs = do_epoch("train", teacher_net, student_net, device, train_loaders, i,
                                                                    loss_fns, loss_weights, n_class,
                                                                    savedir=savedir if args.save_train else "",
                                                                    metric_axis=args.metric_axis,
                                                                    compute_miou=args.compute_miou,
                                                                    temperature=args.temperature)
        with torch.no_grad():
            val_loss, val_dice, val_batch_dice, val_haussdorf, val_mIoUs = do_epoch("val", teacher_net, student_net, device, val_loaders, i,
                                                                                    [loss_fns[val_f]],
                                                                                    [loss_weights[val_f]],
                                                                                    n_class,
                                                                                    savedir=savedir,
                                                                                    metric_axis=args.metric_axis,
                                                                                    compute_haussdorf=args.compute_haussdorf,
                                                                                    compute_miou=args.compute_miou,
                                                                                    temperature=args.temperature)

        # Sort and save the metrics
        for k in metrics:
            assert metrics[k][i].shape == eval(k).shape, (metrics[k][i].shape, eval(k).shape, k)
            metrics[k][i] = eval(k)

        for k, e in metrics.items():
            np.save(Path(savedir, f"{k}.npy"), e.cpu().numpy())

        df = pd.DataFrame({"tra_loss": metrics["tra_loss"].mean(dim=(1, 2)).cpu().numpy(),
                           "val_loss": metrics["val_loss"].mean(dim=(1, 2)).cpu().numpy(),
                           "tra_dice": metrics["tra_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                           "val_dice": metrics["val_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                           "tra_batch_dice": metrics["tra_batch_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                           "val_batch_dice": metrics["val_batch_dice"][:, :, -1].mean(dim=1).cpu().numpy()})
        df.to_csv(Path(savedir, args.csv), float_format="%.4f", index_label="epoch")

        # Save model if better
        current_dice: Tensor = val_dice[:, args.metric_axis].mean()
        if current_dice > best_dice:
            best_epoch = i
            best_dice = current_dice
            if args.compute_haussdorf:
                best_haussdorf = val_haussdorf[:, args.metric_axis].mean()

            with open(Path(savedir, "best_epoch.txt"), 'w') as f:
                f.write(str(i))
            best_folder = Path(savedir, "best_epoch")
            if best_folder.exists():
                rmtree(best_folder)
            copytree(Path(savedir, f"iter{i:03d}"), Path(best_folder))
            torch.save(student_net, Path(savedir, "best.pkl"))

        loss_fns, loss_weights = scheduler(i, loss_fns, loss_weights)

        # if args.schedule and (i > (best_epoch + 20)):
        # if args.schedule and (i % (best_epoch + 20) == 0):  # Yeah, ugly but will clean that later
        #     for optimizer in optimizers:
        #         for param_group in optimizer.param_groups:
        #             lr *= 0.5
        #             param_group['lr'] = lr
        #             print(f'>> New learning Rate: {lr}')

        if i > 0 and not (i % 5):
            maybe_hauss = f', Haussdorf: {best_haussdorf:.3f}' if args.compute_haussdorf else ''
            print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_hauss}")

    # Because displaying the results at the end is actually convenient
    maybe_hauss = f', Haussdorf: {best_haussdorf:.3f}' if args.compute_haussdorf else ''
    print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_hauss}")
    for metric in metrics:
        if "val" in metric or "loss" in metric:  # Do not care about training values, nor the loss (keep it simple)
            print(f"\t{metric}: {metrics[metric][best_epoch].mean(dim=0)}")

    return metrics


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--weak_subfolder', type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--losses", type=str, required=True,
                        help="List of list of (loss_name, loss_params, bounds_name, bounds_params, fn, weight)")
    parser.add_argument("--folders", type=str, required=True,
                        help="List of list of (subfolder, transform, is_hot)")
    parser.add_argument("--network", type=str, required=True, help="The network to use")
    parser.add_argument("--grp_regex", type=str, required=True)
    parser.add_argument("--n_class", type=int, required=True)
    parser.add_argument("--gpu_id", type=str, required=True)
    parser.add_argument("--vis_env", type=str, required=True)
    parser.add_argument("--metric_axis", type=int, nargs='*', required=True, help="Classes to display metrics. \
        Display only the average of everything if empty")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--in_memory", action='store_true')
    parser.add_argument("--schedule", action='store_true')
    parser.add_argument("--use_sgd", action='store_true')
    parser.add_argument("--compute_haussdorf", action='store_true')
    parser.add_argument("--compute_miou", action='store_true')
    parser.add_argument("--save_train", action='store_true')
    parser.add_argument("--group", action='store_true', help="Group the patient slices together for validation. \
        Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.")
    parser.add_argument("--group_train", action='store_true', help="Group the patient slices together for training. \
        Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.")

    parser.add_argument('--n_epoch', nargs='?', type=int, default=200,
                        help='# of the epochs')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4,
                        help='Learning Rate')
    parser.add_argument('--temperature', type=float, default=1, help="Temperature for the softmax")
    parser.add_argument("--scheduler", type=str, default="DummyScheduler")
    parser.add_argument("--scheduler_params", type=str, default="{}")
    parser.add_argument("--modalities", type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--weights", type=str, default='', help="Stored weights to restore")
    parser.add_argument("--training_folders", type=str, nargs="+", default=["train"])
    parser.add_argument("--validation_folder", type=str, default="val")
    parser.add_argument("--val_loader_id", type=int, default=-1, help="""
                        Kinda housefiry at the moment. When we have several train loader (for hybrid training
                        for instance), wants only one validation loader. The way the dataloading creation is
                        written at the moment, it will create several validation loader on the same topfolder (val),
                        but with different folders/bounds ; which will basically duplicate the evaluation.
                        """)

    args = parser.parse_args()
    if args.metric_axis == []:
        args.metric_axis = list(range(args.n_class))
    print("\n", args)

    return args


if __name__ == '__main__':
    run(get_args())

#!/usr/bin/env python3.6

import argparse
from random import random, uniform, randint
from pathlib import Path
from multiprocessing.pool import Pool

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

import torch
import numpy as np
from tqdm import tqdm
from torch import einsum
from torch import Tensor
from functools import partial, reduce
from skimage.io import imsave
from PIL import Image, ImageOps
from scipy.spatial.distance import directed_hausdorff


colors = ["c", "r", "g", "b", "m", 'y', 'k', 'chartreuse', 'coral', 'gold', 'lavender',
          'silver', 'tan', 'teal', 'wheat', 'orchid', 'orange', 'tomato']

# functions redefinitions
tqdm_ = partial(tqdm, ncols=175,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def uc_(fn: Callable) -> Callable:
    return partial(uncurry, fn)


def uncurry(fn: Callable, args: List[Any]) -> Any:
    return fn(*args)


def id_(x):
    return x


def flatten_(to_flat: Iterable[Iterable[A]]) -> List[A]:
    return [e for l in to_flat for e in l]


def depth(e: List) -> int:
    """
    Compute the depth of nested lists
    """
    if type(e) == list and e:
        return 1 + depth(e[0])

    return 0


def compose(fns, init):
    return reduce(lambda acc, f: f(acc), fns, init)


def compose_acc(fns, init):
    return reduce(lambda acc, f: acc + [f(acc[-1])], fns, [init])


# fns
def soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->bc", a)[..., None]


def batch_soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->c", a)[..., None]


def soft_centroid(a: Tensor) -> Tensor:
    b, c, w, h = a.shape

    ws, hs = map_(lambda e: Tensor(e).to(a.device).type(torch.float32), np.mgrid[0:w, 0:h])
    assert ws.shape == hs.shape == (w, h)

    flotted = a.type(torch.float32)
    tot = einsum("bcwh->bc", flotted) + 1e-10
    assert tot.dtype == torch.float32

    cw = einsum("bcwh,wh->bc", flotted, ws) / tot
    ch = einsum("bcwh,wh->bc", flotted, hs) / tot
    assert cw.dtype == ch.dtype == torch.float32

    res = torch.stack([cw, ch], dim=2)
    assert res.shape == (b, c, 2)
    assert res.dtype == torch.float32

    return res


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


# # Metrics and shitz
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> float:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bcwh->bc")
dice_batch = partial(meta_dice, "bcwh->c")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a & b


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a | b


def inter_sum(a: Tensor, b: Tensor) -> Tensor:
    return einsum("bcwh->bc", intersection(a, b).type(torch.float32))


def union_sum(a: Tensor, b: Tensor) -> Tensor:
    return einsum("bcwh->bc", union(a, b).type(torch.float32))


def haussdorf(preds: Tensor, target: Tensor) -> Tensor:
    assert preds.shape == target.shape
    assert one_hot(preds)
    assert one_hot(target)

    B, C, _, _ = preds.shape

    res = torch.zeros((B, C), dtype=torch.float32, device=preds.device)
    n_pred = preds.cpu().numpy()
    n_target = target.cpu().numpy()

    for b in range(B):
        if C == 2:
            res[b, :] = numpy_haussdorf(n_pred[b, 0], n_target[b, 0])
            continue

        for c in range(C):
            res[b, c] = numpy_haussdorf(n_pred[b, c], n_target[b, c])

    return res


def numpy_haussdorf(pred: np.ndarray, target: np.ndarray) -> float:
    assert len(pred.shape) == 2
    assert pred.shape == target.shape

    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


def iIoU(pred: Tensor, target: Tensor) -> Tensor:
    IoUs = inter_sum(pred, target) / (union_sum(pred, target) + 1e-10)
    assert IoUs.shape == pred.shape[:2]

    return IoUs


# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    # assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))
    assert len(seg.shape) == 3, seg.shape

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def fast_np_class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        return fast_np_class2one_hot(seg[None, ...], C)[0]
    assert set(np.unique(seg)).issubset(list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = np.zeros((b, C, w, h), dtype=np.int32)
    np.put_along_axis(res, seg[:, None, ...], 1, axis=1)
    assert res.shape == (b, C, w, h)
    assert np.all(res.sum(axis=1) == 1)
    assert set(np.unique(res)).issubset([0, 1])

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    # assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


# Misc utils
def save_images(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int) -> None:
    b, w, h = segs.shape  # Since we have the class numbers, we do not need a C axis

    for seg, name in zip(segs, names):
        save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        imsave(str(save_path), seg.cpu().numpy())


def augment(*arrs: Union[np.ndarray, Image.Image], rotate_angle: float = 45,
            flip: bool = True, mirror: bool = True,
            rotate: bool = True, scale: bool = False) -> List[Image.Image]:
    imgs: List[Image.Image] = map_(Image.fromarray, arrs) if isinstance(arrs[0], np.ndarray) else list(arrs)

    if flip and random() > 0.5:
        imgs = map_(ImageOps.flip, imgs)
    if mirror and random() > 0.5:
        imgs = map_(ImageOps.mirror, imgs)
    if rotate and random() > 0.5:
        angle: float = uniform(-rotate_angle, rotate_angle)
        imgs = map_(lambda e: e.rotate(angle), imgs)
    if scale and random() > 0.5:
        scale_factor: float = uniform(1, 1.2)
        w, h = imgs[0].size  # Tuple[int, int]
        nw, nh = int(w * scale_factor), int(h * scale_factor)  # Tuple[int, int]

        # Resize
        imgs = map_(lambda i: i.resize((nw, nh)), imgs)

        # Now need to crop to original size
        bw, bh = randint(0, nw - w), randint(0, nh - h)  # Tuple[int, int]

        imgs = map_(lambda i: i.crop((bw, bh, bw + w, bh + h)), imgs)
        assert all(i.size == (w, h) for i in imgs)

    return imgs


def augment_arr(*arrs_a: np.ndarray) -> List[np.ndarray]:
    arrs = list(arrs_a)  # manoucherie type check

    if random() > 0.5:
        arrs = map_(np.flip, arrs)
    if random() > 0.5:
        arrs = map_(np.fliplr, arrs)
    # if random() > 0.5:
    #     orig_shape = arrs[0].shape

    #     angle = random() * 90 - 45
    #     arrs = map_(lambda e: sp.ndimage.rotate(e, angle, order=1), arrs)

    #     arrs = get_center(orig_shape, *arrs)

    return arrs


def get_center(shape: Tuple, *arrs: np.ndarray) -> List[np.ndarray]:
    def g_center(arr):
        if arr.shape == shape:
            return arr

        dx = (arr.shape[0] - shape[0]) // 2
        dy = (arr.shape[1] - shape[1]) // 2

        if dx == 0 or dy == 0:
            return arr[:shape[0], :shape[1]]

        res = arr[dx:-dx, dy:-dy][:shape[0], :shape[1]]  # Deal with off-by-one errors
        assert res.shape == shape, (res.shape, shape, dx, dy)

        return res

    return [g_center(arr) for arr in arrs]

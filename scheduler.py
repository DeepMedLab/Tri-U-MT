#!/usr/bin/env python3.7

from typing import Any, Callable, List, Tuple
from operator import add
from functools import partial

from utils import map_, uc_


class DummyScheduler(object):
    def __call__(self, epoch: int, loss_fns: List[List[Callable]], loss_weights: List[List[float]]) \
            -> Tuple[float, List[List[Callable]], List[List[float]]]:
        return loss_fns, loss_weights

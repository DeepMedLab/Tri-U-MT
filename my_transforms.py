from operator import itemgetter
from functools import partial
import torch
import numpy as np
from torchvision import transforms
from utils import class2one_hot


class PNG_Transform():
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.convert('L')
        img_array = np.array(img, dtype='float')[np.newaxis, ...]
        img_array /= 255
        return torch.tensor(img_array, dtype=torch.float32)


class GT_Transform():
    def __init__(self, n_class):
        self.n_class = n_class

    def __call__(self, img):
        img = img.convert('L')
        img_array = np.array(img)[np.newaxis, ...]
        img_tensor = torch.tensor(img_array, dtype=torch.int64)
        img_tensor = class2one_hot(img_tensor, C=self.n_class)
        op = itemgetter(0)
        img_tensor = op(img_tensor)
        return img_tensor


class DUMMY_Transfrom():
    def __init__(self, n_class):
        self.n_class = n_class

    def __call__(self, img):
        img = np.array(img)
        dummy_tensor = torch.zeros((self.n_class, *(img.shape)), dtype=torch.int64)
        return dummy_tensor

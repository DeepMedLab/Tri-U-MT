import os
from PIL import Image
import numpy
import torch
from torch import einsum
import medpy.metric.binary as mb
from numpy import mean, std
from xlwt import Workbook
from pathlib import Path
from utils import dice_coef, probs2class, probs2one_hot, dice_batch


def intersection(a, b):
    return a & b


def meta_dice(sum_str: str, label, pred, smooth: float = 1e-8) -> float:
    inter_size = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)
    dices = (2 * inter_size + smooth) / (sum_sizes + smooth)
    return dices


def comp(x):
    return int(x.split(".")[0].split("_")[-1])


source_dir = ""
img_set = set()
test = []

for root, dirs, filenames in os.walk(source_dir):
    for filename in filenames:
        img_set.add(filename)

folders = []
path = ""
for root, dirs, files in os.walk(path):
    for curDir in dirs:
        dirsInCurdir = os.listdir(root + "/" + curDir)
        if "best.pkl" in dirsInCurdir:
            folders.append(root + "/" + curDir)

file = Workbook(encoding='utf-8')
table = file.add_sheet('data')
table.write(0, 1, "dice")
table.write(0, 2, "asd")
table.write(0, 3, "hd")
table.write(0, 4, "precision")
table.write(0, 5, "recall")
table.write(0, 6, "spe")
table.write(0, 7, "ja")
table.write(0, 8, "acc")
table.write(0, 9, "RVD")
table.write(0, 10, "FNR")
folders.append(path)
# 3D result
for index, folder in enumerate(folders):
    print(folder)
    asd_for_epoch = []
    dice_for_epoch = []
    hd_for_epoch = []
    precision_for_epoch = []
    recall_for_epoch = []
    specificity_for_epoch = []
    ji_for_epoch = []
    acc_for_epoch = []
    rvd_for_epoch = []
    fnr_for_epoch = []

    for i in range(90, 94):
        dice_for_image = []
        asd_for_image = []
        hd_for_image = []
        precision_for_image = []
        recall_for_image = []
        specificity_for_image = []
        ji_for_image = []
        acc_for_image = []
        rvd_for_image = []
        fnr_for_image = []

        result_dir = folder
        patients_set = set()
        for img in img_set:
            patient_names = "_".join(img.split("_")[:-2])
            patients_set.add(patient_names)

        for patient in patients_set:
            patient_slices = [patient_slc for patient_slc in img_set if patient in patient_slc]
            patient_slices.sort(key=comp)
            result_tensor = torch.zeros((len(patient_slices), 256, 256))
            gt_tensor = torch.zeros((len(patient_slices), 256, 256))
            slice_index = 0
            for img in patient_slices:
                gt = Image.open(source_dir + "/" + img)
                result = Image.open(result_dir + "/" + img)

                gt = torch.from_numpy(numpy.array(gt))
                result = torch.from_numpy(numpy.array(result))
                result_tensor[slice_index] = result
                gt_tensor[slice_index] = gt
                slice_index += 1
            asd = mb.obj_asd(result_tensor.numpy(), gt_tensor.numpy())
            asd_for_image.append(asd)
            dice = mb.dc(result_tensor.numpy(), gt_tensor.numpy())
            dice_for_image.append(dice)
            precision = mb.precision(result_tensor.numpy(), gt_tensor.numpy())
            precision_for_image.append(precision)
            recall = mb.recall(result_tensor.numpy(), gt_tensor.numpy())
            recall_for_image.append(recall)
            specificity = mb.specificity(result_tensor.numpy(), gt_tensor.numpy())
            specificity_for_image.append(specificity)
            ji = mb.jc(result_tensor.numpy(), gt_tensor.numpy())
            ji_for_image.append(ji)

            tp = einsum("cwh->", result_tensor * gt_tensor)
            fn = einsum("cwh->", (1 - result_tensor) * gt_tensor)
            fp = einsum("cwh->", result_tensor * (1 - gt_tensor))
            tn = einsum("cwh->", (1 - result_tensor) * (1 - gt_tensor))
            acc = (tp + tn) / (tp + fp + tn + fn)
            acc_for_image.append(acc)
            rvd = (fn - fp) / (fn + tp)
            rvd_for_image.append(rvd)
            fnr = fn / (fn + tp + fp)
            fnr_for_image.append(fnr)

        asd_for_epoch.append(asd_for_image)
        dice_for_epoch.append(dice_for_image)
        hd_for_epoch.append(hd_for_image)
        precision_for_epoch.append(precision_for_image)
        recall_for_epoch.append(recall_for_image)
        specificity_for_epoch.append(specificity_for_image)
        ji_for_epoch.append(ji_for_image)
        acc_for_epoch.append(acc_for_image)
        rvd_for_epoch.append(rvd_for_image)
        fnr_for_epoch.append(fnr_for_image)
        print(dice_for_epoch)

    folder_path = Path(folder)
    table.write(index + 1, 0, folder_path.parent.name + " " + folder_path.name)
    print(mean(dice_for_epoch, axis=1))
    table.write(index + 1, 1, "%f(%f)" % (mean(dice_for_epoch), mean(std(dice_for_epoch, axis=1))))
    table.write(index + 1, 2, "%f(%f)" % (mean(asd_for_epoch), mean(std(asd_for_epoch, axis=1))))
    table.write(index + 1, 3, "%f(%f)" % (mean(hd_for_epoch), mean(std(hd_for_epoch, axis=1))))
    table.write(index + 1, 4, "%f(%f)" % (mean(precision_for_epoch), mean(std(precision_for_epoch, axis=1))))
    table.write(index + 1, 5, "%f(%f)" % (mean(recall_for_epoch), mean(std(recall_for_epoch, axis=1))))
    table.write(index + 1, 6, "%f(%f)" % (mean(specificity_for_epoch), mean(std(specificity_for_epoch, axis=1))))
    table.write(index + 1, 7, "%f(%f)" % (mean(ji_for_epoch), mean(std(ji_for_epoch, axis=1))))
    table.write(index + 1, 8, "%f(%f)" % (mean(acc_for_epoch), mean(std(acc_for_epoch, axis=1))))
    table.write(index + 1, 9, "%f(%f)" % (mean(rvd_for_epoch), mean(std(rvd_for_epoch, axis=1))))
    table.write(index + 1, 10, "%f(%f)" % (mean(fnr_for_epoch), mean(std(fnr_for_epoch, axis=1))))

    file.save('new_promise_30_mtx.xls')

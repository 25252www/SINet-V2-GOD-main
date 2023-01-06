import os
import torch
import imageio
import csv
from tqdm import tqdm

# res, gt都是gpu上的tensor


def get_confusion_matrix(res, gt, num_classes):
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = torch.sum((res == i) & (gt == j))
    return confusion_matrix

# 通过混淆矩阵计算iou
# confusion_matrix是gpu上的tensor


def cal_iou(confusion_matrix, num_classes):
    iou = torch.zeros(num_classes).cuda()
    for i in range(num_classes):
        iou[i] = confusion_matrix[i, i] / (torch.sum(confusion_matrix[i, :]) + torch.sum(
            confusion_matrix[:, i]) - confusion_matrix[i, i])
    return iou


if __name__ == '__main__':
    CLASSES = ['background', 'frontground']
    num_classes = len(CLASSES)
    res_folder = '/home/liuxiangyu/SINet-V2-GOD-main/res/SINet_V2'
    gt_folder = '/home/liuxiangyu/SINet-V2-GOD-main/Dataset/ValDataset/GT'

    res_paths = [os.path.join(res_folder, f) for f in os.listdir(res_folder)]
    gt_paths = [os.path.join(gt_folder, f) for f in os.listdir(gt_folder)]

    res_paths = sorted(res_paths)
    gt_paths = sorted(gt_paths)

    thresholds = [i for i in range(1, 256)]

    csv_file = open('/home/liuxiangyu/SINet-V2-GOD-main/ious.csv', 'w', newline='')
    writer = csv.writer(csv_file)

    # 进度条
    for threshold in tqdm(thresholds):
        confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
        for res_path, gt_path in zip(res_paths, gt_paths):
            # imageio读取res和gt，转换成torch.tensor
            res = torch.tensor(imageio.imread(res_path))
            # 读取灰度图
            gt = torch.tensor(imageio.imread(gt_path, as_gray=True))
            gt_binary = (gt == 255).int()
            # 按threshold将res转换只包含0和1的二值图
            res_binary = (res >= threshold).int()
            confusion_matrix += get_confusion_matrix(
                res_binary, gt_binary, num_classes)
        iou = cal_iou(confusion_matrix, num_classes)
        # 保存ious到csv中
        writer.writerow([threshold, iou[1].item()])

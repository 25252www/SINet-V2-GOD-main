import pathlib
import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

segmentationclassdir = '/home/liuxiangyu/dataset/VOCdevkit/VOC2012/SegmentationClass'
gtdir = '/home/liuxiangyu/SINet-V2-GOD-main/Dataset/ValDataset/GT'

border_px = np.asarray([192, 224, 224])
black_px = np.asarray([0, 0, 0])
white_px = np.asarray([255, 255, 255])

if __name__ == "__main__":
    if not os.path.exists(gtdir):
        os.makedirs(gtdir)
    rgbimgpaths = glob.glob(os.path.join(segmentationclassdir, '*.png'))
    for i, rgbimgpath in enumerate(tqdm(rgbimgpaths)):
        Image = cv2.imread(rgbimgpath)
        (row, col, _) = Image.shape
        for r in range(row):
            for c in range(col):
                px = Image[r][c]
                if any(px != black_px):
                    if all(px == border_px):
                        Image[r][c] = black_px
                    else:
                        Image[r][c] = white_px
        cv2.imwrite(os.path.join(gtdir, os.path.basename(rgbimgpath)), Image)

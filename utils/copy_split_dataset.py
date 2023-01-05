import os
import shutil

traintxt = '/home/liuxiangyu/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
valtxt = '/home/liuxiangyu/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
trainvaltxt = '/home/liuxiangyu/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt'
imgfromdir = '/home/liuxiangyu/dataset/VOCdevkit/VOC2012/JPEGImages'

valGTdir = '/home/liuxiangyu/SINet-V2-GOD-main/Dataset/ValDataset/GT'
valImgsdir = '/home/liuxiangyu/SINet-V2-GOD-main/Dataset/ValDataset/Imgs'

trainGTdir = '/home/liuxiangyu/SINet-V2-GOD-main/Dataset/TrainDataset/GT'
trainImgsdir = '/home/liuxiangyu/SINet-V2-GOD-main/Dataset/TrainDataset/Imgs'

if __name__ == "__main__":
    if not os.path.exists(valImgsdir):
        os.makedirs(valImgsdir)
    for line in open(trainvaltxt,'r'):
        line = line[:-1]
        imgfrompath = os.path.join(imgfromdir,line+'.jpg')
        shutil.copy(imgfrompath,valImgsdir)

    # split train from ValDataset
    if not os.path.exists(trainGTdir):
        os.makedirs(trainGTdir)
    if not os.path.exists(trainImgsdir):
        os.makedirs(trainImgsdir)
    for line in open(traintxt,'r'):
        line = line[:-1]
        shutil.move(os.path.join(valGTdir,line+'.png'),trainGTdir)
        shutil.move(os.path.join(valImgsdir,line+'.jpg'),trainImgsdir)
        
    
    
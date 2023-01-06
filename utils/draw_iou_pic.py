import csv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    csv_path = '/home/liuxiangyu/SINet-V2-GOD-main/ious-god.csv'
    thresholds = []
    ious = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            thresholds.append(int(row[0]))
            ious.append(float(row[1]))
    print("mean iou: ", sum(ious) / len(ious))
    # 画图
    plt.plot(thresholds, ious)
    plt.show()
    # 保存
    plt.savefig('/home/liuxiangyu/SINet-V2-GOD-main/ious-god.png')


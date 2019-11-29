import numpy as np
import cv2
import random
import os
# calculate means and std
from tqdm import tqdm_notebook

train_path = './eye_segmentation_dataset/train'

CNum = 10000  # 挑选多少图片进行计算

img_h, img_w = 224, 224
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []
img_files = []

for root, dirs, files in os.walk(train_path):
    for file_name in files:
        img_files.append(root+'/'+file_name)
random.shuffle(img_files)  # shuffle , 随机挑选图片

for i in tqdm_notebook(range(len(img_files))):
    img = cv2.imread(img_files[i])
    print(img_files[i])
    if img is None:
        print('-1')
    img = cv2.resize(img, (img_h, img_w))
    img = img[:, :, :, np.newaxis]

    imgs = np.concatenate((imgs, img), axis=3)
#         print(i)

imgs = imgs.astype(np.float32) / 255.

for i in tqdm_notebook(range(3)):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
means.reverse()  # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

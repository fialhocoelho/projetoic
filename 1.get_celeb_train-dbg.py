# -*- coding: utf-8 -*- 
import os
import cv2
import imageio
from scipy import misc
from PIL import Image
import numpy as np

# --- alterado -------------------
print(os.system('pwd'))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# --------------------------------

dataset_dir = 'datasets' # datasets
dataset = 'data_crop_512_jpg'
train_dir = 'train'
test_dir = 'val_test'
val_dir = 'val'
resize_to = 256
scale = 4

if not os.path.exists(os.path.join(dataset_dir, train_dir)):
    os.mkdir(os.path.join(dataset_dir, train_dir))
if not os.path.exists(os.path.join(dataset_dir, test_dir)):
    os.mkdir(os.path.join(dataset_dir, test_dir))
if not os.path.exists(os.path.join(dataset_dir, val_dir)):
    os.mkdir(os.path.join(dataset_dir, val_dir))

i = 1
fileList = os.listdir(os.path.join(dataset_dir, dataset))
for index, file in enumerate(fileList):
    imgPath = os.path.join(dataset_dir, dataset, file)
    if os.path.isdir(imgPath):
        continue
    if(i > 1000):
        i = i + 1
        raise Exception('Forced finish')
    print("procesing " + file + " " + str(i+1) + '/' + str(len(fileList)))
    
    #img = cv2.imread(imgPath)
    #print("DBG - img sizes 1", img.shape)
    #imgplot = plt.imshow(img)
    #plt.show()
    
    img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
    #print("DBG - img sizes 2", img.shape)
    #img = misc.imresize(img, (resize_to, resize_to), interp='bilinear')
    img = np.array(Image.fromarray(img).resize((resize_to, resize_to), Image.BILINEAR))
    #print("DBG - img sizes 3", img.shape)
    size = int(img.shape[0] / scale)
    #print("DBG - size", size)
    #resizeImg = misc.imresize(img, (size, size), interp='bilinear')
    resizeImg = np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR))
    #print("DBG - img sizes 4", resizeImg.shape)
    #finalImg = misc.imresize(resizeImg, (img.shape[0], img.shape[0]), interp='bilinear')
    finalImg = np.array(Image.fromarray(resizeImg).resize((img.shape[0], img.shape[0]), Image.BILINEAR))
    #print("DBG - img sizes 5", finalImg.shape)
    combineImg = Image.new('RGB', (img.shape[0]*2, img.shape[0]))
    combineImg.paste(Image.fromarray(finalImg), (0,0))
    combineImg.paste(Image.fromarray(img), (img.shape[0]+1,0))
    savePath = ""
    # sample ratio of train:test:val is 6:2:2
    if index % 10 < 6:
        savePath = os.path.join(dataset_dir, train_dir, file)
    elif index % 10 < 8:
        savePath = os.path.join(dataset_dir, test_dir, file)
    else:
        savePath = os.path.join(dataset_dir, val_dir, file)
    misc.imsave(savePath, combineImg)
    imageio.imwrite(savePath, combineImg)

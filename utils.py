import pandas as pd
import cv2
import keras
import numpy as np
import os
import random
from PIL import Image
import scipy.misc

def normalize_dataframe(data, num_bins=23):
    avg_samples_per_bin = len(data['steering']) / num_bins
    hist, bins = np.histogram(data['steering'], num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) * 0.5

    # Drop random straight steering angles.
    keep_probs = []
    target = avg_samples_per_bin * .3
    for i in range(num_bins):
        if hist[i] < target:
            keep_probs.append(1.0)
        else:
            keep_probs.append(1.0 / (hist[i] / target))

    # Delete from X and y with probability 1 - keep_probs[j].
    remove_list = []
    for i in range(len(data['steering'])):
        angle = data['steering'][i]
        for j in range(num_bins):
            if angle > bins[j] and angle <= bins[j + 1]:
                if np.random.rand() > keep_probs[j]:
                    remove_list.append(i)

    data.drop(data.index[remove_list], inplace=True)
    return data


def tint_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype= np.float64)
    random_bright = 0.5  + np.random.uniform()
    img[:,:,2] = img[:,:,2]*random_bright
    img[:,:,2][img[:,:,2]>255] = 255
    img = np.array(img,dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return resize_crop(img)

def resize_crop(img):
    img  = np.array(img,np.float32)
    img = img[35:135,:]
    img = scipy.misc.imresize(img,(66,200))
    return img


def flip(path):
#If all csv have the same # of columns
    img = cv2.imread(path)
    img  = cv2.flip(img,0)
    return resize_crop(img)

def jitter_image(path,steering):
    img = cv2.imread(path)
    rows, cols,_ = img.shape
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange*np.random.uniform() - transRange/2
    transY = numPixels * np.random.uniform() -numPixels/2
    transMat = np.float32([[1,0,transX],[0,1,transY]])
    steering = steering + transX / transRange * 2 * valPixels
    img = cv2.warpAffine(img,transMat,(cols,rows))
    return resize_crop(img), steering


def load_image(path):
    image = Image.open(path)
    return resize_crop(image)

def load_data(file_name):
    data= pd.read_csv(file_name)
    data.iloc[:,:3] = data.iloc[:,:3].astype(str)
    data.iloc[:,3] = data.iloc[:,3].astype(float)
    data = normalize_dataframe(data)
    images = data.iloc[:,:3]
    angles = data.iloc[:,3]
    return images, angles

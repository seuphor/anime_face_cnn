import numpy as np
import sys
import os
from PIL import Image
import cv2
from .params import *
import matplotlib.pyplot as plt

import numpy as np

class batch_selection():
    def __init__(self):
        self.current_index = 0
    
    def next_batch(self, x, y, batch_size):
        assert(x.shape[0] == y.shape[0])
        start = self.current_index
        
        if start == 0:
            perm = np.arange(x.shape[0])
            np.random.shuffle(perm)
            self._x = x[perm]
            self._y = y[perm]
        
        if (start + batch_size) >= x.shape[0]:
            self.current_index = 0
            return self._x[start:x.shape[0]], self._y[start:x.shape[0]]
        
        else:
            self.current_index += batch_size
            end = self.current_index
            return self._x[start:end], self._y[start:end]

    def next_batch_single(self, x, batch_size):
        #assert(x.shape[0] == y.shape[0])
        start = self.current_index
        
        if start == 0:
            perm = np.arange(x.shape[0])
            np.random.shuffle(perm)
            self._x = x[perm]
            #self._y = y[perm]
        
        if (start + batch_size) >= x.shape[0]:
            self.current_index = 0
            return self._x[start:x.shape[0]]
        
        else:
            self.current_index += batch_size
            end = self.current_index
            return self._x[start:end]


def img2array(img,dim):
     
    if dim == 12:    
        if img.size[0] != 12 or img.size[1] != 12:
            img = img.resize((12,12))
        img = np.asarray(img).astype(np.float32)/255
        if img.shape[2] == 4:
            img = img[:,:,:3]

    if dim == 24:    
        if img.size[0] != 24 or img.size[1] != 24:
            img = img.resize((24,24))
        img = np.asarray(img).astype(np.float32)/255
        if img.shape[2] >= 4:
            img = img[:,:,:3]

    if dim == 48:
        if img.size[0] != 48 or img.size[1] != 48:
            img = img.resize((48,48))
        img = np.asarray(img).astype(np.float32)/255
        if img.shape[2] >= 4:
            img = img[:,:,:3]

    return img

def show_imgarr(arr):
    img_arr = (arr * 255).astype(np.uint8)
    plt.imshow(img_arr)
    plt.show()
    pass

def face_crop(path):
    '''
        Take the path location of a image as parameter and returns
        an array of crop location (height_location, width_location, max_height, max_width)

        SAMPLE: face_crop('example.jpg')
        return: ([height_loc, width_loc]), height, weight
    '''
    pic = cv2.imread(path, 1)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

    counter, height, height_end, width_end= 0, 0, 0, 0
    
    for height_row in pic:
        width = 0
        for width_row in height_row:
            if width_row[0] <= 5 and width_row[1] >= 250 and width_row[2] <= 5 and counter == 0:
                location_arr = np.array([height, width])
                counter += 1
            if width_row[0] <= 5 and width_row[1] >= 250 and width_row[2] <= 5:
                if height_end < height:
                    height_end = height
                if width_end < width:
                    width_end = width
            width += 1
        height += 1

    height, width = pic.shape[0], pic.shape[1]
    return np.array([location_arr[0], location_arr[1], height_end, width_end]), height, width

def face_crop_scale(path):
    '''
        Take the path location of a image as parameter and returns
        an array of crop location (height_location, width_location, max_height, max_width)

        SAMPLE: face_crop('example.jpg')
        return: ([height_loc, width_loc]), height, weight
    '''
    img = Image.open(path)

    if img.size[0] > p_neg_max_bound or img.size[1] > p_neg_max_bound:
        ratio = p_neg_max_bound / max(img.size[0], img.size[1])
        resized = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))
    else:
        resized = img
    
    #adjust_scale = int(resized.size[0] * 1/160.0)
    pic = np.asarray(resized)
    img.close()
    counter, height, height_end, width_end= 0, 0, 0, 0
    
    for height_row in pic:
        width = 0
        for width_row in height_row:
            if width_row[0] <= 5 and width_row[1] >= 250 and width_row[2] <= 5 and counter == 0:
                location_arr = np.array([height, width])
                counter += 1
            if width_row[0] <= 5 and width_row[1] >= 250 and width_row[2] <= 5:
                if height_end < height:
                    height_end = height
                if width_end < width:
                    width_end = width
            width += 1
        height += 1

    height, width = pic.shape[0], pic.shape[1]


    #print(adjust_scale)
    return np.array([max(0, location_arr[1]), location_arr[0], width_end, height_end])


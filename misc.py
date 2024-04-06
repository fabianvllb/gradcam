#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:50:54 2021

@author: siva, derro
"""

from os import listdir
from os.path import join, splitext, dirname, realpath
import numpy as np
""" from scipy.signal import medfilt """
import cv2
""" from pycimg import CImg """
from base import onnxbase
import matplotlib.pyplot as plt


cppround = lambda x: np.floor(x + 0.5).astype(np.int)
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
""" cv2cimg = lambda x : CImg(np.expand_dims(np.transpose(x, (2, 0, 1)), axis=1)) """
cimg2cv = lambda x : np.transpose(np.squeeze(np.clip(x.asarray(), 0, 255).astype(np.uint8), axis=1), (1, 2, 0))


rsz = onnxbase(join(dirname(realpath(__file__)), "resize.ort")) # loads the onnx model that resizes images


def checkimage(img):
    return img is not None and img.size > 0 and img.ndim == 3 and img.dtype == np.uint8


def loadimage(filename):
    img = cv2.imread(filename)
    assert checkimage(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def saveimage(filename, img):
    assert checkimage(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(filename, img)


def isimage(filename):
    return splitext(filename)[1].lower() in [".png", ".jpg", ".jpeg", ".bmp"]


def showimage(img):
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.show()


def simulatejpg(img, quality=100):
    assert quality > 0 and quality <= 100
    assert checkimage(img)
    success, tmp = cv2.imencode(".jpg", img,
        [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert success
    return cv2.imdecode(tmp, 1)


""" def cimgrotate(img, degrees):
    h, w = img.shape[:2]
    tmp = cv2cimg(img)
    tmp.rotate(-degrees, 1) # cimg requires the opposite angle as opencv
    tmp = cimg2cv(tmp)
    t = (tmp.shape[0] >> 1) - (h >> 1)
    l = (tmp.shape[1] >> 1) - (w >> 1)
    return tmp[t:t+h, l:l+w, :] """


""" def cimgresize(img, height, width, areainterp=False):
    if img.shape[0] == height and \
       img.shape[1] == width:
        return img
    elif img.ndim == 2:
        tmp = np.expand_dims(img, axis=2)
        tmp = cimgresize(tmp, height, width, areainterp)
        tmp = np.squeeze(tmp, axis=2)
    elif img.ndim == 3:
        tmp = cv2cimg(img)
        tmp.resize(width, height, 1, img.shape[2], 2 if areainterp else 3) # 3 = linear
        tmp = cimg2cv(tmp)
    else:
        raise Exception("Unexpected number of dims")
    return tmp """


def preprocessimage(img, height, width, center=0.0, scale=1.0):
    tmp = img.astype(np.float32)
    tmp = np.expand_dims(tmp, axis=0)
    if img.shape[1] != height or img.shape[2] != width:
        tmp = rsz.predict([tmp, np.array([[height, width]], dtype=np.float32)])
        tmp = np.floor(tmp + 0.5)
    if center != 0.0: tmp -= center
    if scale != 1.0: tmp /= scale
    return tmp


def forceimageratio(img, wh_ratio, fill_value=0):
    h, w = img.shape[:2]
    ratio = float(w) / float(h)
    if ratio > wh_ratio: # original image is wider than desired -> need vertical pad
        h2 = int(np.round(w / wh_ratio))
        assert h2 >= h # if this throws an error, there must be a bug
        if h2 > h:
            b = h2 - h
            t = b >> 1
            b -= t
            img = cv2.copyMakeBorder(img, t, b, 0, 0,
                cv2.BORDER_CONSTANT, value=fill_value)
    elif ratio < wh_ratio: # original image is taller than desired -> need horizontal pad
        w2 = int(np.round(h * wh_ratio))
        assert w2 >= w # if this throws an error, there must be a bug
        if w2 > w:
            r = w2 - w
            l = r >> 1
            r -= l
            img = cv2.copyMakeBorder(img, 0, 0, l, r,
                cv2.BORDER_CONSTANT, value=fill_value)
    return img


def processfolder(processor, folderin, folderout, processfilefun, ext=None, verbose=True, **kwargs):
    fileout = None
    for filename in listdir(folderin):
        if not isimage(filename): continue
        if verbose: print(filename, flush=True)
        filein = join(folderin, filename)
        if folderout is not None:
            fileout = join(folderout, filename)
            if ext is not None: fileout = splitext(fileout)[0] + ext
        if not processfilefun(processor, filein, fileout, **kwargs):
            print("Error!", flush=True)
    return


""" def medianfilter(x, order):
    assert order > 2 and (order >> 1) << 1 != order # must be odd
    assert x.size >= order
    y = medfilt(x, order)
    half = order >> 1
    tmp = half + 1
    last = x.size - 1
    for i in range(half):
        y[i] = np.median(x[:tmp])
        y[last - i] = np.median(x[-tmp:])
        tmp += 1
    if y.dtype != x.dtype: y = y.astype(x.dtype)
    return y """
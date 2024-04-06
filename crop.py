#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:58:07 2021

@author: siva, derro
"""

import sys
from os.path import exists
import numpy as np
import cv2
from base import tflitebase, onnxbase
from misc import loadimage, saveimage, isimage, showimage, preprocessimage, processfolder, cimgrotate, cppround
from json import dumps as jsondumps, loads as jsonloads


class facecropperbase: # provides the processing associated to cropping functionality

    # forcing these constants to be float32 is the only way of model-matching!
    RAD2DEG = np.float32(180.0) / np.float32(np.pi)
    DEGBIAS = np.float32(360.0)

    input_dim = 128

    def preprocess(self, img):
        return preprocessimage(img, self.input_dim, self.input_dim)

    def getblur(self, img):
        assert self.ready
        assert img.size > 0 and img.ndim == 3
        h, w, _ = img.shape
        tmp = self.preprocess(img)
        blurout = self.predict(tmp)[0][0] # in current model, blur is output 0, not 1
        return blurout[0]

    @staticmethod
    def blur2q(blur):
        assert np.isscalar(blur)
        return 100 * (1 - blur)

    def getquality(self, img):
        return facecropperbase.blur2q(self.getblur(img))

    @staticmethod
    def gendebugimage(img, keypoints, R=255, G=200):
        assert isinstance(keypoints, tuple) and len(keypoints) == 3
        f32toint = lambda x : cppround(x).astype(np.int)
        H, W, _ = img.shape
        rad = int(cppround(0.01 * min(H, W)))
        x, y, c = keypoints
        color = (R, int(np.round(np.interp(c, [0.0, 0.5], [0, G]))), 0) if c < 0.5 \
            else (int(np.round(np.interp(c, [0.5, 1.0], [R, 0]))), G, 0)
        out = np.ndarray.copy(img)
        for xi, yi in zip(f32toint(x), f32toint(y)):
            assert xi <= W and yi <= H # actually it should be < strict
            out = cv2.circle(out, (xi, yi), rad, color=color, thickness=-1)
        return out

    def getkeypoints(self, img, warn=False, debugimage=None):
        assert self.ready
        assert img.size > 0 and img.ndim == 3
        h, w, _ = img.shape
        tmp = self.preprocess(img)
        pred = self.predict(tmp)[1][0] # in current model, keypoints are output 1, not 0
        # pred contains key points: [x0, y0, x1, y1, x2, y2, ... ] followed by confidence
        c = pred[-1]
        # if confidence is low, warning (see https://github.com/Touchless-ID/identy_android/blob/finger_v3_face_v3/face/src/main/java/com/identy/face/TensorFlowFaceKeypoints.java#L136-L139)
        if warn and c < 0.5: print("Warning: confidence is low ({:.2f})".format(c), flush=True)
        pred = pred[:-1]
        x = w * pred[0::2] # x coordinates of key points in range [0, w)
        y = h * pred[1::2] # y coordinates of key points in range [0, h)
        if debugimage is not None:
            saveimage(debugimage, self.gendebugimage(img, (x, y, c)))
        return x, y, c

    @staticmethod
    def expand(ibeg, iend, margin, imax, imin=0):
        margin1 = margin >> 1
        margin2 = margin - margin1
        ibeg = max(imin, ibeg - margin1)
        iend = min(imax, iend + margin2)
        return ibeg, iend

    @staticmethod
    def getrectangle(img, l, r, t, b, verbose=False):
        if type(l) == int and type(r) == int and \
           type(t) == int and type(b) == int:
            w = r - l
            h = b - t
        else:
            w = int(r - l)
            h = int(b - t)
            l = int(l)
            r = l + w
            t = int(t)
            b = t + h
        if verbose:
            print("Rectancle x={} y={} w={} h={}".format(l, t, w, h))
        assert w > 0 and h > 0, "Empty crop"
        # TODO: fill with zeros when crop exceeds image limits. Just checking for now.
        assert t >= 0 and b <= img.shape[0] and \
               l >= 0 and r <= img.shape[1], "Crop exceeds image limits"
        return img[t:b, l:r, :]

    @staticmethod
    def geteyesregion(img, keypoints):
        # needed for eyes quality metrics
        # NOTE THERE'S NO ANGLE CORRECTION HERE
        assert img.size > 0 and img.ndim == 3
        assert isinstance(keypoints, tuple) and len(keypoints) == 3
        x, y, _ = keypoints
        # TODO: remove -2 from right/bottom clip too avoid wasting
        #       two right/bottom points (is clipping needed at all?)
        l = int(max((3.0 * x[36] - x[39]) / 2.0, 0.0))
        r = int(min((3.0 * x[45] - x[42]) / 2.0, img.shape[1] - 2))
        t = int(max((3.0 * y[27] - y[30]) / 2.0, 0.0))
        b = int(min((      y[27] + y[30]) / 2.0, img.shape[0] - 2))
        # since l, r, t and b are int, we don't call getrectangle() in this case
        return facecropperbase.getrectangle(img, l, r, t, b)

    @staticmethod
    def getnoseregion(img, keypoints):
        # no longer needed for AS, but kept just in case
        # took point indices from: https://github.com/Touchless-ID/identy_android/blob/421c51fb95c6177b3ca417321ea50df0f1c6bb4c/face/src/main/java/com/identy/face/FaceBaseProcessor.java#L925
        # 36: left corner of left eye
        # 45: right corner of right eye
        # 11: contour point 12
        # NOTE THERE'S NO ANGLE CORRECTION HERE
        assert img.size > 0 and img.ndim == 3
        assert isinstance(keypoints, tuple) and len(keypoints) == 3
        x, y, _ = keypoints
        #return facecropperbase.getrectangle(img, l=x[36], r=x[45], t=y[36], b=y[11])
        return facecropperbase.getrectangle(img, l=int(x[36]), r=int(x[45]), t=int(y[36]), b=int(y[11]))

    @staticmethod
    def getL3region(img, keypoints):
        # needed for AS
        assert img.size > 0 and img.ndim == 3
        assert isinstance(keypoints, tuple) and len(keypoints) == 3
        x, y, _ = keypoints
        return facecropperbase.getrectangle(img, l=int(x[18]), r=int(x[25]), t=int(y[18]-10), b=int(y[8]+y[57])>>1)

    @staticmethod
    def align(img, keypoints):
        x, y, c = keypoints
        h, w, _ = img.shape
        # calculate the angle of the segment defined by points 19 and 24 (related to left and right eyes)
        thetarad = np.arctan2(y[24] - y[19], x[24] - x[19])
        thetadeg = thetarad * facecropperbase.RAD2DEG
        if thetadeg < 0.0: thetadeg += facecropperbase.DEGBIAS
        if thetadeg == 0.0: return img, keypoints
        # rotate so that the two points become aligned
        out = cimgrotate(img, thetadeg)
        costheta = np.cos(thetarad)
        sintheta = np.sin(thetarad)
        x -= 0.5 * w
        y -= 0.5 * h
        x_ = x * costheta + y * sintheta
        y_ = y * costheta - x * sintheta
        x = x_ + 0.5 * w
        y = y_ + 0.5 * h
        return out, (x, y, c)

    @staticmethod
    def getfaceregion(img, keypoints, align=True, mode="arcface", update_kps=False):
        assert img.size > 0 and img.ndim == 3
        assert isinstance(keypoints, tuple) and len(keypoints) == 3
        if align:
            img, keypoints = facecropperbase.align(img, keypoints)
        h, w, _ = img.shape
        x, y, c = keypoints
        # crop
        l = x[0]
        r = x[16]
        t = y[24]
        b = y[8]
        t -= 0.5 * (b - t)
        if mode == "icao":
            mid = 0.5 * (y[36] + y[45])
            faceh = b - t
            crophmin = faceh / 0.9
            crophmax = faceh / 0.6
            facew = r - l
            cropwmin = facew / 0.75
            cropwmax = facew / 0.5
            if cropwmin > 0.8 * crophmax:
                croph = crophmax
                cropw = cropwmin
            elif cropwmax < 0.74 * crophmin:
                croph = crophmin
                cropw = cropwmax
            else:
                ratiomin = max(0.74, cropwmin / crophmax)
                ratiomax = min(0.8, cropwmax / crophmin)
                ratio = max(min(0.75, ratiomax), ratiomin)
                if crophmin * ratio > cropwmin:
                    cropwmin = crophmin * ratio
                else:
                    crophmin = cropwmin / ratio
                if crophmax * ratio < cropwmax:
                    cropwmax = crophmax * ratio
                else:
                    crophmax = cropwmax / ratio
                croph = 0.5 * (crophmin + crophmax)
                cropw = 0.5 * (cropwmin + cropwmax)
            l = 0.5 * (r + l - cropw)
            r = cppround(l + cropw)
            l = cppround(l)
            t = 0.5 * (max(b - croph, mid - 0.5 * croph) + min(t, mid - 0.3 * croph))
            b = cppround(t + croph)
            t = cppround(t)
        else:
            l = cppround(l)
            r = cppround(r)
            b = cppround(b)
            t = int(np.floor(max(0.0, t)))
            if mode == "arcface":
                # force square shape to keep original aspect ratio
                diff = (b - t) - (r - l)
                if diff > 0:
                    l, r = facecropperbase.expand(l, r, diff, w)
                elif diff < 0:
                    t, b = facecropperbase.expand(t, b, -diff, h)
            elif mode is not None:
                raise Exception("Unknown cropping mode")
        crop = facecropperbase.getrectangle(img, l, r, t, b)
        return (crop, (x - l, y - t, c)) if update_kps else crop

    @staticmethod
    def getfaceregion4qc(img, keypoints):
        assert img.size > 0 and img.ndim == 3
        assert isinstance(keypoints, tuple) and len(keypoints) == 3
        x, y, _ = keypoints
        #return facecropperbase.rectangle(img, l=int(x[0]), r=int(x[16]), t=int(y[19]), b=int(y[8]))
        return facecropperbase.getrectangle(img, l=x[0], r=x[16], t=y[19], b=y[8])

    @staticmethod
    def getbackground(img, keypoints, margin=0.2):
        assert img.size > 0 and img.ndim == 3
        assert isinstance(keypoints, tuple) and len(keypoints) == 3
        h, w, _ = img.shape
        x, y, _ = keypoints
        l = x[0]
        r = x[16]
        d = margin * (r - l)
        l = cppround(max(l - d, 0))
        r = cppround(min(r + d, w))
        h >>= 1
        return [img[:h, :l, :], img[:h, r:, :]]

    @staticmethod
    def getfacepoly(img, keypoints):
        assert img.size > 0 and img.ndim == 3
        assert isinstance(keypoints, tuple) and len(keypoints) == 3
        x, y, _ = keypoints
        poly = np.array([[x[i], y[i]] for i in [0, 2, 4, 33, 12, 15, 16, 27]])
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, pts=[poly.astype(np.int32)], color=255)
        l = int(poly[0, 0])
        t = int(poly[7, 1])
        r = l + int(poly[6, 0] - l)
        b = t + int(max(poly[2, 1], poly[4, 1]) - t)
        out = img[t:b, l:r, :]
        mask = mask[t:b, l:r] > 0 # convert to bool
        return out, mask

    def process(self, img, debugimage=None):
        # this is a shortcut to the typical processing:
        # keypoints + face alignment & cropping (no AS)
        assert self.ready
        assert img.size > 0 and img.ndim == 3
        kp = self.getkeypoints(img, debugimage=debugimage)
        return facecropperbase.getfaceregion(img, kp)

    @staticmethod
    def savekeypoints(filename, keypoints):
        x, y, c = keypoints
        if filename.endswith(".json"):
            tmp = {"confidence": float(c),
                   "keypoints": [[float(xi), float(yi)] for xi, yi in zip(x, y)]}
            with open(filename, "w") as fid:
                fid.write(jsondumps(tmp, indent=1))
        else:
            tmp = np.insert(np.concatenate((x, y)).reshape((2, -1)).transpose().reshape(-1), 0, c)
            tmp.tofile(filename)
        return

    @staticmethod
    def loadkeypoints(filename):
        assert exists(filename)
        if filename.endswith(".json"):
            with open(filename, "r") as fid:
                tmp = jsonloads(fid.read())
            c = np.float32(tmp["confidence"])
            x, y = np.array(tmp["keypoints"], dtype=np.float32).transpose()
        else:
            tmp = np.fromfile(filename, dtype=np.float32)
            c = tmp[0]
            x = tmp[1::2]
            y = tmp[2::2]
        return x, y, c


class facecroppertflite(tflitebase, facecropperbase):

    def __init__(self, modelfile):
        tflitebase.__init__(self, modelfile)
        return


class facecropperonnx(onnxbase, facecropperbase):

    def __init__(self, modelfile):
        onnxbase.__init__(self, modelfile)
        return


def facecropper(modelfile):
    # returns an instance of the appropriate class depending on model file extension
    if modelfile.endswith(".tflite"):
        FC = facecroppertflite(modelfile)
    elif modelfile.endswith(".onnx") or modelfile.endswith(".ort"):
        FC = facecropperonnx(modelfile)
    else:
        raise Exception("Only tflite and onnx models are supported")
    return FC


def processfile(FC, filein, fileout, kpext=None, debug=False):
    img = loadimage(filein)
    if debug: showimage(img)
    if kpext is None:
        out = FC.process(img)
    else:
        # this is useful if you want to keep keypoints for AS, etc
        kp = FC.getkeypoints(img)
        FC.savekeypoints(filein + ".keypoints" + kpext, kp)
        out, kp = FC.getfaceregion(img, kp, update_kps=True)
        FC.savekeypoints(fileout + ".keypoints" + kpext, kp)
    if debug: showimage(out)
    return saveimage(fileout, out)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Args: model, input file/folder, output file/folder")
    else:
        fin = sys.argv[2]
        fout = sys.argv[3]
        assert exists(fin)
        assert fout != fin
        FC = facecropper(sys.argv[1])
        assert FC.ready
        kpext = None
        if isimage(fin) and isimage(fout):
            processfile(FC, fin, fout, kpext=kpext)
        else:
            processfolder(FC, fin, fout, processfile, ext=".png", kpext=kpext)

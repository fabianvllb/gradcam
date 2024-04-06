import sys
import numpy as np
from base import kerasbase
from misc import loadimage, isimage, preprocessimage, processfolder
from types import MethodType
from os.path import exists, basename, join, exists

class asfilterbase:
    def __init__(self, input_dims, mean=0.0, scale=1.0):
      assert len(input_dims) == 4
      self.input_height = input_dims[1]
      self.input_width = input_dims[2]
      self.input_chans = input_dims[3]
      self.mean = mean
      self.scale = scale
      return

    def preprocess(self, img, fname=None):
      assert img.dtype == np.uint8
      out = preprocessimage(img, self.input_height, self.input_width, center=self.mean, scale=self.scale)
      if fname is not None: self.save(fname, out) # useful for model i/o test vectors
      return out

    def runmodel(self, pre, fname="modeloutput"):
        out = self.predict(pre)
        #out = [o[0] for o in out] if isinstance(out, list) else out[0] # it is returning only the first element of the array
        if fname is not None: self.save(fname, out) # useful for model i/o test vectors
        return out

    def process(self, img, fnamein=None, fnameout=None):
      return self.runmodel(self.preprocess(img, fname=fnamein), fname=fnameout)

    @staticmethod
    def save(fname, x): # useful for model i/o test vectors
        if isinstance(fname, x):
            for i, e in enumerate(x):
                e.tofile(fname + ".{}".format(i + 1))
        else:
            x.tofile(fname)
        return
  
class asfilterkeras(kerasbase, asfilterbase):
    def __init__(self, modelfile, **kwargs):
        kerasbase.__init__(self, modelfile)
        asfilterbase.__init__(self, self.model.input_shape, **kwargs)
        return
    
def asfilter(modelfile, expected_dims=None, runmethod=None, **kwargs):
    """ if modelfile.endswith(".tflite"):
        asf = asfiltertflite(modelfile, **kwargs)
    elif modelfile.endswith(".onnx") or modelfile.endswith(".ort"):
        asf = asfilteronnx(modelfile, **kwargs) """
    if modelfile.endswith(".h5") or modelfile.endswith(".keras"):
        asf = asfilterkeras(modelfile, **kwargs)
    else:
        raise Exception("Only Keras models are supported")
    assert asf.ready
    if expected_dims is not None:
        assert len(expected_dims) == 3
        assert asf.input_height == expected_dims[0] and \
               asf.input_width == expected_dims[1] and \
               asf.input_chans == expected_dims[2]
    if runmethod is not None:
        asf.run = MethodType(runmethod, asf)
    return asf

def getoutput(LX, img, **kwargs):
    return LX.process(img, **kwargs)

L1filter = lambda fname : asfilter(fname, expected_dims=(640,480,3), runmethod=getoutput, scale=255.0)

class asengine:
    FUSION_PARAMS = {
        "paper" : {
            "L1"   :  0.0195571,
            "L3e"  :  0.08723006,
            "L3n"  :  0.10049366,
            "L4"   :  0.17539803,
            "L7"   :  0.01996573,
            "L8"   :  0.00487081,
            "bias" : -2.18431368
        },
        "screen" : {
            "L1"   :  0.02201227,
            "L3e"  :  0.05705241,
            "L3n"  :  0.04011848,
            "L4"   :  0.0,
            "L7"   :  0.01546256,
            "L8"   :  0.00674347,
            "bias" : -2.21068087
        },
        "xgrid" : [-5.0, -1.0609872608443562, 0.14147519744708248, 1.6366761448827072, 2.4874924748784757,
                          3.143528076966934 , 3.616338721302119  , 4.056812464435916 , 4.389889249024052 , 10.0],
        "xgrid_desktop" : [-10.0, -2.5696953215528144, -1.0609872608443562, 0.14147519744708248, 1.6366761448827072,
                                   2.4874924748784757,  3.143528076966934 , 3.616338721302119  , 4.056812464435916 , 10.0],
        "ygrid" : [0.0, 0.5, 0.65, 0.75, 0.82, 0.86, 0.9, 0.92, 0.95, 1.0]
    }

    THRESHOLDS = { lev: th for lev, th in zip(
        [None, "LOW", "MED", "MOD", "BH", "HIGH", "BVH", "VH", "HIGHEST"],
        FUSION_PARAMS["ygrid"]) if lev is not None }

    def __init__(self, rootdir="", modelL1=join("L1", "face_as_l1.h5")):
        self.ready = False
        self.L1  = L1filter(join(rootdir, modelL1))
        self.ready = True
        return
    
    def processL1(self, img, desktop=False):
        print("========== Starting processL1 ==========")
        assert self.ready
        assert img.size > 0 and img.ndim == 3
        P1 = self.L1.run(img)
        debug = {
            "paper"  : { "L1": P1[0], "wsum": 0.0 },
            "screen" : { "L1": P1[1], "wsum": 0.0 }
        }
        score = np.inf
        for scenario in ["paper", "screen"]:
            debug_ = debug[scenario]
            fparams = self.FUSION_PARAMS[scenario]
            wsum = 0.0
            for key in ["L1"]:
                wsum += fparams[key] * debug_[key]
            debug[scenario]["wsum"] = wsum
        score = np.interp(score,
            self.FUSION_PARAMS["xgrid_desktop" if desktop else "xgrid"],
            self.FUSION_PARAMS["ygrid"])
        return score, debug
    
    @staticmethod
    def isspoof(score, level):
        th = asengine.THRESHOLDS.get(level, None)
        assert th is not None, "Level not recognized, please check asengine.THRESHOLDS"
        return score < th
    
    @staticmethod
    def printCSVheader(file=sys.stdout):
        header = "filename"
        for s in ["L1", "wsum"]:
            header += ",{}_paper,{}_screen".format(s, s)
        header += ",score"
        for l in asengine.THRESHOLDS:
            header += ",isspoof{}".format(l)
        print(header, file=file, flush=True)
        return
    
    @staticmethod
    def printCSVline(filename, score, debug, file=sys.stdout):
        line = filename
        for s in ["L1", "wsum"]:
            line += ",{},{}".format(debug["paper"][s], debug["screen"][s])
        line += ",{}".format(score)
        for l in asengine.THRESHOLDS:
            line += ",{}".format(int(asengine.isspoof(score, l)))
        print(line, file=file, flush=True)
        return
import sys
from os.path import exists, basename, join, exists, dirname, realpath
import numpy as np
import cv2
import tensorflow as tf
from base import onnxbase

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

rsz = onnxbase(join(dirname(realpath(__file__)), "resize.ort"))

def checkimage(img):
    return img is not None and img.size > 0 and img.ndim == 3 and img.dtype == np.uint8

def loadimage(filename):
    img = cv2.imread(filename)
    assert checkimage(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocessimage(img, height, width, center=0.0, scale=1.0):
    tmp = img.astype(np.float32)
    tmp = np.expand_dims(tmp, axis=0)
    if img.shape[1] != height or img.shape[2] != width:
        tmp = rsz.predict([tmp, np.array([[height, width]], dtype=np.float32)])
        tmp = np.floor(tmp + 0.5)
    if center != 0.0: tmp -= center
    if scale != 1.0: tmp /= scale
    return tmp

def preprocess(img, input_height, input_width, scale):
      assert img.dtype == np.uint8
      out = preprocessimage(img, input_height, input_width, scale=scale)
      #if fname is not None: self.save(fname, out) # useful for model i/o test vectors
      return out

""" def processfile(model, filein, fileout):
    # fileout is ignored, it's just kept for compatibility with processfolder
    ASE = model
    img = loadimage(filein)
    score, debug = ASE.processL1(img)
    ASE.printCSVline(basename(filein), score, debug)
    return True """

def calculateScore(P1, desktop=False):
    debug = {
        "paper"  : { "L1": P1[0], "wsum": 0.0 },
        "screen" : { "L1": P1[1], "wsum": 0.0 }
    }
    score = np.inf
    for scenario in ["paper", "screen"]:
        debug_ = debug[scenario]
        fparams = FUSION_PARAMS[scenario]
        wsum = 0.0
        for key in ["L1"]:
            wsum += fparams[key] * debug_[key]
        debug[scenario]["wsum"] = wsum
    score = np.interp(score,
        FUSION_PARAMS["xgrid_desktop" if desktop else "xgrid"],
        FUSION_PARAMS["ygrid"])
    return score, debug

def isspoof(score, level):
        th = THRESHOLDS.get(level, None)
        assert th is not None, "Level not recognized, please check asengine.THRESHOLDS"
        return score < th
    
def printCSVheader(file=sys.stdout):
    header = "filename"
    for s in ["L1", "wsum"]:
        header += ",{}_paper,{}_screen".format(s, s)
    header += ",score"
    for l in THRESHOLDS:
        header += ",isspoof{}".format(l)
    print(header, file=file, flush=True)
    return

def printCSVline(filename, score, debug, file=sys.stdout):
    line = filename
    for s in ["L1", "wsum"]:
        line += ",{},{}".format(debug["paper"][s], debug["screen"][s])
    line += ",{}".format(score)
    for l in THRESHOLDS:
        line += ",{}".format(int(isspoof(score, l)))
    print(line, file=file, flush=True)
    return

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def main():
    if len(sys.argv) != 3:
        print("Args: AS model folder, input file/folder")
        return
    
    # load model
    rootdir = sys.argv[1]
    filein = sys.argv[2]
    assert exists(filein)
    modelfile = join(rootdir, join("L1", "face_as_l1.h5"))
    assert exists(modelfile)
    if modelfile.endswith(".h5") or modelfile.endswith(".keras"):
        model = tf.keras.models.load_model(modelfile, compile=False)
    else:
        raise Exception("Only Keras models are supported")
    
    expected_dims = (640,480,3)
    scale=255.0

    input_height = model.input_shape[1]
    input_width = model.input_shape[2]
    input_chans = model.input_shape[3]
    assert  input_height == expected_dims[0] and \
            input_width == expected_dims[1] and \
            input_chans == expected_dims[2]
    input_names = [input_layer.name for input_layer in model.inputs]
    
    printCSVheader()

    # pre-process
    img = loadimage(filein)
    pre_img = preprocess(img, input_height, input_width, scale)
    
    # predict
    if not isinstance(pre_img, list):
        pre_img = [pre_img]
    input_dict = {input_name: input_data for input_name, input_data in zip(input_names, pre_img)}
    P1 = model.predict(input_dict)
    if len(P1) == 1: P1 = P1[0]

    print(P1)

    p1paper = tf.math.sigmoid(P1[0])
    p1screen = tf.math.sigmoid(P1[1])

    print("final P1(paper) sigmoid is :", p1paper.numpy())
    print("final P1(screen) sigmoid is :", p1screen.numpy())
    #last_conv_layer_name = "conv2d_13"

    # classify
    #score, debug = calculateScore(P1)

    """ p1paper = sigmoid(debug["paper"]["wsum"])
    p1screen = sigmoid(debug["screen"]["wsum"])

    print("final P1(paper) sigmoid is :", p1paper)
    print("final P1(screen) sigmoid is :", p1screen)

    print("final debug is :", debug)
    print("final score is :", score)

    # report
    printCSVline(basename(filein), score, debug) """

if __name__ == "__main__":
    main()
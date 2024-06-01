import sys
from os.path import exists, basename, join, exists, dirname, realpath, splitext
import numpy as np
import cv2
import tensorflow as tf
from base import onnxbase
from misc import loadimage, isimage
from os import listdir
import os
import shutil
from sklearn.model_selection import train_test_split
import random

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

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

def processfile(model, filein, fileout):
    expected_dims = (640,480,3)
    scale=255.0

    input_height = model.input_shape[1]
    input_width = model.input_shape[2]
    input_chans = model.input_shape[3]
    assert  input_height == expected_dims[0] and \
            input_width == expected_dims[1] and \
            input_chans == expected_dims[2]
    input_names = [input_layer.name for input_layer in model.inputs]

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

# Function to split and copy images
""" def split_and_copy_images(source_dir, train_dir, validation_dir, split_ratio=0.2):
    images = os.listdir(source_dir)
    train_images, validation_images = train_test_split(images, test_size=split_ratio, random_state=42)
    
    for image in train_images:
        src_path = os.path.join(source_dir, image)
        dst_path = os.path.join(train_dir, image)
        shutil.copyfile(src_path, dst_path)
    
    for image in validation_images:
        src_path = os.path.join(source_dir, image)
        dst_path = os.path.join(validation_dir, image)
        shutil.copyfile(src_path, dst_path)

def preparedata():
    # Define paths
    base_dir = 'data/images'
    real_dir = os.path.join(base_dir, 'Real')
    spoof_dir = os.path.join(base_dir, 'Spoof')

    train_dir = 'data/train'
    validation_dir = 'data/validation'

    # Create train and validation directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    train_real_dir = os.path.join(train_dir, 'Real')
    train_spoof_dir = os.path.join(train_dir, 'Spoof')
    validation_real_dir = os.path.join(validation_dir, 'Real')
    validation_spoof_dir = os.path.join(validation_dir, 'Spoof')

    os.makedirs(train_real_dir, exist_ok=True)
    os.makedirs(train_spoof_dir, exist_ok=True)
    os.makedirs(validation_real_dir, exist_ok=True)
    os.makedirs(validation_spoof_dir, exist_ok=True)

    # Split and copy images for both Real and Spoof categories
    split_and_copy_images(real_dir, train_real_dir, validation_real_dir)
    split_and_copy_images(spoof_dir, train_spoof_dir, validation_spoof_dir)

    print("Dataset split and copied successfully!") """

def createDatasets(data_dir, img_height, img_width, img_channels):
    #batch_sizes = [16, 32, 64]
    batch_size = 32

    #for batch_size in batch_sizes:
    print(f"Testing with batch size: {batch_size}")
    
    # Load the dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',  # Labels will be encoded as integers
        color_mode="rgb",
        validation_split=0.2,  # Reserve 20% of the data for validation
        subset="training",
        seed=123,  # For reproducibility
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode='int',
        color_mode="rgb",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(class_names)

    # Prefetch the datasets for optimal performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
    # Print a batch of data to see how it looks
    for images, labels in train_ds.take(1):
        print(f"Batch size: {batch_size} - Batch shape: {images.shape}")
    
    return train_ds, val_ds

def createDataDictionary(data_dir):
    image_dict = {}

    # Walk through the directory
    for class_name in ['real', 'spoof']:
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_dir, filename)
                image_dict[filename] = class_name

    return image_dict

def createDataSplit(data_dict, validation_split=0.2):
    img_list = list(data_dict.keys())
    random.shuffle(img_list)

    split_ratio = 1-validation_split
    split_index = int(len(img_list) * split_ratio)

    training_keys = img_list[:split_index]
    validation_keys = img_list[split_index:]

    training_dict = {key: data_dict[key] for key in training_keys}
    validation_dict = {key: data_dict[key] for key in validation_keys}

    """ print("Training set:", len(training_dict), "->", len(training_dict)/len(img_list))
    print("Validation set:", len(validation_dict), "->", len(validation_dict)/len(img_list))
    print("Total data:", len(img_list)) """

    return training_dict, validation_dict

def main():
    if len(sys.argv) != 2:
        print("Args: AS model folder, input file/folder")
        return
    
    # load model
    rootdir = sys.argv[1]
    #filein = sys.argv[2]
    #assert exists(filein)
    modelfile = join(rootdir, join("L1", "face_as_l1.h5"))
    assert exists(modelfile)
    if modelfile.endswith(".h5") or modelfile.endswith(".keras"):
        model = tf.keras.models.load_model(modelfile, compile=False)
    else:
        raise Exception("Only Keras mode ls are supported")

    data_dir = 'data/images'
    expected_dims = (640,480,3)

    # pre-process
    data_dict = createDataDictionary(data_dir)
    train_ds, val_ds = createDataSplit(data_dict)

    #processfolder(model, filein, None, processfile, verbose=False)

    
    #last_conv_layer_name = "conv2d_13"

    # classify
    #score, debug = calculateScore(P1)

if __name__ == "__main__":
    main()
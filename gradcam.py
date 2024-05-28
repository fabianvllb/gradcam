import sys
from os.path import exists, basename, join, exists, dirname, realpath
import numpy as np
import cv2
import tensorflow as tf
from base import onnxbase
import matplotlib.pyplot as plt
import keras

rsz = onnxbase(join(dirname(realpath(__file__)), "resize.ort"))

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

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
    print(img.shape[1], img.shape[2])
    if img.shape[1] != height or img.shape[2] != width:
        tmp = rsz.predict([tmp, np.array([[height, width]], dtype=np.float32)])
        tmp = np.floor(tmp + 0.5)
    if center != 0.0: tmp -= center
    if scale != 1.0: tmp /= scale
    return tmp

def preprocess(img, input_height, input_width, scale=1.0):
      assert img.dtype == np.uint8
      out = preprocessimage(img, input_height, input_width, scale=scale)
      return out

def load_augmented_model(
        model_file_path,
        intermediate_layer_name=None,
        output_layer_name=None,
        negate=False):
    assert model_file_path.endswith(".h5"), "Model extension must be .h5"
    try:
        model = tf.keras.models.load_model(model_file_path)
    except:
        relu6 = lambda x: tf.keras.backend.relu(x, max_value=6)
        model = tf.keras.models.load_model(model_file_path, custom_objects={"relu6":relu6})
    output = model.outputs[0] if output_layer_name is None \
        else model.get_layer(output_layer_name).output
    assert len(output.shape) == 2, "Output shape must be (None, int)"
    #output = output[:, output_index:output_index+1]
    if negate: output = -output
    if intermediate_layer_name is None:
        for layer in model.layers:
            if not isinstance(layer, tf.keras.layers.InputLayer):
                print(layer.name)
        print("Please choose intermediate layer from the list above")
        sys.exit(0)
    interm = model.get_layer(intermediate_layer_name).output
    outputs = [output, interm]
    aug_model = tf.keras.Model(model.inputs, outputs)
    aug_model.trainable = False
    return aug_model

def compute_gradcam(model, image, class_index=None):
    with tf.GradientTape() as tape:
        logits, conv_outputs = model(image)
        print("P1: ", logits[0])
        if class_index is None:
            class_index = tf.argmin(logits[0])
        class_score = logits[:, class_index]
        print("class_score: ", class_score)
        
    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    #heatmap = tf.squeeze(conv_outputs[0] @ pooled_grads[..., tf.newaxis])

    # Apply ReLU to the Grad-CAM
    heatmap = tf.maximum(cam, 0)

    # Normalize the heatmap to [0, 1]
    heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.4):
    # Resize the heatmap to match the original image dimensions
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    return overlayed

""" def save_and_display_gradcam(img, heatmap, size=3, cam_outfile_path=None, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    fig, axes = plt.subplots(1, 2, figsize=(size * 2, size))

    axes.imshow(img)

    if cam_outfile_path is not None:
        plt.savefig(cam_outfile_path)
    plt.show() """

""" def showimages(imglist, size=3, outfile=None):
    n = len(imglist)
    fig, axes = plt.subplots(1, n, figsize=(size * n, size))
    for i, img in enumerate(imglist):
        axes[i].imshow(img)
    if outfile is not None:
        plt.savefig(outfile)
    plt.show()
    return """

def main():
    print('=================== Gradcam ==========================')
    if len(sys.argv) != 3:
        print("Args: AS model folder, input file/folder")
        return
    
    modelfile = sys.argv[1]
    filein = sys.argv[2]
    assert exists(filein)
    assert exists(modelfile)
    
    expected_dims = (640,480,3)
    scale=255.0

    augmented_model = load_augmented_model(
        model_file_path=modelfile,
        intermediate_layer_name='conv2d_13',
        output_layer_name="output_layer"
    )

    input_height = augmented_model.input_shape[1]
    input_width = augmented_model.input_shape[2]
    input_chans = augmented_model.input_shape[3]
    assert  input_height == expected_dims[0] and \
            input_width == expected_dims[1] and \
            input_chans == expected_dims[2]

    image = loadimage(filein)
    print("image", image)
    prep_image = preprocess(image, input_height, input_width, scale)

    rsz_img = np.uint8(255 * prep_image)
    rsz_img = np.squeeze(rsz_img)

    heatmap = compute_gradcam(augmented_model, prep_image)

    print("heatmap: ", heatmap)

    # Display the heatmap
    """ plt.matshow(heatmap)
    plt.show() """
    """ plt.imshow(heatmap, cmap='jet')
    plt.colorbar()
    plt.title('Grad-CAM Heatmap')
    plt.show() """

    #showimages([prep_image[0] * scale, heatmap])
    overlayed_image = overlay_heatmap(heatmap, image)
    #save_and_display_gradcam(img=prep_image, heatmap=heatmap, cam_outfile_path="grad.jpg")

    """ plt.matshow(heatmap)
    plt.show() """

    # Display the overlaid image
    plt.imshow(overlayed_image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
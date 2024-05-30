import sys
from os.path import exists, basename, join, exists, dirname, realpath
import numpy as np
import cv2
import tensorflow as tf
from base import onnxbase
import matplotlib.pyplot as plt

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

def load_augmented_model(model, intermediate_layer_name=None, output_layer_name=None):
    output = model.outputs[0] if output_layer_name is None \
        else model.get_layer(output_layer_name).output
    assert len(output.shape) == 2, "Output shape must be (None, int)"
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
        logits = logits[0]
        negated_logits = -logits
        if class_index is None:
            print("logits: ", logits)
            if logits.numpy()[0] > 0 and logits.numpy()[1] > 0:
                print("Image is real")
                class_index = tf.argmin(logits)
                class_score = logits[class_index]
            else:
                class_index = tf.argmax(negated_logits)
                if class_index == 0:
                    print("Image is paper spoof")
                else:
                    print("Image is screen spoof")
                class_score = negated_logits[class_index]
        print("class_score: ", class_score)
        
    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    #heatmap = tf.squeeze(conv_outputs[0] @ pooled_grads[..., tf.newaxis])

    heatmap = tf.maximum(cam, 0)

    # Normalize the heatmap to [0, 1]
    heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap

def overlay_heatmap_on_image(heatmap, image, alpha=0.6):
    # Resize the heatmap to match the original image dimensions
    heatmap = cv2.resize(heatmap.numpy(), (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    return overlayed

def print_leaky_relu_alpha_values(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LeakyReLU):
            print(f"Layer: {layer.name}, Alpha: {layer.alpha}")

def get_leaky_relu_first_alpha_value(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LeakyReLU):
            alpha = layer.alpha
            break
    return alpha

@tf.custom_gradient
def guided_leaky_relu(x, alpha=0.3):
    def grad(dy):
        return tf.where(dy > 0, dy, 0) * tf.where(x > 0, 1.0, alpha)
    return tf.nn.leaky_relu(x, alpha), grad #tf.where(x > 0, x, alpha * x)

def modify_model_for_guided_backprop_leaky(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LeakyReLU):
            alpha = layer.alpha
            layer.activation = lambda x, alpha=alpha: guided_leaky_relu(x, alpha)
    return model

def compute_guided_backprop(model, image, class_index=None):
    with tf.GradientTape() as tape:
        tape.watch(image)
        logits = model(image)[0]
        negated_logits = -logits
        if class_index is None:
            if logits.numpy()[0] > 0 and logits.numpy()[1] > 0:
                class_index = tf.argmin(logits)
                class_score = logits[class_index]
            else:
                class_index = tf.argmax(negated_logits)
                class_score = negated_logits[class_index]
        """ if class_index is None:
            class_index = tf.argmax(logits[0])
        class_score = logits[:, class_index] """

    grads = tape.gradient(class_score, image)
    guided_backprop = tf.maximum(grads, 0)
    
    return guided_backprop[0]

def get_guided_gradcam(guided_backprop, gradcam_heatmap):
    guided_backprop = (guided_backprop - tf.reduce_min(guided_backprop)) / (tf.reduce_max(guided_backprop) - tf.reduce_min(guided_backprop))
    guided_backprop = tf.cast(guided_backprop * 255, tf.uint8).numpy()
    
    gradcam_heatmap_resized = cv2.resize(gradcam_heatmap.numpy(), (guided_backprop.shape[1], guided_backprop.shape[0]))
    gradcam_heatmap_resized = np.repeat(gradcam_heatmap_resized[..., np.newaxis], 3, axis=-1)
    
    combined_image = guided_backprop * gradcam_heatmap_resized
    combined_image = (combined_image - np.min(combined_image)) / (np.max(combined_image) - np.min(combined_image))
    combined_image = (combined_image * 255).astype(np.uint8)
    
    return combined_image

def main():
    print('=================== Gradcam ==========================')
    if len(sys.argv) != 3:
        print("Args: AS model folder, input file")
        return
    
    modelfile = sys.argv[1]
    filein = sys.argv[2]
    assert exists(filein)
    assert exists(modelfile)
    assert modelfile.endswith(".h5"), "Model extension must be .h5"
    
    expected_dims = (640,480,3)
    scale=255.0

    model = tf.keras.models.load_model(modelfile)
    model = modify_model_for_guided_backprop_leaky(model)
    
    augmented_model = load_augmented_model(
        model=model,
        intermediate_layer_name='conv2d_13',
        output_layer_name="output_layer"
    )

    model_input_height = augmented_model.input_shape[1]
    model_input_width = augmented_model.input_shape[2]
    model_input_chans = augmented_model.input_shape[3]
    assert  model_input_height == expected_dims[0] and \
            model_input_width == expected_dims[1] and \
            model_input_chans == expected_dims[2]

    original_image  = loadimage(filein)
    prep_image = preprocess(original_image , model_input_height, model_input_width, scale)
    if original_image.shape[1] != model_input_height or original_image.shape[2] != model_input_width:
        resized_img = np.uint8(255 * prep_image)
        original_image = np.squeeze(resized_img)
    
    gradcam_heatmap = compute_gradcam(augmented_model, prep_image)
    
    input_image = tf.convert_to_tensor([original_image], dtype=tf.float32)
    guided_backprop = compute_guided_backprop(model, input_image)
    gradcam_overlay = overlay_heatmap_on_image(gradcam_heatmap, original_image)
    guided_gradcam = get_guided_gradcam(guided_backprop, gradcam_heatmap)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gradcam_overlay)
    plt.title('Grad-CAM Overlay')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(guided_gradcam)
    plt.title('Guided Grad-CAM')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
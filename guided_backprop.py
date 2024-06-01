import sys
from os.path import exists, basename, join, exists, dirname, realpath
import keras.activations
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

def check_image(img):
    return img is not None and tf.size(img) > 0 and len(img.shape) == 3 and img.dtype == tf.uint8

def load_image(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    assert check_image(img)
    return img

def preprocess_image(img, height, width, center=0.0, scale=1.0):
    tmp = tf.cast(img, tf.float32)
    tmp = tf.expand_dims(tmp, axis=0)
    print(img.shape[1], img.shape[2])
    if img.shape[1] != height or img.shape[2] != width:
        tmp = tf.image.resize(tmp, [height, width])
        tmp = tf.math.floor(tmp + 0.5)
    if center != 0.0: tmp -= center
    if scale != 1.0: tmp /= scale
    return tmp

def preprocess(img, input_height, input_width, scale=1.0):
    assert img.dtype == tf.uint8
    out = preprocess_image(img, input_height, input_width, scale=scale)
    return out

def norm_flat_image(img):
    grads_norm = img[:,:,0] + img[:,:,1] + img[:,:,2]
    grads_norm = (grads_norm - tf.reduce_min(grads_norm))/ (tf.reduce_max(grads_norm)- tf.reduce_min(grads_norm))
    return grads_norm

def plot_maps(img1, img2,vmin=0.3,vmax=0.7, mix_val=2):
    f = plt.figure(figsize=(15,45))
    plt.subplot(1,3,1)
    plt.imshow(img1,vmin=vmin, vmax=vmax, cmap="gray")
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(img2, cmap = "gray")
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(img1*mix_val+img2/mix_val, cmap = "gray" )
    plt.axis("off")

    plt.show()

@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,tf.float32)  * tf.cast(x>0,tf.float32) * dy
    return tf.nn.relu(x), grad

@tf.custom_gradient
def guided_leaky_relu(x, alpha=0.3):
    def grad(dy):
        return dy * tf.where(x > 0, 1.0, alpha)
    return tf.nn.leaky_relu(x, alpha), grad #tf.where(x > 0, x, alpha * x)

""" @tf.custom_gradient
def guided_leaky_relu(x, alpha=0.3):
    def grad(dy):
        return tf.cast(dy > 0, tf.float32) * tf.cast(x > 0, tf.float32) * dy, tf.zeros_like(alpha)
    return tf.nn.leaky_relu(x, alpha), grad """

def modify_model_for_guided_backprop_leaky(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LeakyReLU):
            alpha = layer.alpha
            layer.activation = lambda x, alpha=alpha: guided_leaky_relu(x, alpha)
            print(layer.activation)
    return model

def main():
    print('=================== Saliency Map ==========================')
    
    img_path = sys.argv[1]
    assert exists(img_path), "Image file does not exist"
    
    # ==================== Saliency Map Identy ==========================
    expected_dims = (640,480,3)
    model_path = join("models",join("L1", "face_as_l1.h5"))
    assert exists(model_path), "Model file does not exist"
    model = tf.keras.models.load_model(model_path, compile=False)

    original_image = load_image(img_path)
    input_img = preprocess(original_image , expected_dims[0], expected_dims[1], 255.0)
    #img = tf.cast((input_img * 255), tf.uint8)
    #print(img)

    model = modify_model_for_guided_backprop_leaky(model)
    #if layer.activation == keras.activations.leaky_relu:
    """ activation_num = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LeakyReLU):
            print(layer)
            #layer.activation = guidedRelu """

    result = model(input_img)[0]
    class_index = tf.argmin(result)

    with tf.GradientTape() as tape:
        tape.watch(input_img)
        logits = model(input_img)[0]
        print("logits: ", logits)
        class_score = logits[class_index]
    grads = tape.gradient(class_score, input_img)
    plot_maps(norm_flat_image(grads[0]), norm_flat_image(input_img[0]))

if __name__ == "__main__":
    main()
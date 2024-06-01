import sys
from os.path import exists, basename, join, exists, dirname, realpath
import keras.activations
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

def prep_input(path, expected_dims):
    image = tf.image.decode_png(tf.io.read_file(path))
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, expected_dims[:2])
    return image

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

def modify_model_for_guided_backprop_leaky(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LeakyReLU):
            layer.activation = lambda x: guidedRelu(x)
    return model

def main():
    print('=================== Saliency Map ==========================')
    
    img_path = sys.argv[1]
    assert exists(img_path), "Image file does not exist"

    # ==================== Saliency Map DenseNet201 ==========================
    expected_dims = (224,224,3)
    model = tf.keras.applications.densenet.DenseNet201()

    input_img = prep_input(img_path, expected_dims)
    input_img = tf.keras.applications.densenet.preprocess_input(input_img)

    activation_num = 0
    layer_dict = [layer for layer in model.layers[1:] if hasattr(layer,'activation')]
    for layer in layer_dict:
        if layer.activation == tf.keras.activations.relu:
            print(layer)
            layer.activation = guidedRelu
            print("changed activation", activation_num)
            activation_num += 1

    logits = model(input_img)
    max_idx = tf.argmax(logits,axis = 1)
    tf.keras.applications.imagenet_utils.decode_predictions(logits.numpy())

    with tf.GradientTape() as tape:
        tape.watch(input_img)
        logits = model(input_img)
        class_score = logits[0,max_idx[0]]
    grads = tape.gradient(class_score, input_img)
    plot_maps(norm_flat_image(grads[0]), norm_flat_image(input_img[0]))

if __name__ == "__main__":
    main()
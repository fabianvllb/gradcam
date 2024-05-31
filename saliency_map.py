import sys
from os.path import exists, basename, join, exists, dirname, realpath
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import napari

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

def main():
    print('=================== Saliency Map ==========================')
    
    img_path = sys.argv[1]
    assert exists(img_path), "Image file does not exist"

    expected_dims = (640,480,3)
    model_path = join("models",join("L1", "face_as_l1.h5"))
    assert exists(model_path), "Model file does not exist"
    test_model = tf.keras.models.load_model(model_path, compile=False)
    """ expected_dims = (224,224,3)
    test_model = tf.keras.applications.densenet.DenseNet201() """

    #test_model.summary()
    original_image = load_image(img_path)
    input_img = preprocess(original_image , expected_dims[0], expected_dims[1], 255.0)
    """ img = tf.cast((input_img * 255), tf.uint8)
    print(img) """
    """ input_img = prep_input(img_path, expected_dims)
    input_img = tf.keras.applications.densenet.preprocess_input(input_img) """

    """ result = test_model(input_img)
    max_idx = tf.argmax(result,axis = 1)
    tf.keras.applications.imagenet_utils.decode_predictions(result.numpy())

    with tf.GradientTape() as tape:
        tape.watch(input_img)
        result = test_model(input_img)
        max_score = result[0,max_idx[0]]
    grads = tape.gradient(max_score, input_img)
    plot_maps(norm_flat_image(grads[0]), norm_flat_image(input_img[0])) """

    with tf.GradientTape() as tape:
        tape.watch(input_img)
        logits = test_model(input_img)[0]
        print("logits: ", logits)
        class_index = tf.argmin(logits)
        class_score = logits[class_index]
    grads = tape.gradient(class_score, input_img)
    plot_maps(norm_flat_image(grads[0]), norm_flat_image(input_img[0]))

if __name__ == "__main__":
    main()
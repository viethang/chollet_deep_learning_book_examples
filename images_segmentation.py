# from tensorflow import keras
import keras
from keras.utils import array_to_img
from keras import layers
import random
import numpy as np
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import os

input_dir = "data/segmentation/images"
target_dir = "data/segmentation/annotations/trimaps/"
input_img_paths = sorted(
    [os.path.join(input_dir, fname)
     for fname in os.listdir(input_dir)
     if fname.endswith(".jpg")])

target_paths = sorted(
    [os.path.join(target_dir, fname)
     for fname in os.listdir(target_dir)
     if fname.endswith(".png") and not fname.startswith(".")])

plt.axis("off")
# plt.imshow(load_img(input_img_paths[9]))
# plt.show()


def display_target(target_array):
    normalized_array = (target_array.astype("uint8") - 1) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0], cmap='gray')


# img = img_to_array(load_img(target_paths[9], color_mode="grayscale"))
# display_target(img)
# plt.show()


img_size = (200, 200)
num_imgs = len(input_img_paths)
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)


def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))


def path_to_target(path):
    img = img_to_array(
        load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8") - 1
    return img


input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")
for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_img_paths[i])
    targets[i] = path_to_target(target_paths[i])

num_val_samples = 1000
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_imgs = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:]


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(
        256, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(
        128, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(
        64, 3, activation="relu", padding="same", strides=2)(x)
    outputs = layers.Conv2D(num_classes, 3, activation="softmax",
                            padding="same")(x)
    model = keras.Model(inputs, outputs)
    return model


model = get_model(img_size=img_size, num_classes=3)
model.summary()


def train():
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    callbacks = [
        keras.callbacks.ModelCheckpoint("models/oxford_segmentation.keras",
                                        save_best_only=True)
    ]
    history = model.fit(train_input_imgs, train_targets,
                        epochs=50,
                        callbacks=callbacks,
                        batch_size=64,
                        validation_data=(val_input_imgs, val_targets))


def plot_history(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()


def test():
    model = keras.models.load_model("models/oxford_segmentation.keras")
    i = 5
    test_image = val_input_imgs[i]
    plt.axis("off")
    plt.imshow(array_to_img(test_image))
    plt.show()
    mask = model.predict(np.expand_dims(test_image, 0))[0]
    display_mask(mask)


def test_other_image(path):
    model = keras.models.load_model("models/oxford_segmentation.keras")
    test_image = img_to_array(load_img(path, target_size=img_size))
    plt.axis("off")
    plt.imshow(array_to_img(test_image))
    plt.show()
    mask = model.predict(np.expand_dims(test_image, 0))[0]
    display_mask(mask)


def display_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis("off")
    plt.imshow(mask)
    plt.show()

# train()

# test()

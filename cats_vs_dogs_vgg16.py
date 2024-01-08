from tensorflow import keras
import numpy as np
import pathlib
from matplotlib import pyplot as plt

conv_base = keras.applications.vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(180, 180, 3))
conv_base.trainable = False

conv_base.summary()
data_dir = pathlib.Path(
    "data/cats_vs_dogs_small")


def get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
    print("labels", len(all_labels), all_labels[0].shape)
    return np.concatenate(all_features), np.concatenate(all_labels)


def train():
    train_dataset = keras.utils.image_dataset_from_directory(
        data_dir / "train",
        image_size=(180, 180),
        batch_size=32)
    validation_dataset = keras.utils.image_dataset_from_directory(
        data_dir / "validation",
        image_size=(180, 180),
        batch_size=32)

    train_features, train_labels = get_features_and_labels(train_dataset)
    print("train_features", train_features.shape)
    print("train_labels", train_labels.shape)
    val_features, val_labels = get_features_and_labels(validation_dataset)
    print("val_features", val_features.shape)
    

    model = get_model()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath="models/feature_extraction.keras",
                                       save_best_only=True,
                                       monitor='val_loss'),
        keras.callbacks.TensorBoard(log_dir="logs/feature_extraction_logs")
    ]
    history = model.fit(train_features, train_labels,
                        epochs=20,
                        validation_data=(val_features, val_labels),
                        callbacks=callbacks)

    plot_history(history)

def get_model():
    inputs = keras.Input(shape=(5, 5, 512))
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.Dropout(.5)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model

def plot_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.show()

def test_model():
    model = keras.models.load_model("models/feature_extraction.keras")
    model.summary()
    test_dataset = keras.utils.image_dataset_from_directory(
        data_dir / "test",
        image_size=(180, 180),
        batch_size=32)
    test_features, test_labels = get_features_and_labels(test_dataset)
    print("test_features", test_features.shape)
    loss, accu = model.evaluate(test_features, test_labels)
    print(f"Accuracy {accu: .3f}")

train()
# test_model()
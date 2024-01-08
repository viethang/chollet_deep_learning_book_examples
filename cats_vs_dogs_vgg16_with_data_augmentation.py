from tensorflow import keras
import numpy as np
import pathlib
from matplotlib import pyplot as plt


data_dir = pathlib.Path(
    "data/cats_vs_dogs_small")


def train():
    train_dataset = keras.utils.image_dataset_from_directory(
        data_dir / "train",
        image_size=(180, 180),
        batch_size=32)
    validation_dataset = keras.utils.image_dataset_from_directory(
        data_dir / "validation",
        image_size=(180, 180),
        batch_size=32)

    model = get_model()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath="models/feature_extraction_data_augmentation.keras",
                                        save_best_only=True,

                                        monitor='val_loss'),
        keras.callbacks.TensorBoard(
            log_dir="logs/feature_extraction_data_augmentation_logs")
    ]
    history = model.fit(train_dataset,
                        epochs=20,
                        validation_data=validation_dataset,
                        callbacks=callbacks)

    plot_history(history)


def get_model():
    conv_base = keras.applications.vgg16.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(180, 180, 3))
    conv_base.trainable = False
    conv_base.summary()

    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.2),
    ])

    inputs = keras.Input(shape=(180, 180, 3))
    x = data_augmentation(inputs)
    x = keras.layers.Lambda(
        lambda x: keras.applications.vgg16.preprocess_input(x))(x)
    x = conv_base(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

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
    model = keras.models.load_model(
        "models/feature_extraction_data_augmentation.keras", safe_mode=False)
    model.summary()
    test_dataset = keras.utils.image_dataset_from_directory(
        data_dir / "test",
        image_size=(180, 180),
        batch_size=32)
    loss, accu = model.evaluate(test_dataset)
    print(f"Accuracy {accu: .3f}")


# train()
test_model()

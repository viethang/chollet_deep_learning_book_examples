import matplotlib.pyplot as plt
from tensorflow import keras
import pathlib

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.2)
])


def get_model():
  model = keras.Sequential([
      keras.Input(shape=(180, 180, 3)),
      keras.layers.RandomFlip("horizontal"),
      keras.layers.RandomRotation(0.1),
      keras.layers.RandomZoom(0.2),
      keras.layers.Rescaling(1./255),
      keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
      keras.layers.Flatten(),
      keras.layers.Dense(1, activation='sigmoid')
  ])
  return model


data_dir = pathlib.Path(
    "data/cats_vs_dogs_small")


def display_augmented_images():
  train_dataset = keras.utils.image_dataset_from_directory(
      data_dir / "train",
      image_size=(180, 180),
      batch_size=32)

  plt.figure(figsize=(10, 10))
  for images, _ in train_dataset.take(1):
    for i in range(9):
      augmented_images = data_augmentation(images)
      ax = plt.subplot(3, 3, i+1)
      plt.imshow(augmented_images[0].numpy().astype("uint8"))
      plt.axis("off")
  plt.show()


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
  model.summary()
  model.compile(loss='binary_crossentropy',
                optimizer='rmsprop', metrics=['accuracy'])
  callbacks = [
      keras.callbacks.ModelCheckpoint(
          filepath="models/convnet_augmented.keras",
          save_best_only=True,
          monitor="val_loss")
  ]
  history = model.fit(
      train_dataset,
      epochs=100,
      validation_data=validation_dataset,
      callbacks=callbacks)
  plot_history(history)


def plot_history(history):
  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  epochs = range(1, len(accuracy) + 1)
  plt.plot(epochs, accuracy, "bo", label="Training accuracy")
  plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
  plt.title("Training and validation accuracy")
  plt.legend()
  plt.figure()
  plt.plot(epochs, loss, "bo", label="Training loss")
  plt.plot(epochs, val_loss, "b", label="Validation loss")
  plt.title("Training and validation loss")
  plt.legend()
  plt.show()

train()


def test_model():
  model = keras.models.load_model("models/convnet_augmented.keras")
  model.summary()
  test_dataset = keras.utils.image_dataset_from_directory(
      data_dir / "test",
      image_size=(180, 180),
      batch_size=32)

  loss, accu = model.evaluate(test_dataset)
  print(f"Test accuracy: {accu: .3f}")


test_model()

# display_augmented_images()

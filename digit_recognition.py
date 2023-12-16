from tensorflow.keras.datasets import mnist

from tensorflow import keras
from tensorflow.keras import layers
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

model = keras.Sequential([
  layers.Dense(512, activation="relu"),
  layers.Dense(10, activation="softmax")
])

model.compile(
  optimizer="rmsprop",
  loss="sparse_categorical_crossentropy",
  metrics=["accuracy"]
)

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print("Prediction 0", predictions[0])
print ("Prediction 0 argmax", predictions[0].argmax())
print("Test label 0",test_labels[0])
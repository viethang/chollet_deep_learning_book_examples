from tensorflow.keras.datasets import imdb
# IMDB dataset: a set of 50,000 highly polarized reviews from the
# Internet Movie Database. They’re split into 25,000 reviews for training and 25,000
# reviews for testing, each set consisting of 50% negative and 50% positive reviews.

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# only keep the top 10,000 most frequently occurring words in the training data
# label = 0 means negative review, 1 means positive review
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

# word_index = imdb.get_word_index()
# reverse_word_index = dict(
#     [(value, key) for (key, value) in word_index.items()])
# decoded_review = " ".join(
#     [reverse_word_index.get(i - 3, "?") for i in train_data[0]])
# print("decoded", decoded_review)

# Multi-hot encode your lists to turn them into vectors of 0s and 1s.
# This would mean, for instance, turning the sequence [8, 5] into a 10,000-dimensional vector
# that would be all 0s except for indices 8 and 5, which would be 1s. 
def vectorize_sequence(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    for j in sequence:
      results[i, j] = 1
  return results


x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

model = keras.Sequential([
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

# model.compile(optimizer="rmsprop", loss="binary_crossentropy",
#               metrics=["accuracy"])

# x_val = x_train[:10000]
# partial_x_train = x_train[10000:]
# y_val = y_train[:10000]
# partial_y_train = y_train[10000:]

# history = model.fit(partial_x_train, partial_y_train,
#           epochs=20,
#           batch_size=512,
#           validation_data=(x_val, y_val)
#           )


# history_dict = history.history
# loss_values = history_dict["loss"]
# val_loss_values = history_dict["val_loss"]
# epochs = range(1, len(loss_values) + 1)
# plt.plot(epochs, loss_values, "bo", label="Training loss")
# plt.plot(epochs, val_loss_values, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# plt.clf()
# acc = history_dict["accuracy"]
# val_acc = history_dict["val_accuracy"]
# plt.plot(epochs, acc, "bo", label="Training acc")
# plt.plot(epochs, val_acc, "b", label="Validation acc")
# plt.title("Training and validation accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

model.compile(optimizer="rmsprop",
loss="binary_crossentropy",
metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)
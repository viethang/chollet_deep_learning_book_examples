import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class)

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
targets = np.vstack(
    (np.zeros((num_samples_per_class, 1), dtype="float32"),
     np.ones((num_samples_per_class, 1), dtype="float32"))
)
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
# plt.show()


input_dim = 2
output_dim = 1
W = tf.Variable(initial_value = tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value = tf.zeros(shape=(output_dim,)))

def model(inputs):
  return tf.matmul(inputs, W) + b

def square_loss(predictions, targets):
  per_sample_loss = tf.square(predictions - targets)
  return tf.reduce_mean(per_sample_loss)

learning_rate = 0.1
def one_training_step(inputs):
  with tf.GradientTape() as tape:
    predictions = model(inputs)
    loss = square_loss(predictions, targets)
  gradient_w, gradient_b = tape.gradient(loss, [W,b])
  W.assign_sub(gradient_w * learning_rate)
  b.assign_sub(gradient_b * learning_rate)
  return loss
  
epoch_num = 10

for epoch_counter in range(epoch_num):
  print(f"Training epoch {epoch_counter}")
  loss = one_training_step(inputs)
  print(f"Loss {loss: .3f}")
  
predictions = model(inputs)  
x = np.linspace(-1, 4, 100)
y = - W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
  
  
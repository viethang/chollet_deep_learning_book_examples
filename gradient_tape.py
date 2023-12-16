import tensorflow as tf
import numpy as np

x = tf.Variable(2.)
with tf.GradientTape() as tape1:
  y = 2*x + 3
  grad_y_wrt_x = tape1.gradient(y, x)

with tf.GradientTape() as tape2:
  z = x**2 + x + 1
  grad_z_wrt_x = tape2.gradient(z, x)

print("grad(y,x)", grad_y_wrt_x)
print("grad(z,x)", grad_z_wrt_x)


W = tf.Variable(tf.random.uniform((2, 2)))
b = tf.Variable(tf.zeros((2,)))
x = tf.random.uniform((2, 2))
print("W", W)
print("b", b)
print("x", x)
with tf.GradientTape() as tape:
  y = tf.matmul(x, W) + b
  
print("matmul", tf.matmul(x, W))
grad_of_y_wrt_W_and_b = tape.gradient(y, [W, b])
print("grad_of_y_wrt_W_and_b", grad_of_y_wrt_W_and_b)

x = tf.Variable(1.)
with tf.GradientTape() as tape:
  y = [x**2, 2 * x]
  print("grad_y_wrt_x", tape.gradient(y, x))
  
time = tf.Variable(0.)
with tf.GradientTape() as outer_tape:
  with tf.GradientTape() as inner_tape:
    position = 4.9 * time ** 2
  speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)
print("acceleration", acceleration)
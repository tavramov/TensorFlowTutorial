# Getting Started With TensorFlow Tutorial

# 1. The Computational Graph
import tensorflow as tf
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1,node2)
sess = tf.Session()
print(sess.run([node1,node2]))

node3 = tf.add(node1,node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

loss = tf.reduce_sum(tf.square(linear_model - y)) # sun of the squares
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# 2. tf.train API
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
print(sess.run([W, b]))
for i in range(1000):
    sess.run(train, {x: [1,2,3,4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))
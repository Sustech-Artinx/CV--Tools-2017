import tensorflow as tf
import cv2
import numpy as np
import random
data = []
txt = open("label\\label.txt")
for i in range(300):
    s = txt.readline().split(" ")
    data.append([cv2.imread("data\\" + str(i) + ".png", 0).reshape(90000), [int(s[2]), int(s[3])], [int(s[1])] ])

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    ret = tf.Variable(initial, name=name)
    return ret

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 90000], name="input")
y_ = tf.placeholder(tf.float32, [None, 2], name="coordinate")
target_ = tf.placeholder(tf.float32, [None, 1], name="target")
with tf.name_scope("conv1"):
    W_conv1 = weight_variable([5, 5, 1, 4], name="W_conv1")
#    tf.summary.image("weight", tf.unpack(W_conv1, axis=3), max_outputs=32)
    b_conv1 = bias_variable([1], name="b_conv1")
#    tf.summary.histogram("bias", b_conv1)
W_conv2 = weight_variable([5, 5, 4, 8], name="W_conv2")
b_conv2 = bias_variable([8], name="b_conv2")
W_fc1 = weight_variable([75 * 75 * 8, 2], name="W_fc1")
b_fc1 = bias_variable([2], name="b_fc1")
W_fc2 = weight_variable([75 * 75 * 8, 10], name="W_fc2")
b_fc2 = bias_variable([10], name="b_fc2")
W_fc3 = weight_variable([10, 1], name="W_fc3")
b_fc3 = bias_variable([1], name="b_fc3")


'''
saver = tf.train.import_meta_graph('model.ckpt.meta')
saver = tf.get_default_graph()
W_conv1 = saver.get_tensor_by_name('conv1/W_conv1:0')
b_conv1 = saver.get_tensor_by_name('conv1/b_conv1:0')
W_conv2 = saver.get_tensor_by_name('W_conv2:0')
b_conv2 = saver.get_tensor_by_name('b_conv2:0')
W_fc1 = saver.get_tensor_by_name('W_fc1:0')
b_fc1 = saver.get_tensor_by_name('b_fc1:0')
'''
x_image = tf.reshape(x, [-1,300,300,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 75 * 75 * 8])
y_conv = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
fc2_out = tf.sigmoid(tf.matmul(h_pool2_flat, W_fc2) + b_fc2)
silence = tf.sigmoid(tf.matmul(fc2_out, W_fc3) + b_fc3)

position_dist = tf.reduce_sum(tf.abs(tf.sub(y_conv, y_)), 1, keep_dims=True)
type_dist = tf.abs(tf.sub(silence, target_))
type_true = tf.equal(target_, 1)
dist = tf.mul(tf.to_float(type_true), position_dist)
train_dist_step = tf.train.AdamOptimizer(1e-4).minimize(dist)
train_type_step = tf.train.AdamOptimizer(1e-4).minimize(type_dist)
with tf.name_scope("train_accuracy"):
    global_accuracy = tf.reduce_mean(dist)
    position_accuracy = tf.reduce_mean(tf.mul(position_dist, target_))
    type_accuracy = tf.reduce_mean(type_dist)
#    tf.summary.scalar("global_accuracy", global_accuracy)
#    tf.summary.scalar("position_accuracy", position_accuracy)
#    tf.summary.scalar("type_accuracy", type_accuracy)

#merged = tf.summary.merge_all()
#writer = tf.summary.FileWriter("./summary", sess.graph)
sess.run(tf.global_variables_initializer())
''''
saver = tf.train.import_meta_graph('model.ckpt.meta')
saver = tf.get_default_graph()
W_conv1 = saver.get_tensor_by_name('conv1/W_conv1:0')
b_conv1 = saver.get_tensor_by_name('conv1/b_conv1:0')
W_conv2 = saver.get_tensor_by_name('W_conv2:0')
b_conv2 = saver.get_tensor_by_name('b_conv2:0')
W_fc1 = saver.get_tensor_by_name('W_fc1:0')
b_fc1 = saver.get_tensor_by_name('b_fc1:0')
'''
for i in range(5):
    batch = random.sample(data[0: 250], 5)

    f = []
    l = []
    t = []
    for j in range(5):
        f.append(batch[j][0])
        l.append(batch[j][1])
        t.append(batch[j][2])
        #train_dist_step.run(feed_dict={x: f, y_: l, target_ : t})
        train_type_step.run(feed_dict={x: f, y_: l, target_ : t})
    if i % 20 == 0:
        train_accuracy = global_accuracy.eval(feed_dict={x: f, y_: l, target_ : t})
        train_position_accuracy = position_accuracy.eval(feed_dict={x: f, y_: l, target_ : t})
        train_type_accuracy = type_accuracy.eval(feed_dict={x: f, y_: l, target_ : t})
        print("step %d, global accuracy %g, dist accuracy %g, type accuracy %g"
              % (i, train_accuracy, train_position_accuracy, train_type_accuracy))
        #result = sess.run(merged, feed_dict={x: f, y_: l, target_ : t})
        #writer.add_summary(result, i)

f = []
l = []
t = []
for i in range(20):
    f.append(data[250 + i][0])
    l.append(data[250 + i][1])
    t.append(data[250 + i][2])
res = sess.run(y_conv, feed_dict={x: f, y_: l, target_ : t})
sig = sess.run(silence, feed_dict={x: f, y_: l, target_ : t})

for i in range(len(res)):
    print(str(sig[i]) + " " + str(res[i]) + " " + str(t[i]) + " " + str(l[i]))
print("test accuracy %g" % global_accuracy.eval(feed_dict={x: f, y_: l, target_ : t}))
'''
saver = tf.train.Saver()
save_path = saver.save(sess, "model.ckpt")
print("Model saved in file: %s" % save_path)
'''

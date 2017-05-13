from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import cv2
import numpy as np


class Digit:
    sess = None
    y_conv = None
    x = None
    keep_prob = None
    def __init__(self):

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

        with tf.name_scope("conv1"):
            W_conv1 = weight_variable([5, 5, 1, 32], name="W_conv1")
            tf.summary.image("weight", tf.unpack(W_conv1, axis=3), max_outputs=32)
            b_conv1 = bias_variable([32], name="b_conv1")
            tf.summary.histogram("bias", b_conv1)

        self.x = tf.placeholder(tf.float32, [None, 784], name="input")
        y_ = tf.placeholder(tf.float32, [None, 10], name="label")
        x_image = tf.reshape(self.x, [-1,28,28,1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64], name="W_conv2")
        b_conv2 = bias_variable([64], name="b_conv2")

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024], name="W_fc1")
        b_fc1 = bias_variable([1024], name="b_fc1")

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = weight_variable([1024, 10], name="W_fc2")
        b_fc2 = bias_variable([10], name="b_fc2")

        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_conv, y_))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()


        saver.restore(self.sess, save_path="./model.ckpt")

    def decode(self, img):

        temp = self.sess.run(self.y_conv, feed_dict={self.x: img, self.keep_prob: 1.0})
        return temp.argmax()

digit = Digit()
img = cv2.imread("C:\\Users\\lenovo\\Desktop\\Tensorflow\\pic2num-nonconv\\2.1\\8_out.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.resize(gray, (28, 28)).reshape((1, 784))
result = digit.decode(gray)

cv2.imshow("t", img)
print(result)
cv2.waitKey()



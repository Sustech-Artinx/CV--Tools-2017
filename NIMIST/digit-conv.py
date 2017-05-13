from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


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

x = tf.placeholder(tf.float32, [None, 784], name="input")
y_ = tf.placeholder(tf.float32, [None, 10], name="label")
x_image = tf.reshape(x, [-1,28,28,1])

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

keep_prob = tf.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10], name="W_fc2")
b_fc2 = bias_variable([10], name="b_fc2")

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
with tf.name_scope("train_accuracy"):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("train_accuracy", accuracy)

sess = tf.InteractiveSession()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./graph", sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(50):
    batch = mnist.train.next_batch(50)
    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    result = sess.run(merged, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    writer.add_summary(result, i)
#print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

saver = tf.train.Saver()
save_path = saver.save(sess, "./model.ckpt")
print("Model saved in file: %s" % save_path)
'''
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
'''
'''
temp = sess.run(W_conv1)
for i in range(32):
    plt.subplot(8, 4, i + 1); plt.imshow(temp[:, :, 0, i], cmap ='gray')
    plt.axis('off')
'''

'''
sample = mnist.train.next_batch(100)
for i in range(25):
    plt.subplot(5, 5, i + 1); plt.imshow(numpy.array(sample[0][i]).reshape(28, 28), cmap ='gray')
    plt.axis('off')
    plt.title(sample[1][i].argmax())
'''
import tensorflow as tf
import numpy as np
import matplotlib as plt
from sys import argv
from PIL import Image

MNIST_DATA_SIZE = 28 * 28
DEFAULT_MODEL_DIR = "./autoencoder_models/"
LEARNING_RATE = 0.001

class Autoencoder:
    def __init__(self, input_size, hidden_size, model_name = "model"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model_name = model_name
        
        self.sess = tf.Session()

        #with tf.device("/gpu:0"):
        self.__create_model__(input_size, hidden_size)

        self.saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(model_name)
        if checkpoint is not None:
            self.saver.restore(self.sess, checkpoint)
            self.epoch = int(checkpoint[checkpoint.rfind('-')+1:])
            print("Restoring old model!")
        else:
            self.sess.run(tf.global_variables_initializer())
            self.epoch = 1
            print("Created new model...")

    def __create_model__(self, input_size, hidden_size):
        self.X = tf.placeholder(dtype=tf.float32, shape=[input_size])

        input_x = tf.reshape(self.X, [1, input_size])

        hidden_layer = tf.layers.dense(
            inputs=input_x,
            units=hidden_size,
            activation=tf.nn.relu)

        output_layer = tf.layers.dense(
            inputs=hidden_layer,
            units=input_size,
            activation=tf.nn.relu)

        output_layer = tf.reshape(output_layer, [1, input_size])

        #self.Y = tf.nn.sigmoid(output_layer)
        self.Y = tf.nn.tanh(output_layer)

        self.cost = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=input_x,
                logits=output_layer))

        self.train_op = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)

    def fit(self, train_x):
        return self.sess.run((self.train_op, self.cost), feed_dict={self.X: train_x})

    def save(self):
        self.epoch += 1
        self.saver.save(self.sess, self.model_name, global_step=self.epoch)

    def eval(self, x):
        return self.sess.run(self.Y, feed_dict={self.X: x})


# Load mnist dataset from Tensorflow
def load_mnist():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    return mnist.train.images # Returns np.array


def test_internal(encoder, test_x):
    y = encoder.eval(test_x)

    test_x = np.dot(test_x, 256).astype(np.uint8).reshape((28,28))
    im_x = Image.fromarray(test_x)
    im_y = Image.fromarray(y.dot(256).astype(np.uint8).reshape((28, 28)))
    im_x.show()
    im_y.show()


def train(num_epochs):
    train_x = load_mnist()

    encoder = Autoencoder(MNIST_DATA_SIZE, int(MNIST_DATA_SIZE * 0.67), DEFAULT_MODEL_DIR)
    epoch = encoder.epoch

    print("epoch: %d" % (epoch))
    for e in range(num_epochs):
        total_cost = 0
        for i in range(len(train_x)):
            total_cost = encoder.fit(train_x[i])[1]
            if i % 1000 == 0 and i > 0:
                print("Evaluated: %d images, cost: %f" % (i, total_cost / i))
        
        print("epoch: %d, cost: %f" % (epoch + e, total_cost / len(train_x)))
        encoder.save()
        test_internal(encoder, train_x[epoch])

    
def test():
    train_x = load_mnist()

    encoder = Autoencoder(MNIST_DATA_SIZE, int(MNIST_DATA_SIZE * 0.67), DEFAULT_MODEL_DIR)
    print("epoch: %d" % (encoder.epoch))

    test_internal(encoder, train_x[encoder.epoch])


# Main
if __name__ == "__main__":

    num_args = len(argv)
    if num_args > 1 and argv[1] == 'train':
        print("training model...")
        train(1)
    elif num_args > 1 and argv[1] == 'test':
        print("verifying model...")
        test()
    else:
        print("valid arguments: train or test")

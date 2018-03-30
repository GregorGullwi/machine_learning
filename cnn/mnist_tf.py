import tensorflow as tf
import numpy as np

MNIST_DATA_SIZE = 28
DEFAULT_MODEL_DIR = "cnn_models"
LEARNING_RATE = 0.001
NUM_CONV_LAYERS = 2
NUM_FEATURES_PER_CONV_LAYER = 32

# Convolusional neural network model to use
def cnn_model_fn(features, labels, mode):
    # Uniform size
    current_layer_size = MNIST_DATA_SIZE
    
    # Input layer
    layer = tf.reshape(features["x"], [-1, current_layer_size, current_layer_size, 1]) #batch_size, width, height, channels

    num_features_for_current_layer = NUM_FEATURES_PER_CONV_LAYER
    for i in range(NUM_CONV_LAYERS):
        # Convolusion layer x[i]
        layer = tf.layers.conv2d(
            inputs=layer,
            filters=num_features_for_current_layer,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu)

        # Pool layer x[i]
        layer = tf.layers.max_pooling2d(
            inputs=layer,
            pool_size=[2, 2],
            strides=2)
        
        current_layer_size = int(current_layer_size / 2)

        if i != NUM_CONV_LAYERS - 1:
            num_features_for_current_layer = num_features_for_current_layer * 2

    # Flatten pool to be able to connect to a fully dense layer
    layer = tf.reshape(
        layer,
        [-1, current_layer_size * current_layer_size * num_features_for_current_layer])

    # Dense layer
    layer = tf.layers.dense(
        inputs=layer,
        units=1024,#current_layer_size,
        activation=tf.nn.relu)

    # Random dropout layer
    layer = tf.layers.dropout(
        inputs=layer,
        rate=0.1,
        training=(mode == tf.estimator.ModeKeys.TRAIN))
    
    # Output layer
    layer = tf.layers.dense(
        inputs=layer,
        units=10)

    classes = tf.argmax(input=layer, axis=1)
    predictions = {
        "classes" : classes,
        "probabilities" : tf.nn.softmax(layer, name="softmax_tensor")
    }

    # Predict mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions)

    onehot_labels = tf.one_hot(
        indices=tf.cast(labels, tf.int32),
        depth=10)
        
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels,
        logits=layer)

    # Training mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=LEARNING_RATE)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op)

    # Test/Eval mode
    eval_metric_ops = {
        "accuracy" : tf.metrics.accuracy(labels=labels, predictions=classes)
    }

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=loss,
        eval_metric_ops=eval_metric_ops)

# Load mnist dataset from Tensorflow
def load_mnist(is_training):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    if (is_training):
        train_data = mnist.train.images # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        return train_data, train_labels
    
    test_data = mnist.test.images # Returns np.array
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    return test_data, test_labels


# Train new/confinue training model
def train():
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=DEFAULT_MODEL_DIR)
    
    train_data, train_labels = load_mnist(True)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    logging_hook = tf.train.LoggingTensorHook(
        tensors={"probabilities": "softmax_tensor"},
        every_n_iter=50)
    
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps = 20000,
        hooks=[logging_hook])


# Test previously trained model
def test():
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=DEFAULT_MODEL_DIR)

    test_data, test_labels = load_mnist(False)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)

    results = mnist_classifier.evaluate(input_fn=test_input_fn)
    print(results)


# Predict with previously trained model
def predict():
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=DEFAULT_MODEL_DIR)

    test_data, test_labels = load_mnist(False)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":test_data},
        num_epochs=1,
        shuffle=False)

    predictions = list(mnist_classifier.predict(input_fn=predict_input_fn))

    output = []
    i = 1
    for p in predictions:
        output += [(i, p['classes'])]
        i += 1
    #print(output)
    np.savetxt('test_predictions.csv', output, fmt='%d', delimiter=',', header="ImageId,Label")


# Main
def main(args):
    
    num_args = len(args)
    if num_args > 1 and args[1] == 'train':
        print("training model...")
        train()
    elif num_args > 1 and args[1] == 'test':
        print("verifying model...")
        test()
    elif num_args > 1 and args[1] == 'predict':
        print("predicting with model...")
        predict()
    else:
        print("valid arguments: train or test")
        

tf.app.run()

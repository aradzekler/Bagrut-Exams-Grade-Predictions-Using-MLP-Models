import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import seaborn as sns  # data visualization
import HelperFunctions as hf
import Scalers
import TrainTest
from pathlib import Path
from datetime import datetime

'''
###############################
MULTILAYER PERCEPTRON MODEL TEST
###############################
'''

'''
BOILERPLATE
'''
# eager execution feature is making problems
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

data_folder_xlsx = Path("schoolDbEng.xlsx")
data_folder_csv = Path("schoolDBcsv.csv")

# school_dataset.info() to display information
school_dataset = pd.read_csv(data_folder_csv, encoding='utf-8')

df = pd.DataFrame(school_dataset)  # dataframe for easier handling of the data.
hf.refractor_df(df)

'''
DEFINE SCALER
'''
# standard scaler doesnt work well here because our data is not distributed
# minmax scaler doesnt work well because it shrinks some of the data too much so numbers are rounded up to 0 or 1.
# the robust scaler works pretty well and arranges our data between -10 and 10, is nice!
Y = df['avg_final_grades'].copy()  # Y label: dependent variable avg_final_grades
X = df.drop(columns=['profession', 'city_name', 'school_name']).to_numpy()  # X label: features, dropping named features
# need to convert the dataframe to a numpy array because of native function.
Scalers.RobustScaler().fit(X).transform(X)

'''
SPLIT THE DATA INTO TRAIN AND TEST
'''
np.random.seed(70)  # random seed for train test split
X_train, X_test, Y_train, Y_test = TrainTest.train_test_split(X, Y, test_size=0.2, random_state=13)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

'''
NEURAL NETWORK PARAMETERS
'''
n_hidden_layers = 10  # neurons inside hidden layer
n_train = X_train.shape[0]
n_test = X_test.shape[0]
n_layer_0 = X_train.shape[1]  # 7 numeric columns as input layer
n_layer_1 = n_hidden_layers  # 2 hidden layers with 20 neurons
n_layer_2 = n_hidden_layers
n_layer_3 = 1  # output layer
learning_rate = 0.0005  # our starting learning rate

print('Training samples: ' + str(n_train) + '\nTest Samples: ' + str(n_test))

# create variables for features and prediction
X = tf.compat.v1.placeholder(tf.float32, [None, n_layer_0], name="features")
Y = tf.compat.v1.placeholder(tf.float32, [None, 1], name="output")
Weights = {  # outputs random values from a normal distribution.
    'W1': tf.Variable(tf.random.normal([n_layer_0, n_layer_1], stddev=0.01), name='W1'),
    'W2': tf.Variable(tf.random.normal([n_layer_1, n_layer_2], stddev=0.01), name='W2'),
    'W3': tf.Variable(tf.random.normal([n_layer_2, n_layer_3], stddev=0.01), name='W3')
}
Biases = {
    'b1': tf.Variable(tf.random.normal([n_layer_1]), name='b1'),
    'b2': tf.Variable(tf.random.normal([n_layer_2]), name='b2'),
    'b3': tf.Variable(tf.random.normal([n_layer_3]), name='b3')
}

'''
DEFINING THE MODEL
'''
# name_scope is a context manager which allows us to refer to tensors and how the graph shows in TensorBoard
# activation functions inside hidden layer are ReLu and function in the output layer is Sigmoid.
def multilayer_perceptron_model(X, W, b):
    with tf.name_scope('hidden_layer_1'):
        layer_1 = tf.add(tf.matmul(X, W['W1']), b['b1'])
        layer_1 = tf.nn.relu(layer_1)
    with tf.name_scope('hidden_layer_2'):
        layer_2 = tf.add(tf.matmul(layer_1, W['W2']), b['b2'])
        layer_2 = tf.nn.relu(layer_2)
    with tf.name_scope('layer_output'):
        layer_3 = tf.add(tf.matmul(layer_2, W['W3']), b['b3'])
        # softmax for classification, sigmoid for true values, linear for regression (when values are unbounded)
        layer_3 = tf.nn.sigmoid(layer_3)
        return layer_3


# the prediction
with tf.name_scope("MLP"):
    Y_pred = multilayer_perceptron_model(X, Weights, Biases)

# measuring loss preformace, every call to 'loss' will activate l1 loss function (LAD)
# we will use LAD when there are outliers and MSE when no outliers are present.
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.abs(Y - Y_pred), name='loss')
    # mse = tf.reduce_mean(tf.square(Y - Y_pred), name='MSE')

# using the Adam optimizer which is using adaptive learning rate
# methods to find individual learning rates for each parameter instead of using
# exponential decay for decaying our learning rate.
# we will also want to minimize our loss function here.
with tf.name_scope("train"):
    training_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# our accuracy every epoch
with tf.name_scope("eval"):
    correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

'''
EXECUTING THE MODEL
'''
# define some parameters
n_epochs = 50
display_epoch = 5  # between how many epochs we want to display results.
batch_size = 1024
n_batches = int(len(X_train) / batch_size)

# set up the directory to store the results for tensorboard
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# store results through every epoch
mse_train_list = []
mse_test_list = []
learning_list = []
prediction_results = []


# merged_summary = tf.compat.v1.summary.merge_all()

print("FINISHED PREPARATIONS, EXECUTING MODEL")

# todo: finish the model, visualize
with tf.compat.v1.Session() as sess:
    with tf.device("/cpu:0"):
        print("ENTERING SESSION")

        # write logs for tensorboard
        # summary_writer = tf.compat.v1.summary.FileWriter(logdir, graph=tf.compat.v1.get_default_graph())
        summary_writer = tf.summary.create_file_writer(logdir)

        tf.compat.v1.global_variables_initializer().run()
        for epoch in range(n_epochs):
            idx = np.random.permutation(n_batches)
            X_random = X_train[idx]
            Y_random = Y_train[idx].values.reshape(-1, 1)
            for i in range(n_batches):
                X_batch = X_random[i * batch_size:(i + 1) * batch_size]
                Y_batch = Y_random[i * batch_size:(i + 1) * batch_size]

                sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch})
                # Write logs at every iteration
                with summary_writer.as_default():
                    tf.summary.scalar("loss", loss, step=epoch * n_batches + i)
                    tf.summary.scalar("learn_rate", learning_rate, step=epoch * n_batches + i)
                    tf.summary.scalar("accuracy", accuracy, step=epoch * n_batches + i)
                # Write logs at every iteration
                # summary_writer.add_summary(summary, epoch * n_batches + i)
            # measure performance and display the results
            if (epoch + 1) % display_epoch == 0:
                _mse_train = sess.run(loss, feed_dict={X: X_train, Y: Y_train.values.reshape(-1, 1)})
                _mse_test = sess.run(loss, feed_dict={X: X_test, Y: Y_test.values.reshape(-1, 1)})
                # append to list for displaying
                mse_train_list.append(_mse_train)
                mse_test_list.append(_mse_test)
                learning_list.append(0.5)  # TODO: LEARNING RATE PRINTING

                # Save model weights to disk for reproducibility
                # saver = tf.compat.v1.train.Saver(max_to_keep=15)
                # saver.save(sess, "tf_checkpoints/epoch{:04}.ckpt".format((epoch + 1)))

                print("Epoch: {:04}\tTrainMSE: {:06.5f}\tTestMSE: {:06.5f}, Learning: {:06.7f}".format((epoch + 1),
                                                                                                       _mse_train,
                                                                                                       _mse_test,
                                                                                                       learning_list[
                                                                                                           -1]))
                prediction_results = sess.run(Y_pred, feed_dict={X: X_test})


'''
GRAPHING THE MODEL
'''
# set up legend
print("GRAPHING")
blue_patch = patches.Patch(color='blue', label='Train MSE')
red_patch = patches.Patch(color='red', label='Test MSE')
plt.legend(handles=[blue_patch, red_patch])
plt.grid()

plt.plot(mse_train_list, color='blue')
plt.plot(mse_test_list, color='red')

plt.xlabel('epochs (x{})'.format(display_epoch))
plt.ylabel('MSE [minimize]')

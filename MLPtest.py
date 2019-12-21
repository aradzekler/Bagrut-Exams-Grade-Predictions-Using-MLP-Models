import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
# DO NOT RUN MORE THEN 1 INSTANCE OF THIS CODE, OR ELSE IT WILL ERROR
'''
BOILERPLATE
'''
# eager execution feature is making problems
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

# our data paths
data_folder_xlsx = Path("schoolDbEng.xlsx")
data_folder_csv = Path("schoolDBcsv.csv")

# school_dataset.info() to display information
school_dataset = pd.read_csv(data_folder_csv, encoding='utf-8')

df = pd.DataFrame(school_dataset)  # dataframe for easier handling of the data.
hf.refractor_df(df)  # ordering our database

'''
DEFINE SCALER
'''
# standard scaler doesnt work well here because our data is not distributed
# minmax scaler doesnt work well because it shrinks some of the data too much so numbers are rounded up to 0 or 1.
# the robust scaler works pretty well and arranges our data between -10 and 10, is nice!
Y = df['avg_final_grades'].copy()  # Y label: dependent variable avg_final_grades
X = df.drop(columns=['profession', 'city_name',
                     'school_name']).to_numpy()  # X label: features, dropping named features replacing by numeric ones.
# need to convert the dataframe to a numpy array because of native function.
Scalers.RobustScaler().fit(X).transform(X)

'''
SPLIT THE DATA INTO TRAIN AND TEST
'''
np.random.seed(44)  # random seed for train test split
X_train, X_test, Y_train, Y_test = TrainTest.train_test_split(X, Y, test_size=0.2, random_state=13)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

'''
NEURAL NETWORK HYPERPARAMETERS
'''
n_hidden_layers = 7  # neurons inside hidden layer
n_train = X_train.shape[0]
n_test = X_test.shape[0]
n_layer_0 = X_train.shape[1]  # 7 numeric columns as input layer
n_layer_1 = n_hidden_layers
n_layer_2 = n_hidden_layers*2
n_layer_3 = 1  # output layer

init_learning_rate = 0.3  # our starting learning rate
global_step = tf.Variable(0, trainable=False)
decay_rate = 0.90
decay_step = 10000

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
        layer_1 = tf.nn.dropout(layer_1, rate=0.5)  # applying dropout after ReLu
    with tf.name_scope('hidden_layer_2'):
        layer_2 = tf.add(tf.matmul(layer_1, W['W2']), b['b2'])
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, rate=0.5)
    with tf.name_scope('layer_output'):
        layer_3 = tf.add(tf.matmul(layer_2, W['W3']), b['b3'])
        # softmax for classification, sigmoid for true values, linear for regression (when values are unbounded)
        # for some reason, the output function is messing our loss! we will get MUCH better results without.
        return layer_3


# adjusting the learning rate with exponential decay (adam optimizer
# is also doing that but even in the Adam optimization method,
# the learning rate is a hyperparameter and needs to be tuned
# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
learning_rate = tf.compat.v1.train.exponential_decay(init_learning_rate,
                                                     global_step, 10000,
                                                     0.95, staircase=True)

# the prediction
Y_prediction = multilayer_perceptron_model(X, Weights, Biases)

# measuring loss preformace, every call to 'loss' will activate l1 loss function (LAD)
# we will use LAD when there are outliers and MSE when no outliers are present.
loss = tf.reduce_mean(tf.abs(Y - Y_prediction), name='loss')  # LAD (l1)
# loss = tf.reduce_mean(tf.square(Y - Y_prediction), name='loss')  # MSE (l2)

# using the Adam optimizer which is using adaptive learning rate
# methods to find individual learning rates for each parameter instead of using
# exponential decay for decaying our learning rate.
training_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)


'''
EXECUTING THE MODEL
'''
# define some parameters
n_epochs = 40
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

print("FINISHED PREPARATIONS, EXECUTING MODEL")

with tf.compat.v1.Session() as sess:
    with tf.device("/cpu:0"):
        print("ENTERING SESSION")
        # our accuracy every epoch
        correct_prediction = tf.equal(tf.argmax(Y_prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # write logs for tensorboard
        # summary_writer = tf.compat.v1.summary.FileWriter(logdir, graph=tf.compat.v1.get_default_graph())
        summary_writer = tf.summary.create_file_writer(logdir)

        tf.compat.v1.global_variables_initializer().run()
        for epoch in range(n_epochs):  # TODO: find optimization for epoch
            idx = np.random.permutation(n_batches)
            X_random = X_train[idx]
            Y_random = Y_train[idx].values.reshape(-1, 1)
            for i in range(n_batches):
                X_batch = X_random[i * batch_size:(i + 1) * batch_size]
                Y_batch = Y_random[i * batch_size:(i + 1) * batch_size]

                _, p, acc = sess.run([training_op, Y_prediction, accuracy], feed_dict={X: X_batch, Y: Y_batch})

                # Write logs at every iteration
                with summary_writer.as_default():
                    tf.summary.scalar("loss", loss, step=epoch * n_batches + i)
                    tf.summary.scalar("learn_rate", learning_rate, step=epoch * n_batches + i)
                    tf.summary.scalar("accuracy", accuracy, step=epoch * n_batches + i)

            # measure performance and display the results
            if (epoch + 1) % display_epoch == 0:
                _mse_train = sess.run(loss, feed_dict={X: X_train, Y: Y_train.values.reshape(-1, 1)})
                _mse_test = sess.run(loss, feed_dict={X: X_test, Y: Y_test.values.reshape(-1, 1)})
                learning_list.append(sess.run(learning_rate))
                # append to list for displaying
                mse_train_list.append(_mse_train)
                mse_test_list.append(_mse_test)

                # Save model weights to disk for reproducibility
                saver = tf.compat.v1.train.Saver(max_to_keep=15)
                saver.save(sess, "tf_checkpoints/epoch{:04}.ckpt".format((epoch + 1)))

                print("Epoch: {:04}, Train loss: {:06.5f}, Test loss: {:06.5f}".format(
                        (epoch + 1),
                        _mse_train,
                        _mse_test))

'''
GRAPHING THE MODEL
'''

print("GRAPHING")
plt.figure(1)
blue_patch = patches.Patch(color='blue', label='Train loss')
red_patch = patches.Patch(color='red', label='Test loss')
plt.legend(handles=[blue_patch, red_patch])
plt.grid()
plt.plot(mse_train_list, color='blue')
plt.plot(mse_test_list, color='red')
plt.xlabel('epochs (x{})'.format(display_epoch))
plt.ylabel('loss [minimize]')

plt.figure(2)
blue_patch = patches.Patch(color='blue', label='Prediction')
red_patch = patches.Patch(color='red', label='Expected Value')
green_patch = patches.Patch(color='green', label='Abs Error')
plt.legend(handles=[blue_patch, red_patch, green_patch])
plt.grid()

x_array = np.arange(len(prediction_results))
plt.scatter(x_array, prediction_results, color='blue')
plt.scatter(x_array, Y_test, color='red')

abs_error = abs(Y_test - prediction_results.reshape(-1))
plt.plot(x_array, abs_error, color='green')

plt.xlabel('Epoch'.format(display_epoch))
plt.ylabel('Average Grade')


# changing column names for easier work.
df.rename(columns={'Final Grade Average': 'avg_final_grades',
                   'Number of Testees': 'num_of_testers',
                   'Yehidut Limud': 'units',
                   'Graduation': 'grad_year',
                   'Profession': 'profession',
                   'City Name': 'city_name',
                   'School Name': 'school_name',
                   'School ID': 'school_id'},
          inplace=True)

# trim spaces
df['profession'] = df['profession'].str.rstrip()
df['city_name'] = df['city_name'].str.rstrip()
df['school_name'] = df['school_name'].str.rstrip()
unique_prof = df['profession'].unique()
unique_cities = df['city_name'].unique()
unique_schools = df['school_name'].unique()
regular_features = 3
features = regular_features + unique_prof.size + unique_cities.size + unique_schools.size
# columns positions
index_testers = 0
index_units = 1
index_year = 2

# data normalization
norm_testers = 0
norm_units = 2
norm_year = 2012
# unique maps to assign unique id per feature to specific value
unique_prof_dict = dict(
    (val, index + regular_features) for index, val in enumerate(unique_prof))
unique_cities_dict = dict(
    (val, index + regular_features + unique_prof.size) for index, val in enumerate(unique_cities))
unique_schools_dict = dict(
    (val, index + regular_features + unique_prof.size + unique_cities.size) for index, val in
    enumerate(unique_schools))
while True:
    test_testers = int(input("Enter number of testers: "))
    test_units = int(input("Enter units: "))
    test_year = int(input("Enter year: "))

    test_profession = input("Enter profession: ")
    test_city = input("Enter city: ")
    test_school = input("Enter school: ")

    # test data
    data_input_x = np.zeros(
        shape=(1, features),
        dtype=float,
        order='F')
    data_input_x_raw = data_input_x[0]

    # 3 normal features units/year/num of testers
    data_input_x_raw[index_testers] = test_testers - norm_testers
    data_input_x_raw[index_units] = test_units - norm_units
    data_input_x_raw[index_year] = test_year - norm_year

    # take each raw features understand what his is "id"
    # and then assign value at his id position to 1
    data_input_x_raw[unique_prof_dict[test_profession]] = 1
    data_input_x_raw[unique_cities_dict[test_city]] = 1
    data_input_x_raw[unique_schools_dict[test_school]] = 1

    data_test_y = sess.run([Y_prediction], feed_dict={X: data_input_x})
    print('Estimated Grade {0} \n \n'.format(data_test_y))

import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import timeit

tf.disable_eager_execution()
school_data_set = pd.read_csv(Path("schoolDBcsv.csv"), encoding='utf-8')

'''
plt.figure() # plotting
sns.distplot(school_data_set['Final Grade Average'],bins=20,axlabel='Grades',kde=1,norm_hist=0)
plt.show()
'''

data_frame = pd.DataFrame(school_data_set)  # data frame for easier handling of the data.

# columns positions
index_testers = 0
index_units = 1
index_year = 2

# data normalization
norm_testers = 0
norm_units = 2
norm_year = 2012

# linear regression config
random_seed = 4
records = data_frame.shape[0]
batch_size = 1000
train_iteration_print_each = 1000
train_iteration_count = 25000  # 10000
train_percentage = 0.85

# time when load data start
load_data_start_time = timeit.default_timer()

# changing column names for easier work.
data_frame.rename(columns={'Final Grade Average': 'avg_final_grades',
                           'Number of Testees': 'num_of_testers',
                           'Yehidut Limud': 'units',
                           'Graduation': 'grad_year',
                           'Profession': 'profession',
                           'City Name': 'city_name',
                           'School Name': 'school_name',
                           'School ID': 'school_id'},
                  inplace=True)

# trim spaces
data_frame['profession'] = data_frame['profession'].str.rstrip()
data_frame['city_name'] = data_frame['city_name'].str.rstrip()
data_frame['school_name'] = data_frame['school_name'].str.rstrip()

# adding numeric columns to data frame instead of nasty hebrew ones.
unique_prof = data_frame['profession'].unique()
unique_cities = data_frame['city_name'].unique()
unique_schools = data_frame['school_name'].unique()

# numbers of features records and batch size
regular_features = 3
features = regular_features + unique_prof.size + unique_cities.size + unique_schools.size

# unique maps to assign unique id per feature to specific value
unique_prof_dict = dict(
    (val, index + regular_features) for index, val in enumerate(unique_prof))
unique_cities_dict = dict(
    (val, index + regular_features + unique_prof.size) for index, val in enumerate(unique_cities))
unique_schools_dict = dict(
    (val, index + regular_features + unique_prof.size + unique_cities.size) for index, val in
    enumerate(unique_schools))

# the data_x array with shape =(records, features)
# contain the regular_features feature for each city/school/profession
data_x = np.zeros(
    shape=(records, features),
    dtype=float,
    order='F')

# data_y contain the grades
data_y = np.zeros(
    shape=(records, 1),
    dtype=float,
    order='F')

# create a random stack and shuffle it
# in order to create a mapping for shuffling the data
stack = list(range(records))
random.Random(random_seed).shuffle(stack)

for index, row in data_frame.iterrows():
    # the random new index used to shuffle the data
    random_index = stack[index]

    # the data_y is the final grade
    data_y[random_index] = row['avg_final_grades']
    data_raw = data_x[random_index]

    # 3 normal features units/year/num of testers
    data_raw[index_testers] = row['num_of_testers'] - norm_testers
    data_raw[index_units] = row['units'] - norm_units
    data_raw[index_year] = row['grad_year'] - norm_year

    # take each raw features understand what his is "id"
    # and then assign value at his id position to 1
    data_raw[unique_prof_dict[row['profession']]] = 1
    data_raw[unique_cities_dict[row['city_name']]] = 1
    data_raw[unique_schools_dict[row['school_name']]] = 1

# time when load data end
load_data_end_time = timeit.default_timer()
print('Loading data Time: ', load_data_end_time - load_data_start_time)

# tensor flow linear regression model
x_ = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.zeros([features, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(x_, w) + b

# loss function and GradientDescentOptimizer
loss = tf.reduce_mean(tf.pow(y - y_, 2))
update = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

# init session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# the train data
train_records = int(records * train_percentage)
data_train_x = data_x[0:train_records, :]
data_train_y = data_y[0:train_records, :]

# resolve the sub array relevant for the test
test_records = records - train_records
date_test_x = data_x[train_records:records, :]
data_test_y = data_y[train_records:records, :]

# graphs for showing train&test plot
graph_train = []
graph_test = []

# time when train data start
train_data_start_time = timeit.default_timer()

for i in range(0, train_iteration_count):
    # resolve a start and end position according to the batch size
    data_start = batch_size * i % train_records
    data_end = batch_size * (i + 1) % train_records

    # in case when the end position is bigger then the start
    # because the batch_size is probably not a perfect divider to records amount
    # and if it is bigger decreasing it so it will be in the range
    if data_end < data_start:
        data_end = train_records

    # resolve the sub array according to batch start&end position
    sub_x = data_train_x[data_start:data_end, :]
    sub_y = data_train_y[data_start:data_end, :]

    # updating the session
    sess.run(update, feed_dict={x_: sub_x, y_: sub_y})

    # print progress each iteration_print_each iteration's
    if i % train_iteration_print_each == 0:
        # compute the loss
        loss_train = loss.eval(session=sess, feed_dict={x_: data_train_x, y_: data_train_y})
        loss_test = loss.eval(session=sess, feed_dict={x_: date_test_x, y_: data_test_y})

        # save the loses to show the graph
        graph_train.append(loss_train)
        graph_test.append(loss_test)

        # print the loss
        print('Iteration:', i, 'train loss:', loss_train, ' test loss:', loss_test)

# time when train data end
train_data_end_time = timeit.default_timer()
print('Training data Time: ', train_data_end_time - train_data_start_time,
      " batch size: ", batch_size, " trains: ", train_iteration_count)

# config the plot
patch_blue = patches.Patch(color='blue', label='Train MSE')
patch_red = patches.Patch(color='red', label='Test MSE')
plt.legend(handles=[patch_blue, patch_red])
plt.grid()

# label the axis
plt.xlabel('epochs (x{})'.format(train_iteration_print_each))
plt.ylabel('MSE [minimize]')

# print the result
plt.plot(graph_train, color='blue')
plt.plot(graph_test, color='red')

# show the result
plt.show()

# input from the user to manual test
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

    data_test_y = np.matmul(data_input_x, sess.run(w)) + sess.run(b)
    print('Estimated Grade {0} \n \n'.format(data_test_y))

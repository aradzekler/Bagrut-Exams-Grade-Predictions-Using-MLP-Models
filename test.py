import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns  # data visualization
from pathlib import Path


# HELPER FUNCTIONS#


# function to convert xlsx files to csv.
# path as C:\Users\97254\Desktop\deepLearningCourseProject\schoolDB.xlsx
def xlsx_to_csv(xlsx_file_path, csv_file_path):
    read_file = pd.read_excel(xlsx_file_path)
    read_file.to_csv(csv_file_path, index=False, header=True, encoding='utf-8')
    print("SUCCESS\ncsv created in path folder")


# function to check how much values are missing per column
def percentage_missing(data_set):
    if isinstance(data_set, pd.DataFrame):
        dictionary = {}  # dictionary that conatins keys columns names and values percentage of missing values in the columns
        for col in data_set.columns:
            dictionary[col] = (np.count_nonzero(data_set[col].isnull()) * 100) / len(data_set[col])
        return pd.DataFrame(dictionary, index=['% of missing'], columns=dictionary.keys())
    else:
        raise TypeError("can only be used with panda dataframe")


# making the dataframe easier.
def refactor_data_frame(data_frame):
    # adding numeric columns to dataframe instead of nasty hebrew ones.
    unique_cities = data_frame['City Name'].unique()
    unique_prof = data_frame['Profession'].unique()

    unique_cities_dict = dict(
        (val, index) for index, val in enumerate(unique_cities))  # numerical values for cities. by order (308 cities)
    unique_prof_dict = dict((val, index + 500) for index, val in enumerate(unique_prof))  # values above 500 by order.

    data_frame['city_id'] = data_frame['City Name'].map(unique_cities_dict)  # putting new columns into the dataframe.
    data_frame['prof_id'] = data_frame['Profession'].map(unique_prof_dict)

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


# setting up train and test sets (4/5 to train and 1/5 to test)
def train_and_test_div(data):
    msk = np.random.rand(len(data)) < 0.8
    train_set = data[msk]
    test_set = data[~msk]
    print(len(train_set), 'train examples')
    print(len(test_set), 'test examples')


data_folder_xlsx = Path("schoolDbEng.xlsx")
data_folder_csv = Path("schoolDBcsv.csv")

school_data_set = pd.read_csv(data_folder_csv, encoding='utf-8')
school_data_set.info()

'''
plt.figure() # plotting
sns.distplot(school_data_set['Final Grade Average'],bins=20,axlabel='Grades',kde=1,norm_hist=0)
plt.show()
'''

# features
categorical_vars = ['city_id', 'school_id', 'units', 'profession', 'grad_year']
continuous_vars = ['avg_final_grades', 'num_of_testers', ]

df = pd.DataFrame(school_data_set)  # dataframe for easier handling of the data.


# refactor_data_frame(df)
# print(df)

# making the data frame easier.
def linear_reg(data_frame):
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
    records = data_frame.size
    batch_size = 1000
    train_iteration_print_each = 100
    train_iteration_count = 2500  # 10000
    train_percentage = 0.85
    train_records = int(records * train_percentage)

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

    for index, row in df.iterrows():
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

    # tensor flow linear regression model
    x_ = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, 1])
    w = tf.Variable(tf.zeros([features, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x_, w) + b

    # loss function and GradientDescentOptimizer
    loss = tf.reduce_mean(tf.pow(y - y_, 2))
    update = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # init session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(0, train_iteration_count):
        # resolve a start and end position according to the batch size
        data_start = batch_size * i % train_records
        data_end = (batch_size + 1) * i % train_records

        # in case when the end position is bigger then the start
        # because the batch_size is probably not a perfect divider to records amount
        # and if it is bigger decreasing it so it will be in the range
        if data_end < data_start:
            data_end = train_records

        # resolve the sub array according to batch start&end position
        sub_x = data_x[data_start:data_end, :]
        sub_y = data_y[data_start:data_end, :]

        # updating the session
        sess.run(update, feed_dict={x_: sub_x, y_: sub_y})

        # print progress each iteration_print_each iteration's
        if i % train_iteration_print_each == 0:
            error = loss.eval(session=sess, feed_dict={x_: sub_x, y_: sub_y})
            error = np.sqrt(error / (data_end - data_start))
            print('Iteration:', i, ' loss:', error)

    # resolve the sub array relevant for the test
    sub_x_test = data_x[train_records:records, :]
    sub_y_test = data_y[train_records:records, :]

    # test the test data
    test_error = loss.eval(session=sess, feed_dict={x_: sub_x_test, y_: sub_y_test})
    test_error = np.sqrt(test_error / (records - train_records))
    print('Test error {0} \n \n'.format(test_error))

    # input from the user to manual test
    while True:
        test_testers = int(input("Enter number of testers: "))
        test_units = int(input("Enter units: "))
        test_year = int(input("Enter year: "))

        test_profession = input("Enter profession: ")
        test_city = input("Enter city: ")
        test_school = input("Enter school: ")

        # test data
        data_test_x = np.zeros(
            shape=(1, features),
            dtype=float,
            order='F')
        data_test_x_raw = data_test_x[0]

        # 3 normal features units/year/num of testers
        data_test_x_raw[index_testers] = test_testers - norm_testers
        data_test_x_raw[index_units] = test_units - norm_units
        data_test_x_raw[index_year] = test_year - norm_year

        # take each raw features understand what his is "id"
        # and then assign value at his id position to 1
        data_test_x_raw[unique_prof_dict[test_profession]] = 1
        data_test_x_raw[unique_cities_dict[test_city]] = 1
        data_test_x_raw[unique_schools_dict[test_school]] = 1

        data_test_y = np.matmul(data_test_x, sess.run(w)) + sess.run(b)
        print('Estimated Grade {0} \n \n'.format(data_test_y))


linear_reg(df)

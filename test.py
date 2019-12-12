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
# xlsx_to_csv(data_folder_xlsx, data_folder_csv)

school_data_set = pd.read_csv(data_folder_csv, encoding='utf-8')
school_data_set.info()

'''
plt.figure() # plotting
sns.distplot(school_dataset['Final Grade Average'],bins=20,axlabel='Grades',kde=1,norm_hist=0)
plt.show()
'''

# features
categorical_vars = ['city_id', 'school_id', 'units', 'profession', 'grad_year']
continuous_vars = ['avg_final_grades', 'num_of_testers', ]

df = pd.DataFrame(school_data_set)  # dataframe for easier handling of the data.


# refactor_data_frame(df)
# print(df)

# making the dataframe easier.
def linear_reg(data_frame):
    columns_pos = {'avg_final_grades': 0,
                   'num_of_testers': 1,
                   'units': 2,
                   'grad_year': 3,
                   'profession': 4,
                   'City city_name': 5,
                   'school_name': 6}

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

    # adding numeric columns to data frame instead of nasty hebrew ones.
    unique_prof = data_frame['profession'].unique()
    unique_cities = data_frame['city_name'].unique()
    unique_schools = data_frame['school_name'].unique()

    # numbers of features records and batch size
    regular_features = 3
    features = regular_features + unique_prof.size + unique_cities.size + unique_schools.size
    records = data_frame.size
    batch_size = 1000

    # unique maps to assign unique id per feature to specific value
    unique_prof_dict = dict(
        (val, index + regular_features) for index, val in enumerate(unique_prof))
    unique_cities_dict = dict(
        (val, index + regular_features + unique_prof.size) for index, val in enumerate(unique_cities))
    unique_schools_dict = dict(
        (val, index + regular_features + unique_prof.size + unique_cities.size) for index, val in enumerate(unique_schools))

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
    random.shuffle(stack)

    for index, row in df.iterrows():
        # the random new index used to shuffle the data
        random_index = stack[index]

        # the data_y is the final grade
        data_y[random_index] = row['avg_final_grades']
        data_raw = data_x[random_index]

        # 3 normal features units/year/num of testers
        data_raw[0] = row['num_of_testers']
        data_raw[1] = row['units'] - 2
        data_raw[2] = row['grad_year'] - 2012

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
    update = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

    # init session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(0, 10000):
        # resolve a start and end position according to the barch size
        data_start = batch_size * i % records
        data_end = (batch_size + 1) * i % records

        # in case when the end position is bigger then the start
        # because the batch_size is probably not a perfect divider to records amount
        # and if it is bigger decreasing it so it will be in the range
        if data_end < data_start:
            data_end = records

        # resolve the sub array according to batch start&end position
        sub_x = data_x[data_start:data_end, :]
        sub_y = data_y[data_start:data_end, :]

        # updating the session
        sess.run(update, feed_dict={x_: sub_x, y_: sub_y})

        # print progress each 100 iteration
        if i % 100 == 0:
            print('Iteration:', i, ' W:', sess.run(w), ' b:', sess.run(b), ' loss:',
                  loss.eval(session=sess, feed_dict={x_: sub_x, y_: sub_y}))

    # x_axis = np.arange(0, 8, 0.1)
    # x_data = []
    # for i in x_axis:
    #    x_data.append(vecto(i))
    # x_data = np.array(x_data)
    # y_vals = np.matmul(x_data, sess.run(w)) + sess.run(b)
    # import matplotlib.pyplot as plt
    # plt.plot(x_axis, y_vals)
    # plt.show()


linear_reg(df)

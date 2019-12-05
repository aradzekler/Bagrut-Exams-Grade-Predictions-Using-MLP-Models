import pandas as pd
import numpy as np
import tensorflow as tf


# HELPER FUNCTIONS#


# function to convert xlsx files to csv.
# path as C:\Users\97254\Desktop\deepLearningCourseProject\schoolDB.xlsx
def xlsx_to_csv(xlsx_file_path, csv_file_path):
    read_file = pd.read_excel(xlsx_file_path)
    read_file.to_csv(csv_file_path, index=False, header=True, encoding='utf-8')
    print("SUCCESS\ncsv created in path folder")


# function to check how much values are missing per column
def percentageMissing(df):
    if isinstance(df, pd.DataFrame):
        adict = {}  # dictionary that conatins keys columns names and values percentage of missing values in the columns
        for col in df.columns:
            adict[col] = (np.count_nonzero(df[col].isnull()) * 100) / len(df[col])
        return pd.DataFrame(adict, index=['% of missing'], columns=adict.keys())
    else:
        raise TypeError("can only be used with panda dataframe")


# making the dataframe easier.
def refractor_df(df):
    # adding numeric columns to dataframe instead of nasty hebrew ones.
    unique_cities = df['City Name'].unique()
    unique_prof = df['Profession'].unique()

    unique_cities_dict = dict(
        (val, index) for index, val in enumerate(unique_cities))  # numerical values for cities. by order (308 cities)
    unique_prof_dict = dict((val, index + 500) for index, val in enumerate(unique_prof))  # values above 500 by order.

    df['city_id'] = df['City Name'].map(unique_cities_dict)  # putting new columns into the dataframe.
    df['prof_id'] = df['Profession'].map(unique_prof_dict)

    # changing column names for easier work.
    df.rename(columns={'Final Grade Average': 'avg_final_grades',
                       'Number of Testees': 'num_of_testees',
                       'Yehidut Limud': 'yehidot_l',
                       'Graduation': 'grad_year',
                       'Profession': 'profession',
                       'City Name': 'city_name',
                       'School Name': 'school_name',
                       'School ID': 'school_id'},
              inplace=True)

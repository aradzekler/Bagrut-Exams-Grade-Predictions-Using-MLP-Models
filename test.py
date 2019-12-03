import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns # data visualization
from pathlib import Path

#HELPER FUNCTIONS#


# function to convert xlsx files to csv.
# path as C:\Users\97254\Desktop\deepLearningCourseProject\schoolDB.xlsx
def xlsx_to_csv(xlsx_file_path, csv_file_path):
    read_file = pd.read_excel(xlsx_file_path)
    read_file.to_csv(csv_file_path, index = False, header=True, encoding='utf-8')
    print("SUCCESS\ncsv created in path folder")

# function to check how much values are missing per column
def percentageMissing(Dataset):
    if isinstance(Dataset,pd.DataFrame):
        adict={} # dictionary that conatins keys columns names and values percentage of missing values in the columns
        for col in Dataset.columns:
            adict[col]=(np.count_nonzero(Dataset[col].isnull())*100)/len(Dataset[col])
        return pd.DataFrame(adict,index=['% of missing'],columns=adict.keys())
    else:
        raise TypeError("can only be used with panda dataframe")

# making the dataframe easier.
def refractor_df(df):
    # adding numeric columns to dataframe instead of nasty hebrew ones.
    uniqueCities = df['City Name'].unique()
    uniqueProf = df['Profession'].unique()

    unique_cities_dict = dict((val, index) for index, val in enumerate(uniqueCities)) # numerical values for cities. by order (308 cities)
    unique_prof_dict = dict((val, index+500) for index, val in enumerate(uniqueProf)) # values above 500 by order.

    df['city_id'] = df['City Name'].map(unique_cities_dict) # putting new columns into the dataframe.
    df['prof_id'] = df['Profession'].map(unique_prof_dict)

    # changing column names for easier work.
    df.rename(columns={'Final Grade Average':'avg_final_grades',
                          'Number of Testees':'num_of_testees',
                          'Yehidut Limud':'yehidot_l',
                          'Graduation':'grad_year',
                          'Profession':'profession',
                          'City Name': 'city_name',
                          'School Name':'school_name',
                          'School ID': 'school_id'}, 
                 inplace=True)


# setting up train and test sets (4/5 to train and 1/5 to test)
def train_and_test_div(df):
    msk = np.random.rand(len(df)) < 0.8
    train_set = df[msk]
    test_set = df[~msk]
    print(len(train_set), 'train examples')
    print(len(test_set), 'test examples')

        

data_folder_xlsx = Path("C:/Users/97254/Desktop/deepLearningCourseProject/schoolDbEng.xlsx")
data_folder_csv = Path("C:/Users/97254/Desktop/deepLearningCourseProject/schoolDBcsv.csv")
#xlsx_to_csv(data_folder_xlsx, data_folder_csv)

school_dataset = pd.read_csv(data_folder_csv, encoding='utf-8')
school_dataset.info()


'''
plt.figure() # plotting
sns.distplot(school_dataset['Final Grade Average'],bins=20,axlabel='Grades',kde=1,norm_hist=0)
plt.show()
'''

#features
categorial_vars = ['city_id', 'school_id', 'yehidot_l', 'profession', 'grad_year']
continious_vars = ['avg_final_grades', 'num_of_testees',]


df = pd.DataFrame(school_dataset) #dataframe for easier handling of the data.
refractor_df(df)
print(df)




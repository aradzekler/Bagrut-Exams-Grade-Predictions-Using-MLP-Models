import numpy as np
from math import ceil, floor


# splitting our data into train and test
class ShuffleSplit():
    '''
    n_splits - Number of splits and re-shuffling to our data
    train_size - the proportion of the data to include in the train split
    test_size - the proportion of the data to include in the test split
    random_state - the random seed we will use to split the data.
    '''

    def __init__(self, n_splits=10,
                 train_size=0.9, test_size=0.1, random_state=0):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.random_state = random_state

    '''
    n_train - Train data
    n_test - Test data
    rng - Random number for a random split
    '''

    def split(self, X, y=None):
        n_train = floor(self.train_size * X.shape[0])
        n_test = ceil(self.test_size * X.shape[0])
        rng = np.random.RandomState(self.random_state)
        for _ in range(self.n_splits):
            permutation = rng.permutation(X.shape[0])  # returns a random sequence in size of num of rows.
            yield (permutation[n_test:(n_test + n_train)],
                   # yield provides a result to its caller (like return) without destroying local variables
                   permutation[:n_test])


# split function using shuffle-split using 4/5 for train and 1/5 for test.
def train_test_split(X, y, train_size=0.8, test_size=0.2,
                     random_state=0):
    tts = ShuffleSplit(train_size=train_size, test_size=test_size, random_state=0)
    train, test = next(tts.split(X))
    return X[train], X[test], y[train], y[test]


#  BASIC METHOD! NOT VERY RANDOM
# setting up train and test sets (4/5 to train and 1/5 to test)
def train_and_test_div(df):
    msk = np.random.rand(len(df)) < 0.8
    train_set = df[msk]
    test_set = df[~msk]
    print(len(train_set), 'train examples')
    print(len(test_set), 'test examples')

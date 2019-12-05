import numpy as np

'''
Scalers for scaling our features easily.
our data contains features with alot of variation in the values
we need to scale these numbers down to a range of values which will be easier to
work with.
'''


# scaling the data that is centered around 0 with deviation of 1.
class MinMaxScaler():
    def __init__(self, feature_range=(0, 1)):  # the range of our features
        self.feature_range = feature_range

    def fit(self, X):
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        self.data_range_ = self.data_max_ - self.data_min_  # max(x) - min(x)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
        return self

    def transform(self, X):  # Xi - min(x)
        return self.feature_range[0] + (X - self.data_min_) * self.scale_


# a scaler which uses the quantile range which divides the data into 3 'ranges' (the <25% precentile, 25-75 and >75%)
# will give us numbers bigger then 0-1 and will be useful to datasets with plenty of outliner values.
class RobustScaler:
    def __init__(self, quantile_range=(25, 75)):
        self.quantile_range = quantile_range

    def fit(self, X):
        self.center_ = np.median(X, axis=0)
        quantiles = np.percentile(X, self.quantile_range, axis=0)
        self.scale_ = quantiles[1] - quantiles[0]
        return self

    def transform(self, X):
        return (X - self.center_) / self.scale_


# standard scale used for when data is evenly distributed.
class StandardScaler():
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score

training_columns = ['hour', 'HeartRate_15_mean',
                    'HeartRate_15_max', 'HeartRate_15_min', 'HeartRate_15_std',
                    'HeartRate_15_range', 'ActiveEnergyBurned_15_mean',
                    'ActiveEnergyBurned_15_max', 'ActiveEnergyBurned_15_min',
                    'ActiveEnergyBurned_15_std', 'ActiveEnergyBurned_15_range',
                    'AppleStandTime_15_mean', 'AppleStandTime_15_max',
                    'AppleStandTime_15_min', 'AppleStandTime_15_range',
                    'HeartRate_15_ar1_coef', 'ActiveEnergyBurned_15_ar1_coef',
                    'HeartRate_30_mean', 'HeartRate_30_max', 'HeartRate_30_min',
                    'HeartRate_30_std', 'HeartRate_30_range', 'ActiveEnergyBurned_30_mean',
                    'ActiveEnergyBurned_30_max', 'ActiveEnergyBurned_30_min',
                    'ActiveEnergyBurned_30_std', 'ActiveEnergyBurned_30_range',
                    'AppleStandTime_30_mean', 'AppleStandTime_30_max',
                    'AppleStandTime_30_min', 'AppleStandTime_30_range',
                    'HeartRate_30_ar1_coef', 'ActiveEnergyBurned_30_ar1_coef']#

def get_user_data(data, source="cr"):
    X_source = data.query("source==@source")
    
    y = X_source["posture"]
    X = X_source.drop("posture",axis=1)
    return X,y


def split_data(X,source="cr",test_size=0.2):
    # split into source and non-source data:
    X_source = X[X["source"]==source]
    X_non_source = X[X["source"]!=source]

    # extract label y and drop from X
    y_source = X_source["posture"]
    y_non_source = X_non_source["posture"]
    X_source = X_source.drop("posture",axis=1)
    X_non_source = X_non_source.drop("posture",axis=1)

    # set source to 1 if source and 0 if non-source
    X_source["source"] = 1
    X_non_source["source"] = 0

    # split data of source into train and test
    X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(X_source, y_source, test_size=test_size, random_state=42)

    # append non-source data to train data
    X_train = pd.concat([X_source_train,X_non_source])
    y_train = pd.concat([y_source_train,y_non_source])

    return X_train, X_source_test, y_train, y_source_test

def get_n_splits(X,y, source="cr",**kwargs):
    # get source mask
    is_source = X["source"]==source
    
    X["source"] = is_source.astype(int)

    # split X and y into source and non-source data
    X_source = X.loc[is_source,:]
    X_non_source = X.loc[~is_source,:]

    y_source = y.loc[is_source]
    y_non_source = y.loc[~is_source] # never used

    # get Shuffled split of source data
    sss = StratifiedShuffleSplit(**kwargs)
    # for each split, add non-source data to train data and return train and test indices
    for train_index, test_index in sss.split(X_source, y_source):
        train_index = np.concatenate((train_index,X_non_source.index))
        yield (train_index, test_index)



def get_training_weights(X,target=0.7):
    current_ratio = X["source"].mean()
    k = (1/current_ratio)*(target/(1-target))
    weights = X["source"].apply(lambda x: k if x==1 else 1)
    return weights


# column selector to select features for training
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select columns from a dataframe."""
    def __init__(self,columns):
        self.columns = columns
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        return X[self.columns]
    

# create scoring function that considers score metric (e.g. accuracy) only of data points from source
def user_score(estimator, X, y, metric=accuracy_score):
    # get source data
    is_source_bool = X["source"]==1
    y_source = y[is_source_bool]
    # get predictions
    y_pred = estimator.predict(X.loc[is_source_bool,training_columns])
    # calculate score
    score = metric(y_source,y_pred)
    return score



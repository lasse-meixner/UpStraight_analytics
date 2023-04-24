import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, make_scorer


def split_data(X,source="cr"):
    # split into source and non-source data:
    X_source = X[X["source"]==source]
    X_non_source = X[X["source"]!=source]

    # split data into X and y
    y_source = X_source["posture"]
    y_non_source = X_non_source["posture"]
    X_source = X_source.drop("posture",axis=1)
    X_non_source = X_non_source.drop("posture",axis=1)
    # split data of source into train and test
    X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(X_source, y_source, test_size=0.2, random_state=42)

    # append non-source data to train data
    X_train = pd.concat([X_source_train,X_non_source])
    y_train = pd.concat([y_source_train,y_non_source])

    return X_train, X_source_test, y_train, y_source_test


def get_training_weights(X,source="cr",target=0.7):
    current_ratio = X["source"].value_counts(normalize=True)[source]
    k = (1/current_ratio)*(target/(1-target))
    weights = X["source"].apply(lambda x: k if x==source else 1)
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

# create scoring function from confusion matrix that penalises false positives more than false negatives
def cm_score(y_true,y_pred):
    cm = confusion_matrix(y_true,y_pred)
    return cm[0,1]*2 + cm[1,0]

cm_score = make_scorer(cm_score,greater_is_better=False)
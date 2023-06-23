# Wrapper script to automate the model building process as explored in notebooks/app_data_epa.ipynb
# selection of features, grid search parameters, model type, is all done in UpStraight_Train.py

# Example usage: python train_slouching_classifier.py --user lass

import sys
import pandas as pd
import click
import joblib

sys.path.append("../")
from UpStraight_Train import train_user_tree_cv

# TODO: add wrapper for training that reports on GridSearchCV results and hold out performance (confusion matrix)

# main -> currently using RF model
@click.command()
@click.option("--user", default="cr", help="User to train model for")
def train_slouching_classifier(user, save = True):
    data = pd.read_csv("../data/train.csv").dropna()
    data["posture"] = data["posture"].map({-1:0,0:0,1:1})
    gsCV_results = train_user_tree_cv(data, source=user, target_weight=0.7, test_size=0.2)
    best_estimator = gsCV_results.best_estimator_
    if save:
        joblib.dump(best_estimator, user+"_model.pkl")
    else:
        return best_estimator

if __name__ == "__main__":
    train_slouching_classifier()

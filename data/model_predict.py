# Wrapper script to automate prediction file building process.
# GOAL: This predicts for both a time grid as well as all time points in which the user entered data (training data).
# REQUIRES: train.csv in data directory, and model.pkl in models directory


import sys
sys.path.append("../")
import pandas as pd
import click
import joblib

from UpStraight_Data import process_user_health, build_features
from UpStraight_Train import training_columns

@click.command()
@click.option("--user", default="cr", help="User to build prediction data for")
@click.option("--sample", default=0.1, help="Option to test function on sample of data")
def build_pred_data(user, sample, save = True):
    health_p = process_user_health(pd.read_csv(f"../data/health_filtered_{user}.csv"))

    if sample:
        health_p = health_p.sample(frac=sample, random_state=42)

    # get a predictions dataframe 
    start = health_p["start"].min()
    end = health_p["end"].max()

    pred = pd.DataFrame(pd.date_range(start=start, end=end, freq="15min"), columns=["date"])

    # add hour column manually
    pred["hour"] = pred["date"].dt.hour

    # add features
    pred = build_features(pred, health_p)

    # add training data from source
    train_source = pd.read_csv("../data/train.csv", converters={"date": pd.to_datetime}).query("source==@user")
    train_source["train"] = True
    train = train_source[training_columns + ["date", "train", "state", "state_string", "posture", "posture_string"]]
    pred = pd.concat([pred, train], axis=0).reset_index(drop=True)

    # copy date & train columns to add back after prediction, then prepare for prediction
    date_col = pred["date"]
    train_cols = pred[["train", "state", "state_string", "posture", "posture_string"]]
    pred = pred[training_columns].dropna()

    # load user model
    model = joblib.load(f"../models/{user}_model.pkl")

    # predict
    pred["pred"] = model.predict(pred)
    pred["proba"] = model.predict_proba(pred)[:,1]

    # add back date column
    pred["date"] = date_col.astype("datetime64[ns]")
    # add back train columns
    pred[["train", "state", "state_string", "posture", "posture_string"]] = train_cols
    # fill NA for train
    pred["train"] = pred["train"].fillna(False)
    
    if save:
        pred.to_csv(f"../data/pred_{user}.csv", index=False)

if __name__ == "__main__":
    build_pred_data()
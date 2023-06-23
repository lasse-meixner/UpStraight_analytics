# Wrapper script to automate prediction file building process
import sys
sys.path.append("../")
import pandas as pd
import click

from UpStraight_Data import process_user_health, build_features
from UpStraight_Train import training_columns

@click.command()
@click.option("--user", default="cr", help="User to build prediction data for")
def build_pred_data(user, save = True):
    health_p = process_user_health(pd.read_csv(f"../data/health_filtered_{user}.csv"))
    # get a predictions dataframe 
    start = health_p["start"].min()
    end = health_p["end"].max()

    pred = pd.DataFrame(pd.date_range(start=start, end=end, freq="15min"), columns=["date"])
    # add hour column manually
    pred["hour"] = pred["date"].dt.hour

    # add features
    pred = build_features(pred, health_p)[training_columns]
    
    if save:
        pred.to_csv(f"../data/pred_{user}.csv", index=False)
    
    return pred



if __name__ == "__main__":
    build_pred_data()
import os

from datetime import datetime, time
import pandas as pd
from tqdm import tqdm
from apple_watch_data_package.Apple_Data import light_preprocess, fill_stand_time

state_mapping = {-1:"Unavailable",0:"Lying",1:"Sitting",2:"Standing",3:"Active"}
posture_mapping = {-1:"Unavailable",0:"Straight",1:"Slouched"}

feature_columns = ["AppleStandTime","ActiveEnergyBurned","BasalEnergyBurned","HeartRate","DistanceWalkingRunning","StepCount"]


def preprocess(data_export):
    """Auxiliary function to preprocess the appData.

    Args:
        data_export (df): df

    Returns:
        df: df
    """
    data_export["date"] = pd.to_datetime(data_export["date"])
    data_export["day_date"] = data_export["date"].dt.date
    data_export["day"] = data_export["date"].dt.day
    data_export["time"] = data_export["date"].dt.time
    data_export["state_string"] = data_export["state"].map(state_mapping)
    data_export["posture_string"] = data_export["posture"].map(posture_mapping)
    return data_export

def get_appData(path = "../data/", save = False):
    df = pd.concat([pd.read_csv(f"{path}{f}").assign(source = f.split(".")[0].split("_")[1]) for f in os.listdir("../data") if f.startswith("export_")])
    # can preprocess concatenated file since no step depends on individual data
    df = preprocess(df)
    if save:
        df.to_csv("../data/appData.csv",index=False)
    return df

def get_health_data(path = "../data/", save = False):
    # load and light preprocess each health data file separately
    health = pd.concat([
        fill_stand_time(
        light_preprocess(
            pd.read_csv(f"{path}{f}").assign(source = f.split(".")[0].split("_")[2]))) for f in os.listdir("../data") if f.startswith("health_filtered_")])
    if save:
        health.to_csv("../data/health.csv",index=False)
    return health

def build_features(appData,health):
    # drop unavailable entries
    appData = appData[appData["state"]>-1].reset_index(drop=True)
    # for each notification (entry), get all data from the  minutes before from health data:
    # - heart rate
    # - Active Energy Burned
    # - Apple Stand Time
    for i in tqdm(range(len(appData))):
        for interval in [15,30]:
            start_time = appData.loc[i,"date"] - pd.Timedelta(minutes=interval)
            end_time = appData.loc[i,"date"]
            health_subset = health[(health["start"] >= start_time) & (health["end"] <= end_time)] # NOTE: This part is expensive.
            #print(health_subset.type.value_counts())
            build_simple_interval_features(appData,health_subset,i,interval,column="HeartRate")
            build_simple_interval_features(appData,health_subset,i,interval,column="ActiveEnergyBurned")
        for interval in [60,120]:
            start_time = appData.loc[i,"date"] - pd.Timedelta(minutes=interval)
            end_time = appData.loc[i,"date"]
            health_subset = health[(health["start"] >= start_time) & (health["end"] <= end_time)] # NOTE: This part is expensive.
            build_simple_interval_features(appData,health_subset,i,interval,column="AppleStandTime")
            

    return appData

def build_simple_interval_features(appData,health_subset,i,interval,column):
    """Auxiliary function
    For each entry in the appData, get the mean, max, min, std, and range of the 'column' type of health data for the interval before the notification.

    Args:
        appData (df): appData
        health_subset (df): health_df for the interval before the notification
        i (index): index of corresponding notification entry in appData
        interval (str): time interval
        column (str): health type
    """
    # get the right column
    base = health_subset[health_subset["type"]==column]
    appData.loc[i,f"{column}_{interval}_mean"] = base["value"].mean()
    # get the max heart rate
    appData.loc[i,f"{column}_{interval}_max"] = base["value"].max()
    # get the min heart rate
    appData.loc[i,f"{column}_{interval}_min"] = base["value"].min()
    # get the standard deviation of the heart rate
    appData.loc[i,f"{column}_{interval}_std"] = base["value"].std()
    # get the range of heart rate values
    appData.loc[i,f"{column}_{interval}_range"] = base["value"].max() - base["value"].min()

import os

from datetime import datetime, time
import pandas as pd
import numpy as np
from tqdm import tqdm
from apple_watch_data_package.Apple_Data import light_preprocess, fill_stand_time



state_mapping = {-1:"Unavailable",0:"Lying",1:"Sitting",2:"Standing",3:"Active"}
posture_mapping = {-1:"Unavailable",0:"Straight",1:"Slouched"}

feature_columns = ["AppleStandTime","ActiveEnergyBurned","HeartRate","DistanceWalkingRunning","StepCount"]
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
                    'HeartRate_30_ar1_coef', 'ActiveEnergyBurned_30_ar1_coef']


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
    data_export["hour"] = data_export["date"].dt.hour
    data_export["state_string"] = data_export["state"].map(state_mapping)
    data_export["posture_string"] = data_export["posture"].map(posture_mapping)
    return data_export

def get_appData(path = "../data/", save = False):
    df = pd.concat([pd.read_csv(f"{path}{f}").assign(source = f.split(".")[0].split("_")[1]) for f in os.listdir(path) if f.startswith("export_")])
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
            pd.read_csv(f"{path}{f}").assign(source = f.split(".")[0].split("_")[2]))) for f in os.listdir(path) if f.startswith("health_filtered_")])
    if save:
        health.to_csv("../data/health.csv",index=False)
    return health

def get_training_data(path = "../data/", save = False):
    # for each user, loop over app data export, build feature from health, and combine
    X_train = pd.DataFrame()

    for f in os.listdir(path):
        if f.startswith("export_"):
            user = f.split(".")[0].split("_")[1]
            appData = preprocess(pd.read_csv(f"{path}{f}").assign(source = user))
            health = fill_stand_time(light_preprocess(pd.read_csv(f"{path}health_filtered_{user}.csv")))
            health = health[health["type"].isin(feature_columns)]
            X = build_features(appData,health)
            X_train = pd.concat([X_train,X])      
    if save:
        X_train.to_csv("../data/X_train.csv",index=False)
    return X_train

def build_features(appData,health):
    # drop unavailable entries
    appData = appData[appData["state"]>-1].reset_index(drop=True)

    # build features
    # 1: for each entry in the appData, get the mean, max, min, std, and range of the 'column' type of health data for the interval before the notification.
    # 2: for HeartRate and ActiveEnergyBurned, fit an AR(1) process to the data and get the coefficient.
    for i in tqdm(range(len(appData))):
        for interval in [15,30]:
            start_time = appData.loc[i,"date"] - pd.Timedelta(minutes=interval)
            end_time = appData.loc[i,"date"]
            health_subset = health[(health["start"] >= start_time) & (health["end"] <= end_time)] # NOTE: This part is expensive.
            build_simple_interval_features(appData,health_subset,i,interval,column="HeartRate")
            build_simple_interval_features(appData,health_subset,i,interval,column="ActiveEnergyBurned")
            build_simple_interval_features(appData,health_subset,i,interval,column="AppleStandTime")
            fit_ar1_process(appData,health_subset,i,interval,column="HeartRate")
            fit_ar1_process(appData,health_subset,i,interval,column="ActiveEnergyBurned")

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


def fit_ar1_process(appData,health_subset,i,interval,column):
    """Auxiliary function
    For each entry in the appData, fit an AR1 process to the 'column' type of health data for the interval before the notification.

    Args:
        appData (df): appData
        health_subset (df): health_df for the interval before the notification
        i (index): index of corresponding notification entry in appData
        interval (str): time interval
        column (str): health type
    """
    # get the right column
    t = health_subset[health_subset["type"]==column]
    # shift the column by 1 and fill value by first value
    t_1 = t.shift(1).fillna(method="bfill")
    # get correlation between t and t_1
    coef = t["value"].cov(t_1["value"])/t_1["value"].var()
    # add coef if not nan otherwise add 0
    appData.loc[i,f"{column}_{interval}_ar1_coef"] = coef if not np.isnan(coef) else 0



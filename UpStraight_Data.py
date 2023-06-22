import os

from datetime import datetime, time
import pandas as pd
from tqdm import tqdm
from apple_watch_data_package.Apple_Data import light_preprocess, fill_stand_time



state_mapping = {-1:"Unavailable",0:"Lying",1:"Sitting",2:"Standing",3:"Active"}
posture_mapping = {-1:"Unavailable",0:"Straight",1:"Slouched"}

feature_columns = ["AppleStandTime","ActiveEnergyBurned","HeartRate","DistanceWalkingRunning","StepCount"]


def preprocess_appData(data_export):
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
    """Auxiliary function to load, merge and preprocess the appData for separate analysis.

    Args:
        path (str, optional): path. Defaults to "../data/".
        save (bool, optional): flag, whether to save to appData.csv in local directory. Defaults to False.

    Returns:
        df: df
    """
    df = pd.concat([pd.read_csv(f"{path}{f}").assign(source = f.split(".")[0].split("_")[1]) for f in os.listdir(path) if f.startswith("export_")])
    # can preprocess concatenated file since no step depends on individual data
    df = preprocess_appData(df)
    if save:
        df.to_csv(path  + "appData.csv",index=False)
    return df

def get_health_data(path = "../data/", save = False):
    """Auxiliary function to load, merge and preprocess the health data for separate analysis.

    Args:
        path (str, optional): path. Defaults to "../data/".
        save (bool, optional): flag, whether to save to health.csv in local directory. Defaults to False.

    Returns:
        df: df
    """
    # load and light preprocess each health data file separately
    health = pd.concat([
        fill_stand_time(
        light_preprocess(
            pd.read_csv(f"{path}{f}").assign(source = f.split(".")[0].split("_")[2]))) for f in os.listdir(path) if f.startswith("health_filtered_")])
    if save:
        health.to_csv(path + "health.csv",index=False)
    return health

def build_pre_processed_data(path = "../data/", save = False):
    """Stand alone function to build training data from the appData and health data from pairs of user files.

    Args:
        path (str, optional): path. Defaults to "../data/".
        save (bool, optional): flag, whether to save to X_train.csv in local directory. Defaults to False.

    Returns:
        df: df
    """
    # for each user, loop over app data export, build feature from health, and combine
    X_train = pd.DataFrame()

    for f in os.listdir(path):
        if f.startswith("export_"):
            user = f.split(".")[0].split("_")[1]
            appData = preprocess_appData(pd.read_csv(f"{path}{f}").assign(source = user))
            health = fill_stand_time(light_preprocess(pd.read_csv(f"{path}health_filtered_{user}.csv")))
            health = health[health["type"].isin(feature_columns)]
            X = build_features(appData,health) # get appData_p with features based on that persons health data
            X_train = pd.concat([X_train,X]) # combined all users
    if save:
        X_train.to_csv(path + "pre_processed.csv",index=False)
    return X_train

def build_features(appData_p,health_p):
    """Auxiliary function to build features from appData and health data (of each user: _p) and add them as new columns to appData_p.

    Args:
        appData_p (df): df of appData for one user
        health_p (df): df of health data for one user

    Returns:
        df: appData_p with added feature columns
    """
    # drop unavailable entries
    appData_p = appData_p[appData_p["state"]>-1].reset_index(drop=True)

    # build features
    # 1: for each entry in the appData, get the mean, max, min, std, and range of the 'column' type of health data for the interval before the notification.
    # 2: for HeartRate and ActiveEnergyBurned, fit an AR(1) process to the data and get the coefficient.
    for i in tqdm(range(len(appData_p))):
        for interval in [15,30]:
            start_time = appData_p.loc[i,"date"] - pd.Timedelta(minutes=interval)
            end_time = appData_p.loc[i,"date"]
            health_subset = health_p[(health_p["start"] >= start_time) & (health_p["end"] <= end_time)] # NOTE: This part is expensive.
            build_simple_interval_features(appData_p,health_subset,i,interval,column="HeartRate")
            build_simple_interval_features(appData_p,health_subset,i,interval,column="ActiveEnergyBurned")
            build_simple_interval_features(appData_p,health_subset,i,interval,column="AppleStandTime")
            fit_ar1_process(appData_p,health_subset,i,interval,column="HeartRate")
            fit_ar1_process(appData_p,health_subset,i,interval,column="ActiveEnergyBurned")

    return appData_p

def build_simple_interval_features(appData,health_subset,i,interval,column):
    """Auxiliary function for build_features.
    i -> health_subset 
    For each entry in the appData (i), get the mean, max, min, std, and range of the 'column' type of health data for the interval before the notification (health_subset).

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
    """Auxiliary function for build_features.
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



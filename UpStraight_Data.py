import os

from datetime import datetime, time
import pandas as pd
from numpy import isnan as np_isnan
from tqdm import tqdm
from apple_watch_data_package.Apple_Data import light_preprocess, fill_stand_time



state_mapping = {-1:"Unavailable",0:"Lying",1:"Sitting",2:"Standing",3:"Active"}
posture_mapping = {-1:"Unavailable",0:"Straight",1:"Slouched"}

feature_columns = ["AppleStandTime","ActiveEnergyBurned","HeartRate"]


def preprocess_appData(data_export):
    """Auxiliary function to preprocess the appData.

    Args:
        data_export (df): df

    Returns:
        df: df
    """
    data_export["date"] = pd.to_datetime(data_export["date"]).dt.tz_localize(None)
    data_export["day_date"] = data_export["date"].dt.date
    data_export["day"] = data_export["date"].dt.day
    data_export["time"] = data_export["date"].dt.time
    data_export["hour"] = data_export["date"].dt.hour
    data_export["state_string"] = data_export["state"].map(state_mapping)
    data_export["posture_string"] = data_export["posture"].map(posture_mapping)

    # drop unavailable entries
    data_export = data_export[data_export["state"]>-1].reset_index(drop=True)
    return data_export


def load_appData(path = "../data/", save = False):
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

def process_user_health(health_p):
    pr = fill_stand_time(light_preprocess(health_p))
    return pr.loc[pr["type"].isin(feature_columns),:]


def load_health_data(path = "../data/", save = False):
    """Auxiliary function to load, merge and preprocess the health data for separate analysis.

    Args:
        path (str, optional): path. Defaults to "../data/".
        save (bool, optional): flag, whether to save to health.csv in local directory. Defaults to False.

    Returns:
        df: df
    """
    # load and light preprocess each health data file separately
    health = pd.concat([
        process_user_health(
            pd.read_csv(f"{path}{f}").assign(source = f.split(".")[0].split("_")[2])) for f in os.listdir(path) if f.startswith("health_filtered_")])
    if save:
        health.to_csv(path + "health.csv",index=False)
    return health


def build_training_data(path = "../data/", save = False):
    """Stand alone function to build training data from the appData and health data from pairs of user files.

    Args:
        path (str, optional): path. Defaults to "../data/".
        save (bool, optional): flag, whether to save to X_train.csv in local directory. Defaults to False.

    Returns:
        df: df
    """
    # for each user, loop over app data export, build feature from health, and combine
    X_processed = pd.DataFrame()

    for f in os.listdir(path):
        if f.startswith("export_"):
            user = f.split(".")[0].split("_")[1]
            appData_p = preprocess_appData(pd.read_csv(f"{path}{f}").assign(source = user))
            health_p = process_user_health(pd.read_csv(f"{path}health_filtered_{user}.csv"))

            X = build_features(appData_p,health_p) # get appData_p with features based on that persons health data
            X_processed = pd.concat([X_processed,X]) # combined all users
    if save:
        X_processed.to_csv(path + "train.csv",index=False)
    return X_processed


def build_features(appData_p,health_p):
    """Auxiliary function to build features using timestamps of appData and variables from health data (of each user: _p) and add them as new columns to appData_p.

    Args:
        appData_p (df): df of appData for one user
        health_p (df): df of health data for one user

    Returns:
        df: appData_p with added feature columns
    """
    # set index for health data
    health_p_i = health_p.set_index(["start", "end"])
    # build features
    # 1: for each entry in the appData, get the mean, max, min, std, and range of the 'column' type of health data for the interval before the notification.
    # 2: for HeartRate and ActiveEnergyBurned, fit an AR(1) process to the data and get the coefficient.
    for i in tqdm(range(len(appData_p))):
        for interval in [15,30]:
            start_time = appData_p.loc[i,"date"] - pd.Timedelta(minutes=interval)
            end_time = appData_p.loc[i,"date"]
            mask = (health_p_i.index.get_level_values("start") >= start_time) & (health_p_i.index.get_level_values("end") <= end_time)
            health_subset_p = health_p_i.loc[mask] # NOTE: This part is expensive.
            if health_subset_p.shape[0] == 0:
                continue
            build_simple_interval_features(appData_p,health_subset_p,i,interval,column="HeartRate")
            build_simple_interval_features(appData_p,health_subset_p,i,interval,column="ActiveEnergyBurned")
            build_simple_interval_features(appData_p,health_subset_p,i,interval,column="AppleStandTime")
            build_ar1_features(appData_p,health_subset_p,i,interval,column="HeartRate")
            build_ar1_features(appData_p,health_subset_p,i,interval,column="ActiveEnergyBurned")

    return appData_p


def build_simple_interval_features(appData_p,health_subset_p,i,interval,column):
    """Auxiliary function for build_features.
    i -> health_subset 
    For each entry in the appData (i) and interval, find the corresponding health subset, and compute the mean, max, min, std, and range of the 'column' type.

    Args:
        appData (df): appData
        health_subset (df): health_df for the interval before the notification
        i (index): index of corresponding notification entry in appData
        interval (str): time interval
        column (str): health type
    """
    # get the right column
    base = health_subset_p[health_subset_p["type"]==column]
    # get list of column names as {column}_{interval}_{statistic}
    feature_names = [f"{column}_{interval}_{statistic}" for statistic in ["mean","max","min","std","range"]]
    # get the statistics
    row = fit_simple_interval_features(base["value"])
    # append to appData_p
    appData_p.loc[i,feature_names] = row


def fit_simple_interval_features(data_window):
    # compute mean, max, min, std, and range of the 'value' column of the passed data_window
    # return a df row with the features
    row = [data_window.mean(), 
           data_window.max(), 
           data_window.min(), 
           data_window.std(), 
           data_window.max() - data_window.min()]
    return row


def build_ar1_features(appData,health_subset,i,interval,column):
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
    # get list of column names as {column}_{interval}_{statistic}
    feature_names = [f"{column}_{interval}_{statistic}" for statistic in ["ar1_coef"]]
    # get the statistics
    row = fit_ar1_features(t["value"])
    # append to appData_p
    appData.loc[i,feature_names] = row


def fit_ar1_features(data_window):
    # fit an AR1 process to the 'value' column of the passed data_window
    # return the coefficient
    t = data_window.fillna(method="ffill")
    t_1 = t.shift(1).fillna(method="bfill")
    coef = t.cov(t_1)/t_1.var()
    return coef if not np_isnan(coef) else 0
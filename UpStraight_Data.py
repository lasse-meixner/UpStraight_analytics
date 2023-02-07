from datetime import datetime, time
import pandas as pd
from tqdm import tqdm

state_mapping = {-1:"Unavailable",0:"Lying",1:"Sitting",2:"Standing",3:"Active"}
posture_mapping = {-1:"Unavailable",0:"Straight",1:"Slouched"}


def preprocess(data_export):
    data_export["date"] = pd.to_datetime(data_export["date"])
    data_export["day"] = data_export["date"].dt.day
    data_export["time"] = data_export["date"].dt.time
    data_export["state_string"] = data_export["state"].map(state_mapping)
    data_export["posture_string"] = data_export["posture"].map(posture_mapping)
    return data_export

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

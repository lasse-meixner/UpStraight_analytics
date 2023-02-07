import pandas as pd
import matplotlib.pyplot as plt
import random


def plot_day(appData,health,column,day):
    health_day = health[health["day"]==day]
    appData_day = appData[appData["day"]==day].reset_index(drop=True)

    ss = health_day[health_day["type"]==column]
    
    # baseline plot with health data
    f = plt.figure(figsize=(16,10))
    plt.plot_date(x=ss.start,y=ss.value)
    
    # add vertical lines for appData and annotate
    for i in range(len(appData_day)):
        color = "red" if appData_day.iloc[i].state == 1 else "black" if appData_day.iloc[i].state==-1 else "blue"
        line_style = "solid" if appData_day.iloc[i].posture == 1 else "dashdot"
        plt.axvline(x=appData_day.iloc[i].date,color=color,linestyle=line_style)
        plt.annotate(appData_day.iloc[i].state_string,xy=(appData_day.iloc[i].date,random.uniform(ss.value.min(),ss.value.max())))
        
    plt.title(column)
    plt.show()
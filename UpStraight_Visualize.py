import pandas as pd
import matplotlib.pyplot as plt
import random

import plotly.graph_objects as go


def plot_day(appData,health,column,entry):
    day = appData.groupby("day_date").count().sort_values("date",ascending=False).index[entry]
    health_day = health[health["day_date"]==day]
    appData_day = appData[appData["day_date"]==day].reset_index(drop=True)

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
        
    plt.title(column + " on " + str(day))
    plt.show()

def plot_day_plotly(appData,health,column,day):
    """same as above but using plotly library"""
    health_day = health[health["day_date"]==day]
    appData_day = appData[appData["day_date"]==day].reset_index(drop=True)

    ss = health_day[health_day["type"]==column]
    
    # baseline plot with health data
    f = go.Figure()
    f.add_trace(go.Scatter(x=ss.start,y=ss.value,mode="markers"))
    
    # add vertical lines for appData and annotate
    for i in range(len(appData_day)):
        color = "red" if appData_day.iloc[i].state == 1 else "black" if appData_day.iloc[i].state==-1 else "blue"
        line_style = "solid" if appData_day.iloc[i].posture == 1 else "dashdot"
        f.add_vline(x=appData_day.iloc[i].date,line_width=2,line_dash=line_style,line_color=color)
        f.add_annotation(x=appData_day.iloc[i].date,y=random.uniform(ss.value.min(),ss.value.max()),text=appData_day.iloc[i].state_string)
        
    f.update_layout(title=column + " on " + str(day))
    return f

def plot_day_prediction_plotly(prediction, day):
    """plot prediction"""
    # get selected day
    prediction_day = prediction[prediction["date"].dt.date==day]
    f = go.Figure()
    # add a bar for proba
    f.add_trace(go.Bar(x=prediction_day.date,y=prediction_day.proba, showlegend=False))
    f.add_trace(go.Scatter(x=prediction_day.date,y=prediction_day.proba, mode="lines", showlegend=False))
    # create green backgrounds for intervals of x where proba > 0.5
    f.add_hline(y=0.5,line_width=2,line_dash="dash",line_color="green")
    f.update_layout(title="Predicted slouching probabilities for 15 min intervals on " + str(day))
    return f
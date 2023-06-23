# Dash app for data visualization
import os

import dash
from dash.dependencies import Input, Output
from dash import html, dcc
import plotly.express as px
import pandas as pd

from UpStraight_Data import feature_columns
from UpStraight_Visualize import plot_day_plotly, plot_day_prediction_plotly




# load available prediction files in data folder as dict, with user as key, and df as value, if it exists
pred_files = {f.split(".")[0].split("_")[1]:pd.read_csv(f"data/{f}", converters = {"date": pd.to_datetime}) for f in os.listdir("data/") if f.startswith("pred_")}

# load data. 
appData = pd.read_csv("data/appData.csv", converters= {"date": pd.to_datetime, "day_date": pd.to_datetime})
health = pd.read_csv("data/health.csv", converters= {"date": pd.to_datetime, "day_date": pd.to_datetime})
health = health[health["type"].isin(feature_columns)]

def get_day(source, entry):
    return appData.query("source==@source").groupby("day_date").count().sort_values("date",ascending=False).index[entry-1]

# build app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("UpStraight"),
    html.H2("Visualize"),
    html.Div([
        html.Div([
            html.Label("Column"),
            dcc.Dropdown(
                id="column",
                options=[{"label": i, "value": i} for i in feature_columns],
                value=feature_columns[0],
            ),
        ],style={"width": "48%", "display": "inline-block"}),
        html.Div([
            html.Label("Source"),
            dcc.Dropdown(
                id="source",
                options=[{"label": i, "value": i} for i in appData["source"].unique()],
                value="cr",
            ),
        ],style={"width": "48%", "display": "inline-block"}),
        # left button for days
        html.Div([
            html.Label("Day"),
            dcc.Dropdown(
                id="day_entry",
                value=1,
            ),

        ],style={"width": "48%", "display": "inline-block"}),
    ]),
    dcc.Graph(id="graph"),
    dcc.Graph(id="graph_pred")
])

@app.callback(
    Output("graph", "figure"),
    [Input("column", "value"),
    Input("source", "value"),
    Input("day_entry", "value")])
def update_graph(column, source, day_entry):
    fig = plot_day_plotly(appData.query("source==@source"),health.query("source==@source"),column,get_day(source,day_entry))
    return fig

# callback for day dropdown depending on selected source
@app.callback(
    Output("day_entry", "options"),
    [Input("source", "value")])
def update_day_options(source):
    return [{"label": i, "value": i} for i in range(1,appData.query("source==@source").groupby("day_date").count().shape[0]+1)]

@app.callback(
    Output("graph_pred", "figure"),
    [Input("source", "value"),
     Input("day_entry", "value")])
def update_graph_pred(source, day_entry):
    fig = plot_day_prediction_plotly(pred_files[source], get_day(source,day_entry))
    return fig

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)

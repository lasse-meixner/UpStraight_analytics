# Dash app for data visualization

import dash
from dash.dependencies import Input, Output
from dash import html, dcc
import plotly.express as px
import pandas as pd

from UpStraight_Data import feature_columns
from UpStraight_Visualize import plot_day_plotly


# load data
appData = pd.read_csv("data/appData.csv")
health = pd.read_csv("data/health.csv")
health = health[health["type"].isin(feature_columns)]

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
                id="day",
                value=1,
            ),

        ],style={"width": "48%", "display": "inline-block"}),
    ]),
    dcc.Graph(id="graph"),
])

@app.callback(
    Output("graph", "figure"),
    [Input("column", "value"),
    Input("source", "value"),
    Input("day", "value")])
def update_graph(column, source, day):
    fig = plot_day_plotly(appData.query("source==@source"),health.query("source==@source"),column,day)
    return fig

# callback for day dropdown depending on selected source
@app.callback(
    Output("day", "options"),
    [Input("source", "value")])
def update_day_options(source):
    return [{"label": i, "value": i} for i in range(1,appData.query("source==@source").groupby("day_date").count().shape[0]+1)]


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)

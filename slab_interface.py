from dash import Dash, html, dcc, callback,  Output, Input
import csv
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import numpy as np
    
df = open('slab.csv')

def update_df():
    i=0
    lines = csv.reader(df, delimiter=',')
    time_array=[]
    y_arrray = []
    for row in lines:
        i+=1
        if i%2==0:
            time_array.append(row[2])
            y_arrray.append((-float(row[0]) / 10) +91)
    return time_array, y_arrray

app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP])

fig = go.Figure(data=[go.Scatter(x=[0], y=[0])])

app.layout = html.Div([
    dbc.Row(html.H1(children='Slab algorithm', style={'textAlign': 'center'})),
    dbc.Row([
        dbc.Col(dcc.Graph(id='graph_1', figure=fig)),
        dbc.Col([
            dbc.Col(dcc.Graph(id='graph_2', figure=fig)),
            dbc.Col(dcc.Graph(id='graph_3', figure=fig)),
            ])
    ]),
    dbc.Row([
        dbc.Col([
        html.Button('Update Graph', id='update-graph-button', style={'textAlign': 'center'})
        ],
        width={'size': 2}
        )
        ]),
])

@app.callback(
    Output(component_id='graph_1', component_property= 'figure'),
    Input(component_id='update-graph-button', component_property='value')
)
def update_figure(value):
    global fig
    row_time, y = update_df()
    fig = go.Figure(data=[go.Scatter(x=row_time, y=y)])
    return fig

if __name__=='__main__':
    app.run_server(debug=True)
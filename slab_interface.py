from dash import Dash, html, dcc, callback, Output, Input
import csv
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import numpy as np
import cv2
import base64
import asyncio
from datetime import date


frame = open('Images/img.png')


def update_df():
    df = open('slabdataMetr0308_13.csv')
    i = 0
    lines = csv.reader(df, delimiter=',')
    time_array = []
    y_arrray = []
    for row in lines:
        i += 1
        if i % 2 == 0:
            time_array.append(row[2])
            y_arrray.append((-float(row[0]) / (9 + 0.005 * float(row[0]))) + 93)
    return time_array, y_arrray


def update_image():
    encoded_frame = cv2.imencode('.jpg', frame)[1]
    base64_frame = base64.b64encode(encoded_frame).decode('utf-8')


app = Dash(__name__,
           external_stylesheets=[dbc.themes.DARKLY])

fig = go.Figure(data=[go.Scatter(x=[0], y=[0])])

app.layout = html.Div([
    dbc.Row([
        html.Div([
            html.H1(children='Slab algorithm', style={'float': 'left', 'font-size': '45px', 'margin-right': '30px'}),
            html.H2(children='00:00:00', style={'float': 'left', 'font-size': '45px'}),
            html.H3(children='Текущий Y (м)', style={'float': 'right', 'font-size': '45px', 'margin-left': '30px'}),
            html.H4(children='00:00:00', style={'float': 'right', 'font-size': '45px'}),
        ]),
    ]),
    dbc.Row([
        dbc.Col(html.H1(children='Videoframe', style={'textAlign': 'center'}), align='center', width={"size": 6}),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            dcc.DatePickerRange(
                                month_format='MMMM Y',
                                end_date_placeholder_text='MMMM Y',
                                start_date=date(2023, 8, 9),
                            ),

                            dbc.Button(id='iba_graph_button', children='iba_graph', color='primary',
                                       className='mx-auto',
                                       size='lg'),
                            dbc.Button(id='camera_graph_button', children='camera_graph', color='primary',
                                       className='mx-auto', size='lg')
                        ], className='modal-dialog-centered'),
                    ]),
                ]),
                html.Div(dcc.Graph(id='graph_1', figure=fig),
                         style={'width': '1050px', 'height': '370px', 'float': 'right'}),
                html.Div(dcc.Graph(id='graph_2', figure=fig),
                         style={'width': '1050px', 'height': '370px', 'float': 'right'}),
            ]),
        ],
            width={"size": 6}
        )
    ]),
    dbc.Row([
        html.Div([
            dbc.Button(id='start_button', children='Start', color='primary', className='flex-row-reverse',
                       size="lg", style={'margin': '10px', 'float': 'center'}),
            dbc.Button(id='pause_button', children='Stop', color='primary', className='flex-row-reverse',
                       size="lg", style={'margin': '10px', 'float': 'center'}),
            dbc.Button(id='save_csv', children='Save .scv file', color='primary',
                       className='flex-row-reverse', size="lg", style={'margin': '10px', 'float': 'right'}),
        ],
            style={'width': '1000px', 'height': '80px'}
        ),
    ])
],
style={'overflow': 'hidden'})


@app.callback(
    Output(component_id='graph_1', component_property='figure'),
    Input(component_id='start_button', component_property='value')
)
def update_figure_1(value):
    global fig
    row_time, y = update_df()
    fig = (go.Figure(data=[go.Scatter(x=row_time, y=y)], layout={'paper_bgcolor': '#222'}))
    return fig


@app.callback(
    Output(component_id='graph_2', component_property='figure'),
    Input(component_id='start_button', component_property='value')
)
def update_figure_2(value):
    global fig
    row_time, y = update_df()
    fig = (go.Figure(data=[go.Scatter(x=row_time, y=y)], layout={'paper_bgcolor': '#222'}))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

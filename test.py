from dash import Dash, html, dcc, callback, Output, Input
import csv
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import cv2
import base64
import asyncio
from datetime import date
import plotly.express as px
import numpy as np
import pandas as pd


def update_slabdata_csv(csv_path):
    df = open(csv_path)
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

def update_merged_csv(csv_path):
    df = pd.read_csv(csv_path)
    data_x = df.iloc[:, 0]
    data_x = pd.to_datetime(df.iloc[:, 0], format='%d.%m.%Y %H:%M:%S.%f')
    data_y = df.iloc[:, 1]
    return data_x, data_y


def update_image():
    image = cv2.imread('image.jpg')
    image = np.array(image, dtype=float)
    encoded_frame = cv2.imencode('.jpg', image)[1]
    base64_frame = base64.b64encode(encoded_frame).decode('utf-8')
    return base64_frame


app = Dash(__name__,
           external_stylesheets=[dbc.themes.DARKLY])

fig = go.Figure(layout={'paper_bgcolor': '#222'})

app.layout = html.Div([
    dbc.Row([
        html.Div([
            html.H1(children='Slab algorithm', style={'float': 'left', 'font-size': '45px', 'margin-right': '30px'}),
            html.H2(children='00:00:00',  id='real_time', style={'float': 'left', 'font-size': '45px'}),
            html.H3(children='Текущий Y (м)', id='real_y', style={'float': 'right', 'font-size': '45px', 'margin-left': '30px'}),
            html.H4(children='00:00:00', id='current_time', style={'float': 'right', 'font-size': '45px'}),
        ]),
    ]),
    dbc.Row([
        dbc.Col(html.Img(src='data:image/jpg;base64,{}'.format(update_image()), style={'width': '800px', 'margin': '10px', 'margin-top': '150px', })),
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
    Output(component_id='graph_2', component_property='figure'),
    Input(component_id='start_button', component_property='value')
)
def update_merged_plot(value):
    '''
    fig2 = make_subplots()
    fig1.add_trace(
        go.Scatter(x=data_x, y=data_y, name="iba_file"),
    '''
    row_time, y = update_merged_csv('merged.csv')
    fig_1 = go.Figure(layout={'paper_bgcolor': '#222'})
    fig_1.add_trace(go.Scatter(x=row_time, y=y))
    row_time, y = update_merged_csv('slabdataMetr0308_13.csv')
    fig_2 = go.Figure(layout={'paper_bgcolor': '#222'})
    fig_1.add_trace(go.Scatter(x=row_time, y=y))
    return fig_1, fig_2


if __name__ == '__main__':
    app.run_server(debug=True)

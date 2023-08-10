from dash import Dash
from dash import html, dcc, Output, Input
import csv
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import cv2
import base64
import asyncio
import numpy as np
import pandas as pd
import datetime as dt
from plotly.subplots import make_subplots
from statistics import mean
from collections import deque

g_kernel_h = cv2.getGaborKernel((3, 5), 5.0, np.pi, 5.0, 0.2, 0, ktype=cv2.CV_32F)
h, w = g_kernel_h.shape[:2]
g_kernel_h = cv2.resize(g_kernel_h, (8 * w, 8 * h), interpolation=cv2.INTER_CUBIC)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
t = 0
time = deque([0], 1)
ylist = deque([0], 2)
velocity = deque([0], 1)
ymax = deque([0, 0], 19)
yymax = 0


class Graphs():
    iba_graph_on = False
    slabdata_graph_on = False


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
            y_arrray.append(row[0])
    return time_array, y_arrray


def update_merged_csv(csv_path):
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    data_x = pd.to_datetime(df.iloc[:, 0], format='%d.%m.%Y %H:%M:%S.%f')
    data_y = df.iloc[:, 1]
    return data_x, data_y


def update_image():
    frame = cv2.imread('MicrosoftTeams-image (6).png')
    image = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 20

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame1 = cv2.bitwise_or(gray_frame, image)

    frame1 = cv2.GaussianBlur(frame1, (7, 7), 11)
    thresh = cv2.inRange(frame1, 145, 160)

    thresh = cv2.filter2D(thresh, cv2.CV_8UC3, g_kernel_h)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = thresh[:, 230:370]
    frame1 = frame1[:, 230:370]
    frame = frame[:, 230:370]

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
        rect = cv2.minAreaRect(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        box = cv2.boxPoints(rect)
        center = rect[0]
        area = cv2.contourArea(cnt)
        box = np.intp(box)
        if (len(approx) > 5) & (area > 100):
            cv2.circle(frame, (int(center[0]), int(center[1])), 2, (255, 0, 255), 2)
            cv2.drawContours(frame, [box], 0, (255, 255, 0), 3)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (int(x + w / 2), int(y + h - 20)), 2, (0, 0, 255), 5)
            yy = int(y + h - 20)
            ymax.append(yy)
        yymax = mean(ymax)

        encoded_frame = cv2.imencode('.jpg', gray_frame)[1]
        base64_frame = base64.b64encode(encoded_frame).decode('utf-8')
        return base64_frame


app = Dash(__name__,
           external_stylesheets=[dbc.themes.DARKLY])

fig = go.Figure()  # layout={'paper_bgcolor': '#222'}

app.layout = html.Div([
    dbc.Row([
        html.Div([
            html.H1(children='Slab algorithm', style={'float': 'left', 'font-size': '15px', 'margin-right': '30px'}),
            html.H2(children=dt.datetime.now().strftime("%H:%M:%S"), id='real_time',
                    style={'float': 'left', 'font-size': '15px'}),
            html.H3(children='Текущий Y (м)', id='real_y',
                    style={'float': 'right', 'font-size': '15px', 'margin-left': '30px'}),
            html.H4(children='00:00:00', id='current_time', style={'float': 'right', 'font-size': '15px'}),
        ]),
    ]),
    dbc.Row([
        dbc.Col([html.Img(src='data:image/jpg;base64,{}'.format(update_image()),
                          style={'width': '920px', 'height': '800px', 'margin': '10px', 'margin-top': '50px', }),
                 html.Div([
                     dbc.Button(id='start_button', children='Start', color='primary', className='flex-row-reverse',
                                size="lg", style={'margin-right': '10px', 'margin-left': '10px', 'float': 'top'}),
                     dbc.Button(id='pause_button', children='Stop', color='primary', className='flex-row-reverse',
                                size="lg", style={'margin-right': '10px'}),
                     dbc.Button(id='save_csv', children='Save .scv file', color='primary',
                                size="lg",
                                style={'float': 'right', 'margin-right': '15px'}),
                 ]),
                 ]),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            dcc.DatePickerRange(
                                month_format='MMMM Y',
                                end_date_placeholder_text='MMMM Y',
                                start_date=dt.date(2023, 8, 9),
                                style={'margin-right': '40px', 'float': 'left'}
                            ),

                            dbc.Button(id='iba_graph_button', children='iba_graph', color='primary', size='lg',
                                       style={'float': 'right'}),
                            dbc.Button(id='slabdata_graph_button', children='camera_graph', color='primary', size='lg',
                                       style={'float': 'right', 'margin-right': '20px'})
                        ]),
                    ]),
                ]),
                html.Div(dcc.Graph(id='iba_graph', figure=fig),
                         style={'width': '1000px', 'height': '280px', 'float': 'center', 'margin-bottom': '150px'}),
                html.Div(dcc.Graph(id='slabdata_graph', figure=fig),
                         style={'width': '1000px', 'height': '347px', 'float': 'center'}),
            ]),
        ],
            width={"size": 6}
        )
    ])
])


@app.callback(
    Output(component_id='iba_graph', component_property='figure',  allow_duplicate=True),
    Input(component_id='iba_graph_button', component_property='value'),
    prevent_initial_call='initial_duplicate'
)
def update_iba_graph(value):
    #row_time, y = update_merged_csv('merged.csv')
    row_time, y = update_slabdata_csv('slabdataMetr0308_21.csv')
    data_x, data_y = update_slabdata_csv('slabdataMetr0308_21.csv')
    fig_1 = make_subplots()
    if not Graphs.iba_graph_on and not Graphs.slabdata_graph_on:
        fig_1.add_trace(go.Scatter(x=row_time, y=y, name="iba_file"))
        Graphs.iba_graph_on = True
    elif Graphs.iba_graph_on and Graphs.slabdata_graph_on:
        fig_1.add_trace(go.Scatter(x=data_x, y=data_y, name="slabdata"))
        Graphs.iba_graph_on = False
    else:
        fig_1.add_trace(go.Scatter(x=row_time, y=y, name="iba_file"))
        fig_1.add_trace(go.Scatter(x=data_x, y=data_y, name="slabdata"))

    fig_1.update_xaxes(title_text="Time")
    fig_1.update_yaxes(title_text="Distance")

    return fig_1

@app.callback(
    Output(component_id='iba_graph', component_property='figure', allow_duplicate=True),
    Input(component_id='slabdata_graph_button', component_property='value'),
    prevent_initial_call='initial_duplicate'
)
def update_slab_graph(value):
    #row_time, y = update_merged_csv('merged.csv')
    row_time, y = update_slabdata_csv('slabdataMetr0308_21.csv')
    data_x, data_y = update_slabdata_csv('slabdataMetr0308_21.csv')
    fig_1 = make_subplots()
    if not Graphs.slabdata_graph_on and Graphs.iba_graph_on:
        fig_1.add_trace(go.Scatter(x=data_x, y=data_y, name="slabdata"))
        Graphs.slabdata_graph_on = True
    elif not Graphs.slabdata_graph_on and not Graphs.iba_graph_on:
        fig_1.add_trace(go.Scatter(x=row_time, y=y, name="iba_file"))
        Graphs.slabdata_graph_on = False
    else:
        fig_1.add_trace(go.Scatter(x=row_time, y=y, name="iba_file"))
        fig_1.add_trace(go.Scatter(x=data_x, y=data_y, name="slabdata"))

    fig_1.update_xaxes(title_text="Time")
    fig_1.update_yaxes(title_text="Distance")

    return fig_1

@app.callback(
    Output(component_id='slabdata_graph', component_property='figure'),
    Input(component_id='start_button', component_property='value')
)
def update_slabdata_plot(value):
    fig_2 = make_subplots()
    data_x, data_y = update_slabdata_csv('slabdataMetr0308_21.csv')
    fig_2.add_trace(go.Scatter(x=data_x, y=data_y, name="real_slabdata"))
    fig_2.update_xaxes(title_text="Time")
    fig_2.update_yaxes(title_text="Distance")

    return fig_2


if __name__ == '__main__':
    app.run_server(debug=True)

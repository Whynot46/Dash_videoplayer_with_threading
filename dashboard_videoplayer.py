from dash import Dash, html, dcc, Output, Input
import dash_bootstrap_components as dbc
import cv2
import base64
import numpy as np
import threading


class Video:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ret = None
        self.urlRtsp = "VIDEO_PATH"
        self.frame = np.zeros((512, 512, 3), dtype="uint8")
        self.capture = cv2.VideoCapture(self.urlRtsp)
        self.processed_frame = None
        self.output_frame = None
        self.th_processing_frame = threading.Thread(target=self.update_frame, daemon=True)
        self.th_read_frame = threading.Thread(target=self.read_capture, daemon=True)
        self.th_read_frame.start()
        self.th_processing_frame.start()

    def update_frame(self):
        while True:
            self.video_processing()
            encoded_frame = cv2.imencode('.jpg', self.processed_frame.copy())[1]
            self.output_frame = base64.b64encode(encoded_frame).decode('utf-8')

    def read_capture(self):
        while True:
            i_errors_read = 0
            try:
                self.ret, self.frame = self.capture.read()
                if self.ret == False:
                    i_errors_read += 1
                    if i_errors_read >= 10:
                        self.capture = cv2.VideoCapture(self.urlRtsp)
            except:
                self.frame = cv2.VideoCapture(self.urlRtsp)

    def frame_processing(self):
        return self.frame

    def video_processing(self):
        self.processed_frame = self.frame_processing()


class Dashboard:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        self.video_player = Video()
        self.app.layout = html.Div([
                    html.Img(style={'width': '920px', 'height': '800px', 'margin': '10px', 'margin-top': '50px'}, id='video_player'),
                    dcc.Interval(
                        id='videoframe_fps',
                        interval=1000/60,
                        n_intervals=0),
        ])

        @self.app.callback(
            [Output(component_id='video_player', component_property='src')],
            [Input('videoframe_fps', 'n_intervals')]
        )
        def update_frame(interval):
            self.output_frame = 'data:image/jpg;base64,{}'.format(self.video_player.output_frame)
            return [self.output_frame]

if __name__ == '__main__':
    Dashboard().app.run_server(debug=True)

# candle_chart.py

import os
import shutil
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.plotting import show
from bokeh.plotting import output_notebook
from bokeh.layouts import column, row, layout, gridplot
from bokeh.models import Spacer, Span, Text, Range1d, LinearAxis, DataRange1d
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.io import export_png
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

output_notebook()

TOOLS = "pan,wheel_zoom,box_zoom,box_select,crosshair,reset,save"
TOOLTIPS=[  ( 'date',   '@date' ),
            ( 'close',  '@y' ), 
        ]

class TimeChart():
    Y_AXIS_2ND = 'axis2nd'
    
    
    def __init__(self, title, width, height, ylabel='', date_format='%Y/%m/%d %H:%M'):
        if width is None:        
            self.fig = figure(  x_axis_type="linear",
                                tools=TOOLS, 
                                sizing_mode='stretch_width',
                                outer_height=height,
                                y_axis_label=ylabel,
                                title = title)
        else:
            self.fig = figure(  x_axis_type="linear",
                    tools=TOOLS, 
                    outer_width=width,
                    outer_height=height,
                    #plot_width=width,
                    #plot_height=height,
                    y_axis_label=ylabel,
                    title = title)
        self.width = width
        self.height = height
 
    def add_axis(self, ylabel='', yrange=None):
        if yrange is None:   
            self.fig.extra_y_ranges = {self.Y_AXIS_2ND: DataRange1d()}
        else:
            self.fig.extra_y_ranges = {self.Y_AXIS_2ND: DataRange1d(start=yrange[0], end=yrange[1])}
        axis = LinearAxis(y_range_name=self.Y_AXIS_2ND, axis_label=ylabel)
        self.fig.add_layout(axis,"right")
        
    def set_ylim(self, min, max, yrange):
        r = max - min
        if r > yrange:
            upper = max
            lower = max - yrange
        else:
            center = np.mean([min, max])
            upper = center + yrange / 2
            lower = center - yrange / 2
        self.fig.y_range =  Range1d(lower, upper)

    def line(self, time, y, extra_axis=False, **kwargs):
        if extra_axis:
            self.fig.line(time, np.array(y), y_range_name=self.Y_AXIS_2ND, **kwargs)
        else:
            self.fig.line(time, np.array(y), **kwargs)
        
    def scatter(self, ts, ys, **kwargs):
        indices = [self.time_index(t) for t in ts]
        self.fig.scatter(indices, np.array(ys), **kwargs)
    
        
    def markers(self, signal, times, values, status, marker='o', color='black', alpha=1.0, size=10):
        marks = {'o': 'circle', 'v': 'inverted_triangle', '^': 'triangle', '+': 'cross', 'x': 'x', '*': 'star'}
        indices = []
        ys = []
        for i, (s, v) in enumerate(zip(signal, values)):
            if s == status:
                indices.append(i)
                ys.append(v)
        self.fig.scatter(times, np.array(ys), marker=marks[marker], color=color, alpha=alpha, size=size)
        
    def marker(self, time, value, marker='o', color='black', alpha=1.0, size=10):
        marks = {'o': 'circle', 'v': 'inverted_triangle', '^': 'triangle', '+': 'cross', 'x': 'x', '*': 'star'}
        self.fig.scatter([time], np.array([value]), marker=marks[marker], color=color, alpha=alpha, size=size)    
        
    def vline(self, time, color, width=None):
        span = Span(location=time,
                    dimension='height',
                    line_color=color,
                    line_alpha=0.1,
                    line_width=width)
        self.fig.add_layout(span)
        
    def hline(self, value, color, extra_axis=False, width=1):
        array = np.full(len(self.indices), value)
        self.line(array, extra_axis=extra_axis, line_width=width, color=color)
        
            
    def text(self, time, y, text, color):
        glyph = Text(x="x", y="y", text="text",  text_color=color, text_font_size='9pt')
        source = ColumnDataSource(dict(x=[self.time_index(time)], y=[y], text=[text]))
        self.fig.add_glyph(source, glyph)
        
    def plot_background(self, array, colors):
        for i , a in enumerate(array):
            if a > 0:
                self.vline(i, colors[0])
            elif a < 0:
                self.vline(i, colors[1])    
                
    def to_png(self, filepath):
        export_png(self.fig, filename=filepath)
        
class TimeSeriesChart(TimeChart):
    Y_AXIS_2ND = 'axis2nd'
    
    
    def __init__(self, title, width, height, time, ylabel='', date_format='%Y/%m/%d %H:%M'):
        if width is None:        
            self.fig = figure(  x_axis_type="linear",
                                tools=TOOLS, 
                                sizing_mode='stretch_width',
                                outer_height=height,
                                y_axis_label=ylabel,
                                title = title)
        else:
            self.fig = figure(  x_axis_type="linear",
                    tools=TOOLS, 
                    outer_width=width,
                    outer_height=height,
                    #plot_width=width,
                    #plot_height=height,
                    y_axis_label=ylabel,
                    title = title)
        self.width = width
        self.height = height
        self.time = time
        disp_time = time.copy()
        disp_time = list(disp_time)
        if time is None:
            self.indices = []
            return
        if len(time) < 2:
            self.indices = []
            return
        dt = time[-1] - time[-2]
        for i in range(1, 100):
            disp_time.append(time[-1] + i * dt)
        
        self.fig.xaxis.major_label_overrides = {i: d.strftime(date_format) for i, d in enumerate(disp_time)}
        self.indices =  list(range(len(time)))
        
    def time_index(self, time):
        if isinstance(time, int):
             return time
        for i, d in enumerate(self.time):
            if d > time:
                return i
        return -1      
 
    def add_axis(self, ylabel='', yrange=None):
        if yrange is None:   
            self.fig.extra_y_ranges = {self.Y_AXIS_2ND: DataRange1d()}
        else:
            self.fig.extra_y_ranges = {self.Y_AXIS_2ND: DataRange1d(start=yrange[0], end=yrange[1])}
        axis = LinearAxis(y_range_name=self.Y_AXIS_2ND, axis_label=ylabel)
        self.fig.add_layout(axis,"right")
        
    def set_ylim(self, min, max, yrange):
        r = max - min
        if r > yrange:
            upper = max
            lower = max - yrange
        else:
            center = np.mean([min, max])
            upper = center + yrange / 2
            lower = center - yrange / 2
        self.fig.y_range =  Range1d(lower, upper)

    def line(self, y, extra_axis=False, **kwargs):
        if extra_axis:
            self.fig.line(self.indices, np.array(y), y_range_name=self.Y_AXIS_2ND, **kwargs)
        else:
            self.fig.line(self.indices, np.array(y), **kwargs)
        
    def scatter(self, ts, ys, **kwargs):
        indices = [self.time_index(t) for t in ts]
        self.fig.scatter(indices, np.array(ys), **kwargs)
    
        
    def markers(self, signal, values, status, marker='o', color='black', alpha=1.0, size=10):
        marks = {'o': 'circle', 'v': 'inverted_triangle', '^': 'triangle', '+': 'cross', 'x': 'x', '*': 'star'}
        indices = []
        ys = []
        for i, (s, v) in enumerate(zip(signal, values)):
            if s == status:
                indices.append(i)
                ys.append(v)
        self.fig.scatter(indices, np.array(ys), marker=marks[marker], color=color, alpha=alpha, size=size)
        
    def marker(self, time, value, marker='o', color='black', alpha=1.0, size=10):
        marks = {'o': 'circle', 'v': 'inverted_triangle', '^': 'triangle', '+': 'cross', 'x': 'x', '*': 'star'}
        index = self.time_index(time)
        if index >= 0 and index <= self.indices[-1]:
            self.fig.scatter([index], np.array([value]), marker=marks[marker], color=color, alpha=alpha, size=size)    
        
    def vline(self, index, color, width=None):
        if not isinstance(index, int):
            index = self.time_index(index)
        if width is None:
            width =1000  / len(self.indices) 
        span = Span(location=index,
                    dimension='height',
                    line_color=color,
                    line_alpha=0.1,
                    line_width=width)
        self.fig.add_layout(span)
        
    def hline(self, value, color, extra_axis=False, width=1):
        array = np.full(len(self.indices), value)
        self.line(array, extra_axis=extra_axis, line_width=width, color=color)
        
            
    def text(self, time, y, text, color):
        glyph = Text(x="x", y="y", text="text",  text_color=color, text_font_size='9pt')
        source = ColumnDataSource(dict(x=[self.time_index(time)], y=[y], text=[text]))
        self.fig.add_glyph(source, glyph)
        
    def plot_background(self, array, colors):
        for i , a in enumerate(array):
            if a > 0:
                self.vline(i, colors[0])
            elif a < 0:
                self.vline(i, colors[1])    
                
    def to_png(self, filepath):
        export_png(self.fig, filename=filepath)
        
class CandleChart(TimeChart):
    def __init__(self, title, width, height, time, date_format='%Y/%m/%d %H:%M', yrange=None):
        super().__init__(title, width, height, time, date_format)
        self.date_format = date_format
        self.yrange = yrange

    def pickup(self, valid, arrays):
        out = []
        index = []
        for i, array in enumerate(arrays):
            d = []
            for j, (a, v) in enumerate(zip(array, valid)):
                if v:
                    if i == 0:
                        index.append(j)
                    d.append(a)
            out.append(np.array(d))
        return index, out
                    
    def plot_candle(self, op, hi, lo, cl):
        self.op = np.array(op)
        self.hi = np.array(hi)
        self.lo = np.array(lo)
        self.cl = np.array(cl)
        ascend = self.cl > self.op
        descend = ~ascend
        n = len(ascend)
        up = np.full(n, np.nan)
        under = np.full(n, np.nan)
        for i, [asc, o, c] in enumerate(zip(ascend, op, cl)):
            if asc:
                up[i] = c
                under[i] = o
            else:
                up[i] = o
                under[i] = c
        self.fig.segment(self.indices, up, self.indices, self.hi, color="black")
        self.fig.segment(self.indices, under, self.indices, self.lo, color="black")
        time, values = self.pickup(ascend, [self.op, self.cl])        
        r1 = self.fig.vbar(time, 0.5, values[0], values[1], fill_color="cyan", fill_alpha=0.5, line_color="gray")
        time, values = self.pickup(descend, [self.op, self.cl])        
        self.fig.vbar(time, 0.5, values[0], values[1], fill_color="red", fill_alpha=0.5, line_color="gray")
        
        hover = HoverTool(  renderers=[r1],
                            tooltips=[
                                        ("Date", "@Date1{" + self.date_format + "}"),
                                    ],
                            formatters={'@Date1': 'datetime',},
                            mode='vline', # 縦線に相当する値を表示
                            # mode='mouse',  # マウスポインタを合わせたときに表示
                        )
        self.fig.add_tools(hover)
        
        self.min = np.nanmin(self.lo)
        self.max = np.nanmax(self.hi)
        if self.yrange is not None:
            self.set_ylim(self.min, self.max, self.yrange)
        
                        

def from_pickle(file):
    import pickle
    filepath = f'../BlackSwan/data/Axiory/{file}'
    with open(filepath, 'rb') as f:
        data0 = pickle.load(f)
        return data0
    return None        
        
def test():
    from common import Columns
    from time_utils import TimeUtils
    symbol = 'NIKKEI_M30'
    data0 = from_pickle(f'{symbol}.pkl')
    jst0 = data0[Columns.JST]
    t1 = jst0[-1]
    t0 = t1 - timedelta(days=2)
    n, data = TimeUtils.slice(data0, Columns.JST, t0, t1)
    jst = np.array(data[Columns.JST])
    op = np.array(data[Columns.OPEN])
    hi = np.array(data[Columns.HIGH])
    lo = np.array(data[Columns.LOW])
    cl = np.array(data[Columns.CLOSE])
    chart = CandleChart(symbol, 1200, 500, jst)
    chart.plot_candle(op, hi, lo, cl)
    export_png(chart.fig, filename='./debug/cancle_chart.png')
    
def test1():
    from common import Columns
    from time_utils import TimeUtils
    symbol = 'NIKKEI_M30'
    data0 = from_pickle(f'{symbol}.pkl')
    jst0 = data0[Columns.JST]
    t1 = jst0[-1]
    t0 = t1 - timedelta(days=2)
    n, data = TimeUtils.slice(data0, Columns.JST, t0, t1)
    jst = np.array(data[Columns.JST])
    op = np.array(data[Columns.OPEN])
    hi = np.array(data[Columns.HIGH])
    lo = np.array(data[Columns.LOW])
    cl = np.array(data[Columns.CLOSE])
    chart = TimeChart(symbol, 1200, 500, jst)
    chart.line(cl)
    export_png(chart.fig, filename='./debug/line_chart.png')
        
def test2():
    x = [0, 1, 2]
    y1 = [0, 1, 2]
    y2 = [0, 1, 100]
    
    
    AXIS2 = "Axis2"

    fig = figure(x_axis_label="x", y_axis_label="y1")
    fig.extra_y_ranges = {AXIS2: DataRange1d()}
    fig.add_layout(
        LinearAxis(
            y_range_name=AXIS2,
            axis_label="y2",
        ),
        "right",
    )

    fig.line(x=x, y=y1, legend_label="y1", color="blue")
    fig.line(
        x=x,
        y=y2,
        legend_label="y2",
        color="red",
        y_range_name=AXIS2,
    )

    export_png(fig, filename='./debug1.png')


if __name__ == '__main__':
    test1()

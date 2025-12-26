# backtest_cypress.py

import os
import glob
import pandas as pd
import numpy as np
from random import randint, random
import time
import itertools
from decimal import Decimal, ROUND_FLOOR
from datetime import datetime, timedelta
from bokeh.layouts import column, row
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from dateutil import tz
from candle_chart import CandleChart, TimeChart, fig2png

from mt5_api import server_time_to_utc


JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from common import Columns, Indicators
from html_writer import HtmlWriter

from trade_bot import load_params
from montblanc import Montblanc, MontblancParam


def makeFig(rows, cols, size):
    fig, ax = plt.subplots(rows, cols, figsize=(size[0], size[1]))
    return (fig, ax)

def gridFig(row_rate, size):
    rows = sum(row_rate)
    fig = plt.figure(figsize=size)
    gs = GridSpec(rows, 1, hspace=0.6)
    axes = []
    begin = 0
    for rate in row_rate:
        end = begin + rate
        ax = plt.subplot(gs[begin: end, 0])
        axes.append(ax)
        begin = end
    return (fig, axes)

def read_data(path):
    df = pd.read_csv(path)
    df['jst'] = pd.to_datetime(df['jst'], format='ISO8601')
    return df

def slice(df, t0, t1):
    df1 = df[df['jst'] > t0]
    df2 = df1[df1['jst'] < t1]
    return df2

def tick_to_candle(df_tick: pd.DataFrame, term_sec=10) -> pd.DataFrame:
    df_tick['jst'] = pd.to_datetime(df_tick['jst'])
    df_tick = df_tick.set_index('jst')

    # 全10秒刻みのタイムスタンプを生成
    sec = f'{term_sec}S'
    start = df_tick.index.min().floor(sec)
    end = df_tick.index.max().ceil(sec)
    full_range = pd.date_range(start=start, end=end, freq=sec)

    # Tickデータを10秒グループに分けて自前でOHLC集計
    bars = []
    for t0 in full_range[:-1]:
        t1 = t0 + pd.Timedelta(seconds=term_sec)
        chunk = df_tick[(df_tick.index >= t0) & (df_tick.index < t1)]

        if not chunk.empty:
            o = chunk['bid'].iloc[0]
            h = chunk['bid'].max()
            l = chunk['bid'].min()
            c = chunk['bid'].iloc[-1]
            v = len(chunk)
        else:
            o = h = l = c = np.nan  # あとで補完
            v = 0

        bars.append([t0, o, h, l, c, v])
    df_bar = pd.DataFrame(bars, columns=['jst', 'open', 'high', 'low', 'close', 'volume'])

    # 欠損したOHLC（NaN）を前のcloseで補完（volume=0のまま）
    for col in ['open', 'high', 'low', 'close']:
        df_bar[col] = df_bar[col].ffill()

    return df_bar


def draw_bar(ax, timestamps, signal):
    for i in range(len(timestamps) - 1):
        t0 = timestamps[i]
        t1 = timestamps[i + 1]
        if signal[i] == 1: 
            ax.axvspan(t0, t1, color='green', alpha=0.1)
        elif signal[i] == -1: 
            ax.axvspan(t0, t1, color='red', alpha=0.1)


def round_number(x: float, step: float) -> float:
    """xをstep刻みの“下方向”キリ番に揃える"""
    dx = Decimal(str(x))
    ds = Decimal(str(step))
    q  = (dx / ds).to_integral_value(rounding=ROUND_FLOOR)
    return float(q * ds)

def rand_step(begin, end, step):
    l = end - begin
    n = int(l / step + 0.5) + 1
    while True:
        r = randint(0, n)
        v = begin + r * step
        if v <= end:
            return v
        
def rand_select(array):
    n = len(array)
    r = randint(0, n - 1)
    return array[r]

def market_data_files(symbol, timezone):
    timeframe = 'M1'
    year = 2025
    files = []
    for month in range(4, 10):
        mstr = str(month).zfill(2)
        fs = glob.glob(f"../DayTradeData/{timeframe}/{symbol}/{year}-{mstr}/*")
        for f in fs:
            _, filename = os.path.split(f)
            name, _ = os.path.splitext(filename)
            if int(name[-1]) in timezone:
                files.append(f)
    return files 

def read_data_as_dic(csv_path):
    df = read_data(csv_path)
    timestamps_np = df['jst'].tolist()
    timestamps = df["jst"].values
    op = df[Columns.OPEN].to_numpy()
    hi = df[Columns.HIGH].to_numpy()
    lo = df[Columns.LOW].to_numpy()
    cl = df[Columns.CLOSE].to_numpy()
    dic = {Columns.JST: timestamps_np, Columns.OPEN: op, Columns.HIGH: hi, Columns.LOW: lo, Columns.CLOSE: cl}
    return dic

def load_axiory_data(symbol):
    def read_axiory_data(path):
        df = pd.read_csv(path, delimiter='\t')
        date = df['<DATE>'].to_list() # server time
        time = df['<TIME>'].to_list()
        op = df['<OPEN>'].to_list()
        hi = df['<HIGH>'].to_list()
        lo = df['<LOW>'].to_list()
        cl = df['<CLOSE>'].to_list()
        volume = df['<VOL>'].to_list()
        jst = []
        utc = []
        for d, t in zip(date, time):
            st = datetime.strptime(f'{d} {t}', '%Y.%m.%d %H:%M:%S')
            u = server_time_to_utc(st)
            u = u.replace(tzinfo=UTC)
            utc.append(u)
            j = u.astimezone(JST)
            jst.append(j)
        return time, utc, jst, op, hi, lo, cl, volume
    
    dirpath = f'../Axiory/{symbol}/*.csv'
    files = fs = glob.glob(dirpath)
    time = []
    utc = []
    jst = []
    op = []
    hi = []
    lo = []
    cl = []
    volume = []
    for file in files:
        t, u, j, o, h, l, c, v = read_axiory_data(file)
        time += t
        utc += u
        jst += j
        op += o
        hi += h
        lo += l
        cl += c
        volume += v
    dic = {'time': time, 'utc': utc, 'jst': jst, 'open': op, 'high': hi, 'low': lo, 'close': cl, 'volume': volume}
    return dic

def load_df(symbol):
    import pickle
    path = f'../Axiory/{symbol}/{symbol}_m1_df.pkl'
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            df = pickle.load(f)
        return df
    else:
        dic = load_axiory_data(symbol)
        df = pd.DataFrame(dic)
        with open(path, mode='wb') as f2:
            pickle.dump(df, f2)
        return df

def load_params(strategy, symbol, ver):
    def array_str2int(s):
        i = s.find('[')
        j = s.find(']')
        v = s[i + 1: j]
        return float(v)
    
    print( os.getcwd())
    path = f'./{strategy}/v{ver}/{strategy}_v{ver}_best_trade_params.xlsx'
    df = pd.read_excel(path)
    df = df[df['symbol'] == symbol]
    params = []
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        if strategy == 'MaronPie':
            param = MaronPieParam.load_from_dic(row)
            param.volume = volume
            param.position_max = position_max
            params.append(param)      
        elif strategy == 'Montblanc':
            param = MontblancParam.load_from_dic(row)
            params.append(param)  
        else:
            raise Exception('No definition ' + strategy)
    return params
        
def generate_param(symbol:str, param: MontblancParam):
    param.position_max = 10
    param.sl_mode = rand_select(['fix', 'atr'])
    param.reversal_mode = rand_select(['', 'slope', 'reversal_major', 'reversal_minor'])
    param.ema_term_entry = rand_step(10, 60, 5)
    param.filter_term_exit = rand_step(10, 120, 10)
    param.atr_term = rand_step(5, 50, 5)
    param.trend_minutes = rand_select([4, 5, 7, 10, 15, 30, 45, 60])
    major_minutes = 0
    while major_minutes < param.trend_minutes:
        major_minutes = rand_select([30, 45, 60, 90, 120, 180, 240])
    param.trend_major_minutes = major_minutes
    param.trend_major_multiply = rand_step(1, 1.8, 0.1)
    param.trend_multiply = rand_step(1, 1.8, 0.1)  
    if symbol in ['JP225', 'US30']:
        param.sl_value = rand_step(40, 100, 20)
    elif symbol in ['US100', 'GER40']:
        param.sl_value = rand_step(20, 50, 10)
    elif symbol in ['SP']:
        param.sl_value = rand_step(20, 50, 10)   
    elif symbol in ['XAUUSD']:
        param.sl_value = rand_step(1, 5, 1)        
    elif symbol in ['USDJPY']:
        param.sl_value = rand_step(0.05, 0.5, 0.05)
    elif symbol in ['USOIL', 'XAGUSD']:
        param.sl_value = rand_step(0.1, 1, 0.1)
    elif symbol in ['UK100']:
        param.sl_value = rand_step(20, 50, 10)
    else:
        raise Exception('No sl defined', symbol)

def optimizer(symbol, df0, tbegin, tend, repeat=1000):
    t = tbegin - timedelta(days=10)
    df =  df0[(df0['jst'] >= t) & (df0['jst'] <= tend)]
    #for short_term, long_term, th, sl, tp in itertools.product(short_terms, long_terms, ths, sls, tps):
    param = MontblancParam()
    
    result = []
    for i in range(repeat):
        generate_param(symbol, param)
        maron = Montblanc(symbol, param)
        maron.calc(df)
        df_metric, _ = maron.simulate_doten(tbegin, tend)
        metric = performance(df_metric)
        d1 = param.to_dict()
        d0 = {'i': i}
        d = dict(**d0, **d1)
        dic = dict(**d, **metric)
        result.append(dic)
        print('Montblanc Optimize Phase1', i, symbol, 'profit:', metric['profit'])           
    keys = list(result[0].keys())
    dic = {}
    for key in ['symbol'] + keys:
        array = []
        for d in result:
            if key == 'symbol':
                array.append(symbol)
            else:
                array.append(d[key])
        dic[key] = array
    df_result = pd.DataFrame(dic)
    df_result = df_result.sort_values('profit', ascending=False)
    return df_result
    
    
    
def evaluate0(symbol, ver, params, dir_path):
    df = load_df(symbol)
    jst = dic['jst']
    hours = [[0, 6], [8, 8], [16, 8]]
    tbegin = jst[0]
    tend = jst[-1]

    i = 0
    result = []
    for param in params:
        maron = Montblanc(symbol, param)
        rows = []
        t = tbegin
        length = 3 * 24 * 60
        while t <= tend:
            t0 = t - timedelta(days=3)
            t1 = t + timedelta(days=1)
            df1 = df[(df['jst'] >= t0) & (df['jst'] <= t1)]
            if len(df1) < length:
                t += timedelta(days=1)
                continue
            index = df1.index[-1]   
            if index - length < 0:
                t += timedelta(days=1)
                continue
            df2 = df.iloc[index - length: index + 1, :]      
            maron.calc(df2)
            for b, l in hours:
                t0 = t
                t0 = t0.replace(hour=b)
                t1 = t0 + timedelta(hours=l)
                if t0 == datetime(2021,1,2, 16).astimezone(JST):
                    pass #debug
                (r, columns), _ = maron.simulate_doten(t0, t1)
                if len(r) > 0:
                    rows += r
                    #print(symbol, t0, t1, 'Montblanc Optimize 2nd done')
                else:
                    pass
                    #print(symbol, t0, t1, 'Montblanc Optimize 2nd done ... No Trade')
            t += timedelta(days=1)
            
        if len(rows) == 0:
            continue
        df_metric = pd.DataFrame(data=rows, columns=columns)
        metric = performance(df_metric)
        d1 = param.to_dict()
        d0 = {'i': i}
        d = dict(**d0, **d1)
        dic = dict(**d, **metric)
        result.append(dic)
        print('Montblanc Phase2', i, symbol, 'profit:', metric['profit'])
        if metric['profit'] > 0:
            save_profit_graph(symbol, f"{i}_profit", df_metric, dir_path)
        i += 1
    
    keys = list(result[0].keys())
    dic = {}
    for key in ['symbol', 'version'] + keys:
        array = []
        for d in result:
            if key == 'symbol':
                array.append(symbol)
            elif key == 'version':
                array.append(ver)
            else:
                array.append(d[key])
        dic[key] = array
    df_result = pd.DataFrame(dic)
    df_result = df_result.sort_values('profit', ascending=False)
    df_result.to_excel(os.path.join(dir_path, f'{symbol}_v{ver}_best_trade_params.xlsx'), index=False)
    
    
def evaluate(symbol, ver, params, dir_path):
    df = load_df(symbol)
    jst = df['jst'].to_list()
    
    tbegin = datetime(jst[0].year, jst[0].month, jst[0].day).astimezone(JST)
    tend = datetime(jst[-1].year, jst[-1].month, jst[-1].day).astimezone(JST) - timedelta(days=1)

    length = 3 * 24 * 60

    i = 0
    result = []
    for param in params:
        maron = Montblanc(symbol, param)
        dfs = []
        t = tbegin
        while t <= tend:
            t0 = t - timedelta(days=4)
            t1 = t + timedelta(days=1)
            df1 = df[(df['jst'] >= t0) & (df['jst'] <= t1)]
            if len(df1) < length:
                t += timedelta(days=1)
                continue
            index = df1.index[-1]   
            if index - length < 0:
                t += timedelta(days=1)
                continue
            df2 = df.iloc[index - length: index + 1, :]      
            maron.calc(df2)
            df_metric, _ = maron.simulate_doten(t, t1)
            if len(df_metric) > 0:
                dfs.append(df_metric)
            t = t1            
        if len(dfs) == 0:
            continue
        df_metric = pd.concat(dfs)
        metric = performance(df_metric)
        d1 = param.to_dict()
        d0 = {'i': i}
        d = dict(**d0, **d1)
        dic = dict(**d, **metric)
        result.append(dic)
        print('Montblanc Phase2', i, symbol, 'profit:', metric['profit'])
        if metric['profit'] > 0:
            save_profit_graph(symbol, f"{i}_profit", df_metric, dir_path)
        i += 1
    
    keys = list(result[0].keys())
    dic = {}
    for key in ['symbol', 'version'] + keys:
        array = []
        for d in result:
            if key == 'symbol':
                array.append(symbol)
            elif key == 'version':
                array.append(ver)
            else:
                array.append(d[key])
        dic[key] = array
    df_result = pd.DataFrame(dic)
    df_result = df_result.sort_values('profit', ascending=False)
    df_result.to_excel(os.path.join(dir_path, f'{symbol}_v{ver}_best_trade_params.xlsx'), index=False)


def plot_prices(ax, timestamp, signals, colors, labels, graph_height):
    timestamp = list(timestamp)
    for i in range(len(signals)):
        color, width = colors[i]
        ax.plot(timestamp, signals[i], color=color, linewidth=width, alpha=0.5, label=labels[i])    
    if graph_height is not None:
        vmin = min(signals[0])
        vmax = max(signals[0])
        center = (vmax + vmin) / 2
        ax.axvline(timestamp[0], center - graph_height / 2, center + graph_height / 2, linewidth=10, color='black')
       
def plot_signal_marker(ax, timestamp, signal, values, marker=None):
    if type(timestamp) == pd.Series:
        timestamp = timestamp.to_list()
    if type(signal) == pd.Series:
        signal = signal.to_list()
    if type(values) == pd.Series:
        values = values.to_list()
    n = len(signal)
    for i in range(n):
        if signal[i] == 1:
            # Long
            if marker is None:
                mark = '^'
            else:
                mark = marker
            color = 'green'
            alpha=0.5
        elif signal[i] == -1:
            # Short
            if marker is None:
                mark = 'v'
            else:
                mark = marker
            color = 'red'
            alpha=0.5
        else:
            continue
        ax.scatter(timestamp[i], values[i], marker=mark, color=color, alpha=alpha, s=100)
    
def plot_chart(title, df0: pd.DataFrame, begin, end, param: MontblancParam, graph_height):
    
    df = df0[(df0['jst'] >= begin) & (df0['jst'] <= end)]
    if len(df) < 100:
        return None
    jst = df['jst'].to_list()
    cl = df['close'].to_list()
    colors = [('gray', 1), ('red', 3), ('green', 3), ('red', 1), ('green', 1)]
    labels = ['Close',  'major(+)', 'major(-)', 'minor(+)', 'minor(-)']
    
    w = 1200
    chart1 = CandleChart(title, w, 500, jst)
    chart1.plot_candle(df['open'].to_numpy(), df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy())
    chart1.scatter(df['upper_major'].to_numpy(), color='red', size=4.0)
    chart1.scatter(df['lower_major'], color='blue', size=4.0)
    chart1.scatter(df['upper_minor'], color='orange', size=2.0)
    chart1.scatter(df['lower_minor'], color='cyan', size=2.0)
    chart2 = TimeChart(title, w, 250, jst) 
    chart2.line(df['slope_exit'], color='blue')
    chart2.hline(0.0, 'black')
    for i, v in enumerate(df['entries']):
        t = jst[i]
        if v == 1:
            chart1.marker(t, cl[i], marker='^', color='blue', alpha=0.5)
        elif v == -1:
            chart1.marker(t, cl[i], marker='v', color='red', alpha=0.5)
    for i, v in enumerate(df['exits']):
        t = jst[i]
        if v == 1:
            chart1.marker(t, cl[i], marker='x', color='green')
            chart2.marker(t, 0, marker='x', color='green')
        elif v == -1:
            chart1.marker(t, cl[i], marker='x', color='red')
            chart2.marker(t, 0, marker='x', color='red')
    return column(chart1.fig, chart2.fig)

def performance(df_original):
    # df: simulate_scalping_pips_multi の戻り（profitは価格差）
    df = df_original.copy()
    df["pnl"] = df["profit"]  # 必要なら /pip_size でpips化
    df["cum"] = df["pnl"].cumsum()
    roll_max = df["cum"].cummax()
    dd = df["cum"] - roll_max
    out = {
        "trades": len(df),
        "win_rate": (df["pnl"] > 0).mean(),
        "profit_mean": df["pnl"].mean(),
        "profit_median": df["pnl"].median(),
        "drawdown_min": dd.min(),
        "profit": df["cum"].iloc[-1] if len(df) else 0.0,
    }
    return out

def save_result(symbol, title, df, dir_path):
    profits = df['profit'].to_numpy()
    accum = np.cumsum(profits)
    df['profit_accum'] = accum
    csv_path = os.path.join(dir_path, f'{symbol}_{title}_trade_result.csv')
    df.to_csv(csv_path, index=False)

def save_profit_graph(symbol, title, df, dir_path):
    profits = df['profit'].to_numpy()
    accum = np.cumsum(profits)
    df['profit_accum'] = accum
    fig, ax = makeFig(1, 1, (10, 5))
    time = df['entry_time']
    ax.plot(time, df['profit_accum'], color='blue')
    title = f"{title} {symbol} Scalping Profit Curve"
    ax.set_title(title)
    path = os.path.join(dir_path, f"{title}_{symbol}_profit_curve.png")
    fig.savefig(path)
    plt.close()
    
       
def main(symbol, tp, sl, graph_height):
    timeframe = 'M1'
    year = 2025
    dfs = []
    for month in range(4, 10):
        mstr = str(month).zfill(2)
        files = glob.glob(f"../DayTradeData/{timeframe}/{symbol}/{year}-{mstr}/*")
        writer = HtmlWriter()
        for file in files:
            _, filename = os.path.split(file)
            name, _ = os.path.splitext(filename)
            fig, df = backtest(f'{name}', file,  tp, sl, graph_height)
            dfs.append(df)
            writer.add_fig(fig)
        dir_path = f'./Montblanc/Scalping/{symbol}'
        os.makedirs(dir_path, exist_ok=True)
        path = f'./Montblanc/Scalping/{symbol}/{symbol}_{year}_{mstr}.html'
        writer.write(path)
        
    df = pd.concat(dfs)
    save_result(symbol, '#0', df, dir_path)

        
def optimize(symbol, ver, pass_phase1=False):
    iver = int(ver)
    print('Start', symbol, 'Ver.', ver)
    df0 = load_df(symbol)

    tbegin = datetime(2025, 2, 26).astimezone(JST)
    tend = datetime(2025, 4, 10).astimezone(JST)  

    dirpath = f'./Montblanc/v{ver}/Optimize/{symbol}'
    path = os.path.join(dirpath, f"{symbol}_v{ver}_params_phase1.xlsx")
    if pass_phase1:
        df_top = pd.read_excel(path)
        print(df_top)
    else:
        df_top = optimizer(symbol, df0, tbegin, tend, repeat=5000)
        df_top = df_top.head(50)
        print(df_top)
        os.makedirs(dirpath, exist_ok=True)
        df_top.to_excel(path)   
    params = []
    for i in range(len(df_top)):
        d = df_top.iloc[i, :]
        p = MontblancParam.load_from_dic(d.to_dict())
        params.append(p)
    evaluate(symbol, ver, params, dirpath)
    
    

def loop():
    symbols =  ['NSDQ', 'NIKKEI', 'DOW', 'XAUUSD', 'USDJPY']
    heights = [300, 400, 400, 50, 4]
    #heights = [None, None, None, None, None]

        
    for symbol, height in zip(symbols, heights):
        if symbol == 'XAUUSD':
            sl = 5
            tp = 5
        elif symbol == 'USDJPY':
            sl = 0.01
            tp = 0.01
        else:
            sl = 50
            tp = 20
        main(symbol,  tp, sl, height)    
    
    
def load_data(symbol, begin, end):
    from mt5_trade import Mt5Trade
    from common import TimeFrame
    mt5 = Mt5Trade(3, 2, 11, 1, 3.0) 
    mt5.connect()
    mt5.set_symbol(symbol) 
    t = begin - timedelta(days=5)
    df = mt5.get_rates_jst(symbol, TimeFrame.M1, t, end)
    return df


    
def pickup_data(df0, begin, end, alpha ):
    df1 = df0[df0['jst'] <= begin]
    i0 = df1.index[-1]
    df1 = df0[df0['jst'] <= end]
    i1 = df1.index[-1]
    i0 -= alpha
    if i0 < 0:
        return None
    df = df0.iloc[i0: i1 + 1] 
    return df

def graph(symbols, ver):
    year = 2025
    month = 12
    day = 1
    for symbol in symbols:
        dir_path = f'./debug/{symbol}'
        os.makedirs(dir_path, exist_ok=True)
        df0 = load_df(symbol)
        params = load_params('Montblanc', symbol, ver)
        param = params[0]
        for day in range(day, 31):
            for j in range(1, 3):
                if j == 1:
                    hour = 8
                else:
                    hour = 20
                begin = datetime(year, month, day, hour).astimezone(JST)
                end = begin + timedelta(hours=10)
                path = os.path.join(dir_path, f'{symbol}_{year}-{month}-{day}-{j}')
                sim(symbol, df0, param, begin, end, path)
               
def sim(symbol, df, param, begin, end, filepath):
    heights = {'JP225': 100, 'US30': 100, 'US100': 50, 'XAUUSD': 5, 'USDJPY': .1}
    df0 = pickup_data(df, begin, end, 2000)
    if df0 is None:
        return
    maron = Montblanc(symbol, param)
    maron.calc(df0)
    df1 = maron.result_df()
    df1.to_csv(filepath + '_df.csv', index=False)
    
    fig = plot_chart(symbol, df1, begin, end, param, heights[symbol])
    path = filepath +'.png'
    fig2png(fig, path)
    
    df_summary, df_profit = maron.simulate_doten(begin, end)
    path = filepath + '_summary.csv'
    df_summary.to_csv(path, index=False)
    return fig
    
def test():
    symbol = 'JP225'
    #load_all_data(symbol)
    df0 = load_df(symbol)
    param = MontblancParam()
    begin = datetime(2025, 12, 23, 9).astimezone(JST)
    end = begin + timedelta(hours=6)
    dirpath = f'./debug/{symbol}'
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, f'{symbol}_')
    
    fig = sim(symbol, df0, param, begin, end, path)  
    return fig
  

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #loop()
    optimize('USDJPY', 1)
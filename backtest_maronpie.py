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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from dateutil import tz

from mt5_api import server_time_to_utc


JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from common import Columns, Indicators
from html_writer import HtmlWriter

from trade_bot import load_params
from maron_pie import MaronPie, MaronPieParam


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


def load_all_data(symbol):
    import pickle
    path = f'../Axiory/{symbol}/{symbol}_m1.pkl'
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            dic = pickle.load(f)
    else:
        dic = load_axiory_data(symbol)
        with open(path, mode='wb') as f:
            pickle.dump(dic, f)
    return dic
        
def generate_param(symbol:str, param: MaronPieParam):
    param.ma_term = rand_step(10, 30, 5)
    param.ma_method = 'ema'
    param.atr_term = rand_step(5, 30, 5)
    param.atr_shift_multiply = rand_step(1.0, 5.0, 0.2)
    param.supertrend_atr_term = rand_step(5, 40, 5)
    param.supertrend_minutes = rand_step(1, 5, 1)
    param.supertrend_multiply = rand_step(2.0, 5.0, 0.2)
    param.heikin_threshold = rand_step(0.01, 0.2, 0.01)
    param.heikin_minutes = rand_select([60, 90, 120])
   
    if symbol in ['JP225', 'US30']:
        param.sl = rand_step(40, 100, 20)
    elif symbol in ['US100']:
        param.sl = rand_step(30, 80, 10)
    elif symbol in ['SP']:
        param.sl = rand_step(2, 30, 2)   
    elif symbol in ['XAUUSD']:
        param.sl = rand_step(1, 10, 1)        
    elif symbol in ['USDJPY']:
        param.sl = rand_step(0.01, 0.2, 0.01)

def optimizer(symbol, dic, tbegin, tend, repeat=1000):
    df0 = pd.DataFrame(dic)
    df =  df0[(df0['jst'] >= tbegin) & (df0['jst'] <= tend)]
    hours = [[0, 6], [8, 8], [16, 8]]
    
    i = 0
    result = []
    #for short_term, long_term, th, sl, tp in itertools.product(short_terms, long_terms, ths, sls, tps):
    for _ in range(repeat):                   
        param = MaronPieParam()
        generate_param(symbol, param)
        maron = MaronPie(symbol, param)
        rows = []
        columns = []
        t = tbegin
        length = 2 * 24 * 60
        while t <= tend:
            t0 = t - timedelta(days=10)
            t1 = t + timedelta(days=1)
            df1 = df[(df['jst'] >= t0) & (df['jst'] <= t1)]
            if len(df1) < length:
                t += timedelta(days=1)
                continue
            index = df1.index[-1]   
            if index - length < 0:
                t += timedelta(days=1)
                continue
            df2 = df0[index - length: index + 1]            
            maron.calc(df2)
            for b, l in hours:
                t0 = t
                t0 = t0.replace(hour=b)
                t1 = t0 + timedelta(hours=l)
                (r, columns), _ = maron.simulate_doten(t0, t1)
                if len(r) > 0:
                    rows += r
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
        print('MaronPie Optimize Phase1', i, symbol, 'profit:', metric['profit'])
        i += 1
        
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
    
    
def evaluate(symbol, ver, params, dir_path):
    dic = load_all_data(symbol)
    df = pd.DataFrame(dic)
    jst = dic['jst']
    hours = [[0, 6], [8, 8], [16, 8]]
    tbegin = jst[0]
    tend = jst[-1]

    i = 0
    result = []
    for param in params:
        maron = MaronPie(symbol, param)
        rows = []
        t = tbegin
        length = 2 * 24 * 60
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
                    #print(symbol, t0, t1, 'MaronPie Optimize 2nd done')
                else:
                    pass
                    #print(symbol, t0, t1, 'MaronPie Optimize 2nd done ... No Trade')
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
        print('MaronPie Phase2', i, symbol, 'profit:', metric['profit'])
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
    for signal, label, color in zip(signals, labels, colors):
        ax.plot(timestamp, signal, color=color, alpha=0.5, label=label)    
    if graph_height is not None:
        vmin = min(signals[0])
        vmax = max(signals[0])
        center = (vmax + vmin) / 2
        center = round_number(center, graph_height / 8)
        ax.set_ylim(center - graph_height / 2, center + graph_height / 2)

def plot_signal_marker(ax, timestamp, signal, values):
    n = len(signal)
    for i in range(n):
        if signal[i] == 1:
            # Long
            marker = '^'
            color = 'green'
            alpha=0.5
        elif signal[i] == -1:
            # Short
            marker = 'v'
            color = 'red'
            alpha=0.5
        elif signal[i] == 2:
            #Close
            marker = 'x'
            color = 'gray'
            alpha=1
        else:
            continue
        ax.scatter(timestamp[i], values[i], marker=marker, color=color, alpha=alpha, s=100)
    
def plot_chart(title, timestamp, maron: MaronPie, param: MaronPieParam, graph_height):
    fig, axes = gridFig([7, 1, 1, 1], (18, 12))
    timestamp = maron.timestamp
    colors = ['gray', 'red', 'green' , 'orange', 'blue']
    labels = ['Close',  f'MA({param.ma_term}) + ATR Upper', f'({param.ma_term}) - ATR Lower', 'supertrend(+)', 'supertrend(-)']
    plot_prices(axes[0], timestamp, [maron.cl, maron.ma_upper, maron.ma_lower, maron.supertrend_upper, maron.supertrend_lower], colors, labels, graph_height)
   
    axes[1].plot(timestamp, maron.trend, color='blue', label='Trend')
    plot_signal_marker(axes[0], timestamp, maron.entries, maron.cl)
    plot_signal_marker(axes[0], timestamp, maron.exits, maron.cl)
    
    axes[2].plot(timestamp, maron.reversal, color='blue', label='Reversal')
    
    axes[3].plot(timestamp, maron.no_trend, color='red', label='No Trend')
    
    t0 = timestamp[0]
    t1 = timestamp[-1]
    title += '    ' + str(t0) + ' -> ' + str(t1)
    axes[0].set_title(title)
    
    [ax.legend() for ax in axes]

    plt.xticks(rotation=45)
    #plt.tight_layout()
    plt.close()
    return fig

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
        dir_path = f'./MaronPie/Scalping/{symbol}'
        os.makedirs(dir_path, exist_ok=True)
        path = f'./MaronPie/Scalping/{symbol}/{symbol}_{year}_{mstr}.html'
        writer.write(path)
        
    df = pd.concat(dfs)
    save_result(symbol, '#0', df, dir_path)

        
def optimize(symbol, ver):
    print('Start', symbol)
    dic = load_all_data(symbol)
    if ver >= 4:
        tbegin = datetime(2025, 2, 3).astimezone(JST)
        tend = datetime(2025, 4, 26).astimezone(JST)
    else:
        tbegin = datetime(2024, 5, 16).astimezone(JST)
        tend = datetime(2024, 9, 30).astimezone(JST)
    df = optimizer(symbol, dic, tbegin, tend, repeat=1000)
    df = df.head(50)
    print(df)
    dirpath = f'./MaronPie/v{ver}/Optimize2/{symbol}'
    os.makedirs(dirpath, exist_ok=True)
    df.to_excel(os.path.join(dirpath, f"{symbol}_v{ver}_params_phase1.xlsx"))
    
    
    params = []
    for i in range(len(df)):
        d = df.iloc[i, :]
        p = MaronPieParam.load_from_dic(d.to_dict())
        params.append(p)
    evaluate(symbol, ver, params, dirpath)
    

def optimize2(symbol, ver):
    dirpath = f'./MaronPie/v{ver}/Optimize2/{symbol}'
    df = pd.read_excel(os.path.join(dirpath, f"{symbol}_v{ver}_params_phase1.xlsx"), sheet_name='Sheet1')
    params = []
    for i in range(len(df)):
        if i == 34:
            continue
        d = df.iloc[i, :]
        p = MaronPieParam.load_from_dic(d.to_dict())
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
    
def test():
    symbol = 'JP225'
    dic = load_all_data(symbol)
    df0 = pd.DataFrame(dic)
    t0 = datetime(2025, 10, 3, 8).astimezone(JST)
    t1 = datetime(2025, 10, 3, 18).astimezone(JST)
    df1 = df0[(df0['jst'] >= t0) & (df0['jst'] <= t1)]
    index = df1.index[-1]
    length = 60 * 24 * 2
    df = df0.iloc[index - length: index + 1, :]
    jst = df['jst'].to_list()
    param = MaronPieParam()
    param.ma_term = 14
    param.ma_method = "sma"
    param.atr_term = 24
    param.atr_shift_multiply= 0.2
    param.supertrend_atr_term = 30
    param.supertrend_minutes = 15
    param.supertrend_multiply = 2.0
    param.heikin_minutes = 60
    param.heikin_threshold  = 0.03
    param.sl = 10

    maron = MaronPie(symbol, param)
    maron.calc(df)
    fig = plot_chart(symbol, jst, maron, param, 2000)
    os.makedirs('./debug', exist_ok=True)
    fig.savefig('./debug/jp225.png')
    
    (r, columns),df_profit = maron.simulate_doten(t0, t1)
    df_metric = pd.DataFrame(data=r, columns=columns)
    df_metric.to_csv('./debug/trade_jp225.csv', index=False)  
    df_profit.to_csv('./debug/jp225_profit.csv', index=False)
    
    
def test2():
    dic = load_all_data('NIKKEI')  
    print(dic['jst'][-10:])
    print(dic['utc'][-10:])
    
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #loop()
    optimize2('US100', 4)
    #test()
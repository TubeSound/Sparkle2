# backtest_cypress.py

import os
import glob
import pandas as pd
import numpy as np
from random import randint, random
import time
import itertools
from decimal import Decimal, ROUND_FLOOR
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from common import Columns, Indicators
from html_writer import HtmlWriter

from trade_bot import load_params
from dean import Dean, DeanParam


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

def backtest(title, csv_path, tp, sl, graph_height):
    param = CypressParam()
    param.long_term = 50
    param.mid_term=  25
    param.short_term = 13
    param.atr_term = 20
    param.trend_slope_th = 1.0
    param.sl = 50
    param.tp = 20  
    

    df = read_data(csv_path)
    #df = df0.iloc[:60 * 2]

    # NumPy配列として取り出す
    timestamps_np = df['jst'].tolist()
    timestamps = df["jst"].values
    op = df[Columns.OPEN].to_numpy()
    hi = df[Columns.HIGH].to_numpy()
    lo = df[Columns.LOW].to_numpy()
    cl = df[Columns.CLOSE].to_numpy()
    prices = cl
    dic = {Columns.JST: timestamps_np, Columns.OPEN: op, Columns.HIGH: hi, Columns.LOW: lo, Columns.CLOSE: cl}
        
    t0 = time.time() 
    cypress = Cypress(param)
    cypress.calc(dic)
    print('Elapsed Time: ', time.time() - t0)
    
    fig = plot_chart(title, timestamps_np, cypress, param.short_term, param.mid_term, param.long_term, graph_height)
    df_result = cypress.simulate_scalping_pips_multi(tp, sl)
    return fig , df_result

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

def set_param(symbol:str, param: DeanParam):
    param.short_term = rand_step(5, 30, 2)
    param.trend_threshold = rand_step(0.01, 0.1, 0.01)
    param.upper_timeframe = rand_select([5, 10, 15, 30, 60])
    if symbol in ['NIKKEI', 'NSDQ', 'DOW']:
        param.tp = rand_step(10, 300, 10)
        param.sl = rand_step(10, 300, 10)
        param.ema_shift = rand_step(20, 100, 10)
    elif symbol in ['SP']:
        param.tp = rand_step(2, 60, 2)
        param.sl = rand_step(2, 60, 2)    
        param.ema_shift = rand_step(4, 20, 2)
    elif symbol in ['XAUUSD']:
        param.tp = rand_step(1, 30, 1)
        param.sl = rand_step(1, 30, 1)
        param.ema_shift = rand_step(2, 10, 1)        
    elif symbol in ['USDJPY']:
        param.tp = rand_step(0.01, 0.5, 0.01)
        param.sl = rand_step(0.01, 0.5, 0.01)
        param.ema_shift = rand_step(0.01, 0.5, 0.01)

def optimizer(symbol, save_dir, timezone, repeat=2000):
    dir_path = os.path.join(save_dir, 'Optimize')
    os.makedirs(dir_path, exist_ok=True)
    files = market_data_files(symbol, timezone)
    if len(files) > 10:
        print('Optimize start', symbol)
    else:
        print('No file', symbol)
        return
    
    i = 0
    result = []
    #for short_term, long_term, th, sl, tp in itertools.product(short_terms, long_terms, ths, sls, tps):
    for _ in range(repeat):                   
        param = DeanParam()
        set_param(symbol, param)
        dean = Dean(symbol, param)
        rows = []
        columns = []
        for file in files:
            df = read_data(file)
            dean.calc(df)
            r, columns = dean.simulate_scalping()
            if len(r) > 0:
                rows += r
        df = pd.DataFrame(data=rows, columns=columns)
        if len(df) == 0:
            continue
        metric = performance(df)
        if metric['profit'] > 0:
            save_profit_graph(symbol, str(i).zfill(3), df, dir_path)
        d1 = param.to_dict()
        d0 = {'i': i}
        d = dict(**d0, **d1)
        dic = dict(**d, **metric)
        result.append(dic)
        print(i, symbol, '... done')
        i += 1
    
    keys = result[0].keys()
    dic = {}
    for key in keys:
        array = []
        for d in result:
            array.append(d[key])
        dic[key] = array
    df_result = pd.DataFrame(dic)
    df_result = df_result.sort_values('profit', ascending=False)
    df_result.to_excel(os.path.join(dir_path, f'{symbol}_optimize.xlsx'), index=False)
    
    

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
    
def plot_chart(title, timestamp, dean: Dean, param: DeanParam, graph_height):
    fig, axes = gridFig([7, 1], (18, 12))
    timestamp = dean.timestamp
    colors = ['gray', 'red', 'green', 'blue']
    labels = ['Close', f'EMA({param.short_term})-Short', 'EMA Upper', 'EMA Lower']
    plot_prices(axes[0], timestamp, [dean.cl, dean.ema_short, dean.ema_upper, dean.ema_lower], colors, labels, graph_height)
   
    axes[1].plot(timestamp, dean.trend, color='blue', label='Trend')
    plot_signal_marker(axes[0], timestamp, dean.entries, dean.cl)
    plot_signal_marker(axes[0], timestamp, dean.exits, dean.cl)
    
    
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
        dir_path = f'./Cypress/Scalping/{symbol}'
        os.makedirs(dir_path, exist_ok=True)
        path = f'./Cypress/Scalping/{symbol}/{symbol}_{year}_{mstr}.html'
        writer.write(path)
        
    df = pd.concat(dfs)
    save_result(symbol, '#0', df, dir_path)

        
def optimize(symbol, title, timezone):
    print('Start', symbol, timezone)
    optimizer(symbol, f'./Dean/{symbol}/{title}', timezone)

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
    symbol = 'USDJPY'
    files = market_data_files(symbol, [1])
    df =  read_data(files[7])
    jst = df['jst']
    param = DeanParam()
    dean = Dean(symbol, param)
    dean.calc(df)
    fig = plot_chart(symbol, jst, dean, param, 2.0)
    os.makedirs('./debug', exist_ok=True)
    fig.savefig('./debug/usdjpy.png')
    
    
    
    
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #loop()
    optimize('DOW', "1-3", [1, 2, 3])
    #test()
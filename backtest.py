# backtest.py
import os
import glob
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from common import Columns, Indicators

from technical import emas, ema_diff, detect_birdspeek, detect_taper, detect_trend, calc_range, ATRP, detect_pivots, detect_sticky

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

    
def plot0(ax, timestamps, prices, ema_fast, ema_mid, ema_slow, sticky):
    ax.plot(timestamps, prices, alpha=0.6, color='gray') 
    ax.plot(timestamps, ema_fast, color='red') 
    ax.plot(timestamps, ema_mid, color='blue', linewidth=2.0) 
    ax.plot(timestamps, ema_slow, color='orange')
    for i in range(len(sticky)):
        if sticky[i] == 1:
            ax.scatter(timestamps[i], prices[i], color='gray', marker='o', alpha=0.4, s=100)
    
def plot1(ax, timestamps_np, ema_fast_mid, ema_mid_slow, pivots, trend):
    ax.plot(timestamps_np, ema_fast_mid, color='red')
    ax.plot(timestamps_np, ema_mid_slow, color='blue')
    for i in range(len(pivots)):
        if pivots[i] == 1:
            ax.scatter(timestamps_np[i], ema_fast_mid[i], color='red', marker='v', alpha=0.4, s=200)
        elif pivots[i] == -1:
            ax.scatter(timestamps_np[i], ema_fast_mid[i], color='green', marker='^', alpha=0.4, s=200)
    ax.axhline(y=0, color='black')
    draw_bar1(ax, timestamps_np, trend)
    
def plot2(ax, timestamps_np, ema_fast_mid, ema_mid_slow, entries,exits):
    ax.plot(timestamps_np, ema_fast_mid, color='red')
    ax.plot(timestamps_np, ema_mid_slow, color='blue')
    plot_markers(ax, timestamps_np, ema_mid_slow, entries)
    for i in range(len(exits)):
        if exits[i] == 1:
            ax.scatter(timestamps_np[i], ema_fast_mid[i], color='gray', marker='x', alpha=0.4, s=200)
    ax.axhline(y=0, color='black')

def draw_bar1(ax, timestamps, trend):
    for i in range(len(timestamps) - 1):
        t0 = timestamps[i]
        t1 = timestamps[i + 1]
        if trend[i] == 1: 
            ax.axvspan(t0, t1, color='green', alpha=0.1)
        elif trend[i] == -1: 
            ax.axvspan(t0, t1, color='red', alpha=0.1)
    
def draw_bar2(ax, timestamps, volatility, th=0.015):
    for i in range(len(timestamps) - 1):
        t0 = timestamps[i]
        t1 = timestamps[i + 1]
        if volatility[i] > th: 
            ax.axvspan(t0, t1, color='yellow', alpha=0.1)

            
def plot_signals(ax, signals):
    for signal in signals:
        typ, entry_time, entry_price, exit_time, exit_price = signal
        color = 'green' if typ == 1 else 'red'
        ax.plot([entry_time, exit_time], [entry_price, exit_price], color=color, linewidth=2, linestyle='--')
        ax.scatter(entry_time, entry_price, color=color, marker='^' if typ == 1 else 'v', s=100, label=f'{typ}_entry')
        ax.scatter(exit_time, exit_price, color=color, marker='x', s=100, label=f'{typ}_exit')

def plot_markers(ax, timestamps, values, signal):
    for i in range(len(signal)):
        if signal[i] == 1:
            marker='o'
            color='green'
            alpha=0.1
            s=100
        elif signal[i] == -1:
            marker='o'
            color='red'
            alpha=0.1
            s=100
        elif signal[i] == 2:
            marker='^'
            color='green'
            alpha=0.2
            s=200
        elif signal[i] == -2:
            marker='v'
            color='red'
            alpha=0.2
            s=200    
        else:
            continue    
        ax.scatter(timestamps[i], values[i], marker=marker, color=color, alpha=alpha, s=s)

def analyze_tick(timeframe, png_path, csv_path):
    df = read_data(csv_path)
    #df = df0.iloc[:60 * 2]

    # NumPy配列として取り出す
    timestamps_np = df['jst'].tolist()
    
    timestamps = df["jst"].values
    if timeframe == 'tick':
        prices = df['bid'].to_numpy()
    else:
        op = df[Columns.OPEN].to_numpy()
        hi = df[Columns.HIGH].to_numpy()
        lo = df[Columns.LOW].to_numpy()
        cl = df[Columns.CLOSE].to_numpy()
        prices = cl
        dic = {Columns.OPEN: op, Columns.HIGH: hi, Columns.LOW: lo, Columns.CLOSE: cl}
        
    t0 = time.time()
    ema_fast, ema_mid, ema_slow = emas(timestamps_np, prices, period_fast_sec=60 * 5, period_mid_sec=60 * 13, period_slow_sec=60 * 30)
    ema_fast_mid, ema_mid_slow = ema_diff(prices, ema_fast, ema_mid, ema_slow)
    pivots = detect_pivots(timestamps_np, ema_fast_mid, 15)
    ATRP(dic, 5, 5)
    atrp = dic[Indicators.ATRP]
    atrp[0] = 0.0
    epsilon = 0.003
    peeks = detect_birdspeek(timestamps_np, ema_mid_slow, epsilon, 5, 0.003)
    tapers = detect_taper(timestamps_np, ema_fast_mid, epsilon)
    sticky = detect_sticky(timestamps_np, ema_fast_mid, ema_mid_slow, epsilon)
    trend = detect_trend(timestamps_np, ema_mid_slow, epsilon)
    rng = calc_range(op, cl, 10)
    print('Elapsed Time: ', time.time() - t0)

    fig, axes = gridFig([4, 4, 4, 2, 2], (16, 14))
    plot0(axes[0], timestamps_np, prices, ema_fast, ema_mid, ema_slow, sticky)
    plot1(axes[1], timestamps_np, ema_fast_mid, ema_mid_slow, pivots, trend)
    plot2(axes[2], timestamps_np, ema_fast_mid, ema_mid_slow, peeks, tapers)
    
    axes[3].plot(timestamps_np, atrp, color='orange', label='ATRP')
    draw_bar2(axes[3], timestamps_np, atrp)
    
    axes[4].plot(timestamps_np, rng, color='red', label='RANGEP')
    draw_bar2(axes[4], timestamps_np, rng, th=0.1)
    
    axes[1].set_ylim(-0.15, 0.15)
    axes[2].set_ylim(-0.15, 0.15)
    axes[3].set_ylim(0, 0.1)
    [ax.legend() for ax in axes]

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    
    """
    timestamp_sec = np.array([t.timestamp() for t in timestamps])
    df = pd.DataFrame({'timestamp': timestamps, 'time': timestamp_sec,'bid': prices, 'ema_fast': ema_fast, 'ema_mid': ema_mid, 'ema_slow': ema_slow})
    path = png_path[:-3] + 'csv'
    df.to_csv(path, index=False)
    """
    
def main(symbol):
    timeframe = 'M1'
    year = 2025
    for month in [4, 5, 6, 7]:
        mstr = str(month).zfill(2)
        files = glob.glob(f"../DayTradeData/{timeframe}/{symbol}/{year}-{mstr}/*")
        for file in files:
            analyze(file, symbol, timeframe, year, month)
    
def analyze(csv_path, symbol, timeframe, year, month): 
    mstr = str(month).zfill(2)
    dirpath = f'./chart/{timeframe}/{symbol}/{year}/{mstr}'
    os.makedirs(dirpath, exist_ok=True)
    _, filename = os.path.split(csv_path)
    name, ext = os.path.splitext(filename)
    analyze_tick(timeframe, os.path.join(dirpath, name + '.png'), csv_path)

if __name__ == "__main__":
    #os.chdir(os.path.dirname(os.path.abspath(__file__)))
    symbol = 'NSDQ'
    main(symbol)
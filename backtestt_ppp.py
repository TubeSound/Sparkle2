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
from html_writer import HtmlWriter

from ppp import PPP


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

def draw_bar_level(ax, timestamps, signal, level):
    n = len(signal)
    sign = np.full(n, 0)
    for i in range(n):
        if signal[i] >= level:
            sign[i] = 1
        elif signal[i] <= -level:
            sign[i] = -1
    draw_bar(ax, timestamps, sign)
            
def plot_signals(ax, signals):
    for signal in signals:
        typ, entry_time, entry_price, exit_time, exit_price = signal
        color = 'green' if typ == 1 else 'red'
        ax.plot([entry_time, exit_time], [entry_price, exit_price], color=color, linewidth=2, linestyle='--')
        ax.scatter(entry_time, entry_price, color=color, marker='^' if typ == 1 else 'v', s=100, label=f'{typ}_entry')
        ax.scatter(exit_time, exit_price, color=color, marker='x', s=100, label=f'{typ}_exit')

def plot_markers(ax, timestamps, values, signal):
    label_written0 = False
    label_written1 = False
    for i in range(len(signal)):
        if signal[i] == 1:
            marker='v'
            color='red'
            alpha=0.4
            s=100
            label='V Pivot Point (Sell)'
        elif signal[i] == -1:
            marker='^'
            color='green'
            alpha=0.4
            s=100
            label='^ Pivot Point (Buy)'
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
        if not label_written0 and signal[i] == 1:
            ax.scatter(timestamps[i], values[i], marker=marker, color=color, alpha=alpha, s=s, label=label)
            label_written0 = True
        elif not label_written1 and signal[i] == -1:    
            ax.scatter(timestamps[i], values[i], marker=marker, color=color, alpha=alpha, s=s, label=label)
            label_written1 = True
        else:
            ax.scatter(timestamps[i], values[i], marker=marker, color=color, alpha=alpha, s=s)

def analyze(title, csv_path):
    SHORT_TERM = 10
    MID_TERM = 20
    LONG_TERM = 50
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
    ppp = PPP(SHORT_TERM, MID_TERM, LONG_TERM, 0.7, 0.5)
    ppp.calc(dic)
    print('Elapsed Time: ', time.time() - t0)
   
    return plot_chart(title, timestamps_np, ppp, SHORT_TERM, MID_TERM, LONG_TERM)
    
    
def plot_chart(title, timestamp, ppp, short_term, mid_term, long_term):
    fig, axes = gridFig([4, 3, 3, 3], (12, 10))
    axes[0].plot(timestamp, ppp.cl, color='gray', label='Close')
    axes[0].plot(timestamp, ppp.short, color='red', label = f'EMA({short_term})-Short')
    axes[0].plot(timestamp, ppp.mid, color='green', label=f'EMA({mid_term})-Mid')
    axes[0].plot(timestamp, ppp.long, color='blue', label=f'EMA({long_term})-Long')
    
    axes[1].plot(timestamp, ppp.mid_long_diff_pct, color='blue', label='(Mid-Long)%')
    plot_markers(axes[1], timestamp, ppp.mid_long_diff_pct, ppp.pivot)
    
    axes[2].plot(timestamp, ppp.atrp, color='purple', label='ATRP')
    
    
    axes[3].plot(timestamp, ppp.short_slope, color='red', label='EMA-Short slope')
    axes[3].plot(timestamp, ppp.mid_slope, color='green', label='EMA-Mid slope')
    axes[3].plot(timestamp, ppp.long_slope, color='blue', label='EMA-Long slope')
    
    draw_bar_level(axes[3], timestamp, ppp.mid_slope, 0.2)
    
    
    t0 = timestamp[0]
    t1 = timestamp[-1]
    title += '          ' + str(t0) + ' -> ' + str(t1)
    axes[0].set_title(title)
    [ax.legend() for ax in axes]

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.close()
    return fig
    
    """
    timestamp_sec = np.array([t.timestamp() for t in timestamps])
    df = pd.DataFrame({'timestamp': timestamps, 'time': timestamp_sec,'bid': prices, 'ema_fast': ema_fast, 'ema_mid': ema_mid, 'ema_slow': ema_slow})
    path = png_path[:-3] + 'csv'
    df.to_csv(path, index=False)
    """
    
def main(symbol):
    timeframe = 'M1'
    year = 2025
    for month in [4, 5, 6, 7, 8]:
        mstr = str(month).zfill(2)
        files = glob.glob(f"../DayTradeData/{timeframe}/{symbol}/{year}-{mstr}/*")
        writer = HtmlWriter()
        for file in files:
            _, filename = os.path.split(file)
            name, _ = os.path.splitext(filename)
            fig = analyze(f'{name}', file)
            writer.add_fig(fig)
        os.makedirs(f'./PPP/{symbol}', exist_ok=True)
        path = f'./PPP/{symbol}/{symbol}_{year}_{mstr}.html'
        writer.write(path)
    

if __name__ == "__main__":
    #os.chdir(os.path.dirname(os.path.abspath(__file__)))
    symbol = 'NSDQ'
    main(symbol)
# backtest_cypress.py

import os
import glob
import pandas as pd
import numpy as np
import time
from decimal import Decimal, ROUND_FLOOR
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from common import Columns, Indicators
from html_writer import HtmlWriter

from cypress import Cypress, CypressParam


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

def plot_markers(ax, ax2, cypress:Cypress):
    timestamp = cypress.timestamp
    cl = cypress.cl
    values = sparkle.mid_long_diff_pct
    entry = sparkle.entry_direction
    ext = sparkle.exit_reason
    pivot = sparkle.pivot
    n = len(entry)
    label_written0 = False
    label_written1 = False
    for i in range(n):
        if pivot[i] == 1:
            a = ax2
            v = values
            marker='o'
            color='red'
            alpha=0.3
            s=50
        elif pivot[i] == -1:
            a = ax2
            v = values
            marker='o'
            color='green'
            alpha=0.3
            s=50
        elif entry[i] == Sparkle2.SHORT:
            a = ax
            v = cl
            marker='v'
            color='red'
            alpha=0.4
            s=100
            label='Sell'
        elif entry[i] == Sparkle2.LONG:
            a = ax
            v = cl
            marker='^'
            color='green'
            alpha=0.4
            s=100 
            label='Buy'   
        elif ext[i] > 0:
            a = ax
            marker='x'
            color='gray'
            alpha=0.6
            s=200
            text = Sparkle2.reason[ext[i]]
            a.text(timestamp[i], cl[i], text)
        else:
            continue
        if not label_written0 and entry[i] == Sparkle2.LONG:
            a.scatter(timestamp[i], v[i], marker=marker, color=color, alpha=alpha, s=s, label=label)
            label_written0 = True
        elif not label_written1 and entry[i] == Sparkle2.SHORT:    
            a.scatter(timestamp[i], v[i], marker=marker, color=color, alpha=alpha, s=s, label=label)
            label_written1 = True
        else:
            a.scatter(timestamp[i], v[i], marker=marker, color=color, alpha=alpha, s=s)

def round_number(x: float, step: float) -> float:
    """xをstep刻みの“下方向”キリ番に揃える"""
    dx = Decimal(str(x))
    ds = Decimal(str(step))
    q  = (dx / ds).to_integral_value(rounding=ROUND_FLOOR)
    return float(q * ds)


def analyze(title, csv_path, graph_height):
    param = CypressParam()
    param.long_term = 50
    param.mid_term=  25
    param.short_term = 13
    param.atr_term = 25
    param.trend_slope_abs = 0.5
    param.trend_retrace_pct = 0.01  

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
    #df_result = cypress.simulate_trades_from_signals()
    return fig #, df_result

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
    
def plot_chart(title, timestamp, cypress:Cypress, short_term, mid_term, long_term, graph_height):
    fig, axes = gridFig([7, 1, 2], (18, 12))
    timestamp = cypress.timestamp
    colors = ['gray', 'red', 'green', 'blue']
    labels = ['Close', f'EMA({short_term})-Short', f'EMA({mid_term})-Mid', f'EMA({long_term})-Long']
    plot_prices(axes[0], timestamp, [cypress.cl, cypress.ema_short, cypress.ema_mid, cypress.ema_long], colors, labels, graph_height)
   
    axes[1].plot(timestamp, cypress.signal, color='blue', label='Trade Signal')
    plot_signal_marker(axes[0], timestamp, cypress.signal, cypress.cl)
    
    cypress.long_slope[0] = 0.0
    axes[2].plot(timestamp, cypress.long_slope, color='blue', label='EMA-Long slope') 
    draw_bar(axes[2], timestamp, cypress.trend)


    
    t0 = timestamp[0]
    t1 = timestamp[-1]
    title += '    ' + str(t0) + ' -> ' + str(t1)
    axes[0].set_title(title)
    
    [ax.legend() for ax in axes]

    plt.xticks(rotation=45)
    #plt.tight_layout()
    plt.close()
    return fig
    
def main(symbol, graph_height):
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
            fig = analyze(f'{name}', file, graph_height)
            #dfs.append(df)
            writer.add_fig(fig)
        os.makedirs(f'./Cypress/{symbol}', exist_ok=True)
        path = f'./Cypress/{symbol}/{symbol}_{year}_{mstr}.html'
        writer.write(path)
        
    #df = pd.concat(dfs)
    #path = f'./Cypress/{symbol}/{symbol}_trade_result.csv'
    #df.to_csv(path, index=False)
        
def test():
    filepath = r"C:\Users\docs9\develop\github\TubeSound\DayTradeData\M1\NIKKEI\2025-06\NIKKEI_M1_2025-06-03-2.csv"
    analyze('test', filepath, 400)

def loop():
    symbols =  ['NSDQ', 'NIKKEI', 'DOW', 'XAUUSD', 'USDJPY']
    heights = [300, 400, 400, 5, 4]
    #heights = [None, None, None, None, None]
    for symbol, height in zip(symbols, heights):
        main(symbol, height)    
    
    
    
if __name__ == "__main__":
    #os.chdir(os.path.dirname(os.path.abspath(__file__)))
    loop()
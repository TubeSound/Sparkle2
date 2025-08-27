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

from sparkle2 import Sparkle2


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
            marker='o'
            color='red'
            alpha=0.3
            s=50
        elif signal[i] == -1:
            marker='o'
            color='green'
            alpha=0.3
            s=50
        elif signal[i] == 2:
            marker='v'
            color='red'
            alpha=0.4
            s=100
            label='Sell'
        elif signal[i] == -2:
            marker='^'
            color='green'
            alpha=0.4
            s=100 
            label='Buy'   
        else:
            continue
        if not label_written0 and signal[i] == 2:
            ax.scatter(timestamps[i], values[i], marker=marker, color=color, alpha=alpha, s=s, label=label)
            label_written0 = True
        elif not label_written1 and signal[i] == -2:    
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
    sparkle = Sparkle2(SHORT_TERM, MID_TERM, LONG_TERM, 20, 0.02)
    sparkle.calc(dic)
    print('Elapsed Time: ', time.time() - t0)
   
    return plot_chart(title, timestamps_np, sparkle, SHORT_TERM, MID_TERM, LONG_TERM)
    
    
def plot_chart(title, timestamp, sparkle, short_term, mid_term, long_term):
    fig, axes = gridFig([5, 3, 2, 3, 3], (12, 11))
    axes[0].plot(timestamp, sparkle.cl, color='gray', label='Close')
    axes[0].plot(timestamp, sparkle.short, color='red', label = f'EMA({short_term})-Short')
    axes[0].plot(timestamp, sparkle.mid, color='green', label=f'EMA({mid_term})-Mid')
    axes[0].plot(timestamp, sparkle.long, color='blue', label=f'EMA({long_term})-Long')
    
    axes[1].plot(timestamp, sparkle.mid_long_diff_pct, color='blue', label='(Mid-Long)%')
    plot_markers(axes[1], timestamp, sparkle.mid_long_diff_pct, sparkle.pivot)
    
    axes[2].plot(timestamp, sparkle.pivot, color='blue', alpha=0.4, label='Pivot')
    #axes[2].plot(timestamp, sparkle.pivot, color='red', alpha=0.4, label='Pivot-filtered')
    
    axes[3].plot(timestamp, sparkle.atrp, color='purple', label='ATRP')
    
    axes[4].plot(timestamp, sparkle.short_slope, color='red', label='EMA-Short slope')
    axes[4].plot(timestamp, sparkle.mid_slope, color='green', label='EMA-Mid slope')
    axes[4].plot(timestamp, sparkle.long_slope, color='blue', label='EMA-Long slope')
    
    draw_bar_level(axes[4], timestamp, sparkle.mid_slope, 0.2)
    
    t0 = timestamp[0]
    t1 = timestamp[-1]
    title += '          ' + str(t0) + ' -> ' + str(t1)
    axes[0].set_title(title)
    [ax.legend() for ax in axes]

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.close()
    return fig
    
def simulate_trades(timestamps,
                    price,
                    ema_mid,
                    mid_long_pct,
                    pivot, atrp,
                    touch_exit=True,
                    k_sl=1.2,
                    m_trail=1.5,
                    T=60,
                    r=1.2):
    pos = 0
    entry_i = None
    mfe = None
    exits = []
    for i in range(len(price)):
        # entry
        if pos == 0:
            if pivot[i] == -2:
                pos, entry_i, mfe = +1, i, 0.0
            elif pivot[i] ==  2:
                pos, entry_i, mfe = -1, i, 0.0
            continue

        # dynamic refs
        ret = (price[i] - price[entry_i]) / price[entry_i]
        mfe = max(mfe, ret) if pos > 0 else max(mfe, -ret)
        atrp_now = atrp[entry_i] / 100.0

        # 1) hard stop
        if pos > 0 and ret <= -k_sl * atrp_now:
            exits.append(("SL", entry_i, i))
            pos = 0
        elif pos < 0 and -ret <= -k_sl * atrp_now:
            exits.append(("SL",entry_i, i))
            pos = 0

        # 2) trailing
        if pos != 0:
            trail = m_trail * atrp_now
            if pos > 0 and (mfe - ret) >= trail:
                exits.append(("TR", entry_i, i))
                pos = 0
            if pos < 0 and (mfe - (-ret)) >= trail:
                exits.append(("TR", entry_i, i))
                pos=0

        # 3) EMA-Mid touch
        if pos != 0 and touch_exit and abs(price[i] - ema_mid[i]) / price[i] < 1e-4:
            exits.append(("MID", entry_i, i))
            pos=0

        # 4) growth fail within T bars
        if pos != 0 and (i-entry_i) >= T:
            depth0 = abs(mid_long_pct[entry_i])
            grown  = abs(mid_long_pct[i]) >= r * depth0
            slope_ok = (ema_mid[i] - ema_mid[i - 1]) > 0 if pos > 0 else (ema_mid[i] - ema_mid[i - 1]) < 0
            if not (grown and slope_ok):
                exits.append(("GF", entry_i, i))
                pos=0

        # 5) opposite pivot
        if pos > 0 and pivot[i] == 2:
            exits.append(("OPP", entry_i, i))
            pos=0
        if pos < 0 and pivot[i] == -2: 
            exits.append(("OPP", entry_i, i))
            pos =  0
    return exits

    
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
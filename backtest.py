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

from technical import detect_ema_cross, detect_pivots, detect_pivot_points, volatility, sma_sec, ema_diff, trade_signals

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

    
def plot1(ax, timestamps, prices, ema_quick, ema_fast, ema_slow, golden_cross_idx, dead_cross_idx):
    ax.plot(timestamps, prices, alpha=0.6, color='gray') 
    ax.plot(timestamps, ema_quick, color='red') 
    ax.plot(timestamps, ema_fast, color='blue', linewidth=2.0) 
    ax.plot(timestamps, ema_slow, color='orange')
    for index in golden_cross_idx: 
         ax.scatter(timestamps[index], ema_fast[index], color='red', marker='o', alpha=0.3, s=200)
    for index in dead_cross_idx: 
         ax.scatter(timestamps[index], ema_fast[index], color='green', marker='o', alpha=0.3, s=200)
    
def plot2(ax, pivots):
    times = pivots[0]
    prices = pivots[1]
    status = pivots[2]
    for t, price, s in zip(times, prices, status):
        if s == 1:
            ax.scatter(t, price, color='red', marker='v', alpha=0.4, s=80)
        elif s == -1:
            ax.scatter(t, price, color='green', marker='^', alpha=0.4, s=80)

def draw_bar1(ax, timestamps, fast_mid, mid_slow, rate=0.9):
    status = np.full(len(timestamps), 0)
    for i in range(len(timestamps) - 1):
        t0 = timestamps[i]
        t1 = timestamps[i + 1]
        if fast_mid[i] * rate > mid_slow[i] and mid_slow[i] > 0: 
            ax.axvspan(t0, t1, color='green', alpha=0.05)
            status[i] = 1
        if fast_mid[i] * rate < mid_slow[i] and mid_slow[i] < 0:
            ax.axvspan(t0, t1, color='red', alpha=0.05)
            status[i] = -1
    return status        
    
def draw_bar2(ax, timestamps, volatility, th=0.015):
    status = np.full(len(timestamps), 0)
    for i in range(len(timestamps) - 1):
        t0 = timestamps[i]
        t1 = timestamps[i + 1]
        if volatility[i] > th: 
            ax.axvspan(t0, t1, color='yellow', alpha=0.05)
            status[i] = 1
    return status

def draw_bar3(ax, timestamps, status):
    for i in range(len(timestamps) - 1):
        t0 = timestamps[i]
        t1 = timestamps[i + 1]
        if status[i] > 0: 
            ax.axvspan(t0, t1, color='green', alpha=0.05)
        if status[i] < 0:
            ax.axvspan(t0, t1, color='red', alpha=0.05)
            
def plot_signals(ax, signals):
    for signal in signals:
        typ, entry_time, entry_price, exit_time, exit_price = signal
        color = 'green' if typ == 1 else 'red'
        ax.plot([entry_time, exit_time], [entry_price, exit_price], color=color, linewidth=2, linestyle='--')
        ax.scatter(entry_time, entry_price, color=color, marker='^' if typ == 1 else 'v', s=100, label=f'{typ}_entry')
        ax.scatter(exit_time, exit_price, color=color, marker='x', s=100, label=f'{typ}_exit')

    

def analyze_tick(timeframe, png_path, csv_path):
    df = read_data(csv_path)

    # NumPy配列として取り出す
    timestamps_np = df['jst'].tolist()
    
    timestamps = df["jst"].values
    if timeframe == 'tick':
        prices = df['bid'].to_numpy()
    else:
        prices = df['close'].to_numpy()
        
    t0 = time.time()
    ema_fast, ema_mid, ema_slow, golden_cross_idx, dead_cross_idx = detect_ema_cross(timestamps_np, prices, period_fast_sec=60 * 5, period_mid_sec=60 * 13, period_slow_sec=60 * 30)
    ema_fast_mid, ema_mid_slow = ema_diff(prices, ema_fast, ema_mid, ema_slow)
    # ピボット検出
    pivots = detect_pivot_points(timestamps_np, ema_fast, slide_term_sec=60 * 5)
    volt_times, volt_values = volatility(timestamps, prices)
    ma = sma_sec(timestamps, volt_values, window_sec=60 * 15)
    signals = trade_signals(timestamps, ema_fast, ema_mid, ema_fast_mid, ema_mid_slow, volt_values)
    print('Elapsed Time: ', time.time() - t0)

    fig, axes = gridFig([6, 2, 2], (16, 10))
    plot1(axes[0], timestamps_np, prices, ema_fast, ema_mid, ema_slow, golden_cross_idx, dead_cross_idx)
    #plot2(axes[0], pivots)
    plot_signals(axes[0], signals)
    
    axes[1].plot(timestamps_np, ema_fast_mid, color='red')
    axes[1].plot(timestamps_np, ema_mid_slow, color='blue')
    axes[1].set_ylim(-0.1, 0.1)
    status1 = draw_bar1(axes[1], timestamps_np, ema_fast_mid, ema_mid_slow)
    axes[1].axhline(y=0, color='yellow')
    
    axes[2].plot(timestamps_np, volt_values, color='blue', linewidth=2.0)
    axes[2].plot(timestamps_np, ma, color='red')
    axes[2].set_ylim(0, 0.2)
    axes[2].axhline(y=0.02, color='yellow')
    status2 = draw_bar2(axes[2], timestamps_np, ma)
    status = status1 * status2
    #draw_bar3(axes[0], timestamps_np, status)
    
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
    
def main():
    symbol = 'NIKKEI'
    timeframe = 'M1'
    year = 2025
    month = 7
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
    main()
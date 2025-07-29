# technical.py

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.cluster import DBSCAN

def detect_pivots(timestamps, prices, slide_term_sec=60, center_sec=5):
    pivots = detect_pivot_points(timestamps, prices, slide_term_sec, center_sec)
    pivot_times, pivot_prices, pivot_types = pivots
    return extract_representative_pivot_indices(pivot_times, pivot_prices, pivot_types, timestamps, cluster_eps_sec=5, min_cluster_size=5)

def detect_pivot_points(timestamps, prices, slide_term_sec=60, center_sec=5):
    window_radius = int(slide_term_sec / 2)
 
    pivot_times = []
    pivot_prices = []
    pivot_types = []

    # === ピボット検出ロジック ===
    for i in range(window_radius, len(prices) - window_radius):
        full_window = prices[i - window_radius:i + window_radius + 1]
        center_range = prices[i - center_sec:i + center_sec + 1]

        if np.max(center_range) == np.max(full_window):
            pivot_times.append(timestamps[i].replace(microsecond=0))
            pivot_prices.append(prices[i])
            pivot_types.append(1)

        elif np.min(center_range) == np.min(full_window):
            pivot_times.append(timestamps[i].replace(microsecond=0))
            pivot_prices.append(prices[i])
            pivot_types.append(-1)
    return pivot_times, pivot_prices, pivot_types

def extract_representative_pivot_indices(pivot_times, pivot_prices, pivot_types, timestamps, cluster_eps_sec=5, min_cluster_size=5):
    timestamp_sec = np.array([t.timestamp() for t in pivot_times])
    db = DBSCAN(eps=cluster_eps_sec, min_samples=1).fit(timestamp_sec.reshape(-1, 1))

    timestamps_array = np.array(timestamps)
    results = []

    for cluster_id in np.unique(db.labels_):
        mask = db.labels_ == cluster_id
        if np.sum(mask) < min_cluster_size:
            continue

        group_prices = np.array(pivot_prices)[mask]
        group_types = np.array(pivot_types)[mask]
        group_times = np.array(pivot_times)[mask]

        if group_types[0] == 1:
            idx_in_group = np.argmax(group_prices)
        elif group_types[0] == -1:
            idx_in_group = np.argmin(group_prices)
        else:
            continue

        t_rep = group_times[idx_in_group]
        # indexを求める（最初に一致するインデックス）
        idx_rep = np.searchsorted(timestamps_array, t_rep)
        if idx_rep < len(timestamps_array) and timestamps_array[idx_rep] == t_rep:
            results.append((idx_rep, group_types[0]))

    return results


def extract_representative_pivots(pivot_times, pivot_prices, pivot_types, timestamps, cluster_eps_sec=5, min_cluster_size=5):
    #timestamp_sec = np.array(timestamps.astype("datetime64[s]")).astype(np.int64)
    timestamp_sec = np.array([t.timestamp() for t in pivot_times])
    db = DBSCAN(eps=cluster_eps_sec, min_samples=1).fit(timestamp_sec.reshape(-1, 1))
    reps_time, reps_price, reps_type = [], [], []
    for cluster_id in np.unique(db.labels_):
        mask = db.labels_ == cluster_id
        if np.sum(mask) < min_cluster_size:
            continue

        group_prices = np.array(pivot_prices)[mask]
        group_types = np.array(pivot_types)[mask]
        group_times = np.array(pivot_times)[mask]

        if group_types[0] == 1:
            idx = np.argmax(group_prices)
        elif group_types[0] == -1:
            idx = np.argmin(group_prices)
        else:
            continue

        reps_time.append(group_times[idx])
        reps_price.append(group_prices[idx])
        reps_type.append(group_types[0])

    return reps_time, reps_price, reps_type


def detect_ema_cross(timestamps, prices, period_fast_sec=30, period_mid_sec=60, period_slow_sec=900):
    alpha_fast = 1 / period_fast_sec
    alpha_mid = 1 / period_mid_sec
    alpha_slow = 1 / period_slow_sec

    ema_fast = np.zeros_like(prices)
    ema_mid = np.zeros_like(prices)
    ema_slow = np.zeros_like(prices)
    ema_fast[0] = ema_mid[0] = ema_slow[0] = prices[0]

    for i in range(1, len(prices)):
        dt = (timestamps[i] - timestamps[i - 1]) / np.timedelta64(1, 's')
        alpha_f = 1 - np.exp(-alpha_fast * dt)
        alpha_m = 1 - np.exp(-alpha_mid * dt)
        alpha_s = 1 - np.exp(-alpha_slow * dt)
        ema_fast[i] = alpha_f * prices[i] + (1 - alpha_f) * ema_fast[i - 1]
        ema_mid[i] = alpha_m* prices[i] + (1 - alpha_m) * ema_mid[i - 1]
        ema_slow[i] = alpha_s * prices[i] + (1 - alpha_s) * ema_slow[i - 1]

    # クロス検出
    cross_type = np.sign(ema_mid - ema_slow)
    cross_diff = np.diff(cross_type)
    golden_cross_idx = np.where(cross_diff == 2)[0] + 1
    dead_cross_idx = np.where(cross_diff == -2)[0] + 1
    return ema_fast, ema_mid, ema_slow, golden_cross_idx, dead_cross_idx

def ema_diff(prices, ema_fast, ema_mid, ema_slow):
    # 差分パーセント計算
    with np.errstate(divide='ignore', invalid='ignore'):
        fast_mid_diff_pct = 100 * (ema_fast - ema_mid) / prices
        mid_slow_diff_pct = 100 * (ema_mid - ema_slow) / prices
    return fast_mid_diff_pct, mid_slow_diff_pct
    
    
def volatility(timestamps, prices, window_sec=300):
    vol_times = []
    vol_values = []

    for i in range(len(prices)):
        t_end = timestamps[i]
        t_start = t_end - np.timedelta64(window_sec, 's')
        mask = (timestamps >= t_start) & (timestamps <= t_end)

        if np.sum(mask) < 5:
            vol_values.append(np.nan)
        else:
            window_prices = prices[mask]
            mean_price = np.mean(window_prices)
            std_dev = np.std(window_prices)
            if mean_price != 0:
                vol_percent = (std_dev / mean_price) * 100  # ← ここが変更点
                vol_values.append(vol_percent)
            else:
                vol_values.append(np.nan)

        vol_times.append(
            pd.to_datetime(str(t_end)).tz_localize('UTC').tz_convert('Asia/Tokyo').to_pydatetime()
        )

    return vol_times, vol_values

    

def sma_sec(timestamps, values, window_sec):
    # numpy.datetime64 → Unix秒（int64）
    times = timestamps.astype('datetime64[s]').astype('int64')
    values = np.asarray(values)

    if len(times) != len(values):
        raise ValueError("timestamps と values の長さが一致していません")

    sma = np.full(len(values), np.nan)

    for i in range(len(values)):
        t0 = times[i] - window_sec
        mask = (times >= t0) & (times <= times[i])
        if np.any(mask):
            sma[i] = np.mean(values[mask])  # ← mask が正しく boolean array なら OK

    return sma
  
    
#　timestamps: numpy datetime    
def sma_sec_old(timestamps, values, window_sec):
    # 秒単位でrollingしたい場合
    times = np.array([t.timestamp() for t in timestamps])
    sma = np.full(len(values), np.nan)
    for i in range(len(values)):
        t0 = times[i] - window_sec
        mask = (times >= t0) & (times <= times[i])
        if np.sum(mask) > 0:
            sma[i] = np.mean(values[mask])
    return sma

def detect_ema_pivots(timestamps: np.ndarray, ema: np.ndarray, lookback_sec: int = 150, threshold: float = 0.00005) -> Tuple[List, List]:
    """
    timestamps: np.ndarray of np.datetime64
    ema: np.ndarray of float
    lookback_sec: 前後の秒数（合計2*lookback_secの範囲で比較）
    threshold: ピボットとして認める深さ（差分）
    
    Returns:
        pivot_times: List of timestamps
        pivot_prices: List of EMA values (pivots)
    """
    pivot_times = []
    pivot_prices = []

    for i in range(len(ema)):
        t_center = timestamps[i]
        t_min = t_center - np.timedelta64(lookback_sec, 's')
        t_max = t_center + np.timedelta64(lookback_sec, 's')

        idx_window = np.where((timestamps >= t_min) & (timestamps <= t_max))[0]
        if len(idx_window) < 3:
            continue

        center_price = ema[i]
        window_prices = ema[idx_window]
        center_idx = np.where(idx_window == i)[0]
        if len(center_idx) == 0:
            continue

        other_prices = np.delete(window_prices, center_idx)

        if np.all(center_price > other_prices):
            depth = center_price - np.max(other_prices)
            if depth >= threshold:
                pivot_times.append(timestamps[i])
                pivot_prices.append(center_price)

        elif np.all(center_price < other_prices):
            depth = np.min(other_prices) - center_price
            if depth >= threshold:
                pivot_times.append(timestamps[i])
                pivot_prices.append(center_price)

    return pivot_times, pivot_prices

def trade_signals(timestamps, prices, ema_mid, ema_fast_mid, ema_mid_slow,  volatility_sma, rate=0.9, th=0.015, touch_distance=0.0001):
    position = None  # None / 'long' / 'short'
    entry_idx = None
    signals = []  # [(signal_type, entry_time, entry_price, exit_time, exit_price)]

    for i in range(1, len(prices)):
        # Entry signals
        asc = ema_fast_mid[i] * rate > ema_mid_slow[i] and ema_mid_slow[i] > 0
        desc = ema_fast_mid[i] * rate < ema_mid_slow[i] and ema_mid_slow[i] < 0
        vol_ok = volatility_sma[i] >= th

        if position is None:
            if asc and vol_ok:
                position = 1
                entry_idx = i
            elif desc and vol_ok:
                position = -1
                entry_idx = i

        # Exit signals
        elif position == 1:
            if abs(prices[i] - ema_mid[i]) / prices[i] < touch_distance:
                signals.append((1, timestamps[entry_idx], prices[entry_idx], timestamps[i], prices[i]))
                position = None

        elif position == -1:
            if abs(prices[i] - ema_mid[i]) / prices[i] < touch_distance:
                signals.append((-1, timestamps[entry_idx], prices[entry_idx], timestamps[i], prices[i]))
                position = None

    return signals

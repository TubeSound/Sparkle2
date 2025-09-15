import numpy as np
from technical import calc_sma, calc_ema, ATRP, slopes, detect_pivots, is_nans, slice_upper_abs, detect_perfect_order
from common import Columns, Indicators
import pandas as pd

# --- BeansParam を拡張 ---
class BeansParam:
    long_term = 50
    mid_term = 25
    short_term = 13
    atr_term = 20
    slice_threshold = 0.05
    signal_filter_window = 10
    signal_filter_nummax = 3
    trend_slope_threshold = 0.1
    perfect_order_ignore_level = 0.01
    perfect_order_min_bars = 10
    
    atr_active = 0.03      # ATRPがこの閾値以上なら「アクティブ」
    max_retries = 3        # 同一アクティブ区間での最大再トライ回数
    
    trend_slope_abs = 0.5      # |EMA-mid slope| ≥ 0.5 で“トレンドあり”
    trend_retrace_pct = 0.01   # トレース中ピーク(谷)から0.01%逆行で“反転”


class Beans:
    TECHNICAL_CLOSE = 2
    STOP_LOSS = 2
    TRAILING_STOP = 3
    
    LONG = 1
    SHORT = -1
    CLOSE = 2
    
    reason = {TECHNICAL_CLOSE: 'TECH', STOP_LOSS:'SL', TRAILING_STOP: 'TRAIL'}
    
    
    
    
    
    def __init__(self, param: BeansParam):
        self.param = param
        
    def difference(self, vector1, vector2, ma_window):
        n = len(vector1)
        dif = np.full(n, np.nan)
        for i in range(n):
            if is_nans([vector1[i], vector2[i]]):
                continue
            if vector2[i] == 0.0:
                continue
            dif[i] = (vector1[i] - vector2[i]) / vector2[i] * 100.0
        if ma_window > 0:
            return calc_sma(dif, ma_window)    
        else:
            return dif
        
    def count_value(self, array, value):
        n = 0
        for v in array:
            if v == value:
                n += 1
        return n
        
    def signal_filter(self, signal, dif, window, num_max):
        n = len(signal)
        counts = [0, 0]
        out = np.full(n, 0)
        for i in range(1, n):
            if i == 0:
                if signal[i] != 0:
                    out[i] = signal[i]
            else:
                if signal[i] != 0:
                    begin = i - window
                    if begin < 0:
                        begin = 0
                    s = signal[begin: i]
                    d = dif[begin: i] 
                    if self.count_value(s, signal[i]) < num_max: 
                        if signal[i] == 1:
                            # Long   
                            if dif[i] < min(d):
                                out[i] = 1
                                counts[0] += 1
                        else:
                            # Short
                            if dif[i] > max(d):
                                out[i] = -1
                                counts[1] += 1
        return out, counts                    
        
                    
    def detect_trend(self, signal, threshold):
        n = len(signal)
        out = np.full(n, 0)
        for i in range(n):
            if signal[i] >= threshold:
                out[i] = 1
            elif signal[i] <= -threshold:
                out[i] = -1
        return out
        
    def mask_with_trend(self, signal, trend):
        n = len(signal)
        out = np.full(n, 0)
        for i in range(n):
            if signal[i] == trend[i]:
                out[i] = signal[i]
        return out
    
    def stabilize_signal(self, signal, min_bars=3):
        n = len(signal)
        out = np.zeros(n, dtype=int)
        count = 0
        prev = 0
        for i in range(n):
            if signal[i] == prev and signal[i] != 0:
                count += 1
            else:
                count = 1
            prev = signal[i]
            if count >= min_bars:
                out[i] = signal[i]
        return out

        
    def calc(self, dic):
        cl = dic[Columns.CLOSE]
        self.cl = cl
        self.op = dic[Columns.OPEN]
        self.hi = dic[Columns.HIGH]
        self.lo = dic[Columns.LOW]
        self.timestamp = dic[Columns.JST]        
        self.ema_short = calc_ema(cl, self.param.short_term)
        self.ema_mid = calc_ema(cl, self.param.mid_term)
        self.ema_long = calc_ema(cl, self.param.long_term)
        self.dif = self.difference(self.cl, self.ema_short, 0)
    
        self.atrp = ATRP(dic, self.param.atr_term)        
        short_slope = slopes(self.ema_short, 10)
        mid_slope = slopes(self.ema_mid, 10)
        long_slope = slopes(self.ema_long, 10)
        self.short_slope = short_slope
        self.mid_slope = mid_slope
        self.long_slope = long_slope
        
        self.mid_trend = self.detect_trend(mid_slope, self.param.trend_slope_threshold)
        
        #signal, counts = slice_upper_abs(self.dif, self.param.slice_threshold)
        
        #self.signal, self.counts = self.signal_filter(signal, self.dif, self.param.signal_filter_window, self.param.signal_filter_nummax)
        
        
        # Perfect Order
       # --- Perfect Order ---
        po = detect_perfect_order(self.ema_long, self.ema_mid, self.ema_short,self.param.perfect_order_ignore_level)

        # 点滅抑制（5本連続以上で有効化）
        #po1 = self.stabilize_signal(po0, min_bars=self.param.perfect_order_min_bars)


        # アクティブ判定（ATRP ≥ 閾値）
        self.active = (self.atrp >= self.param.atr_active).astype(int)

        # 参考表示用（従来のperfect_orderは消してもOK）
        self.perfect_order = po
        
        self.trend = self._build_trend_signal(self.ema_mid, self.mid_slope, self.param.trend_slope_abs, self.param.trend_retrace_pct)
        self.trade_signal = self.detect_signal()
        
    
    def detect_signal(self):
        n = len(self.active)
        signal = np.full(n, 0)
        state = 0
        count = 0
        for i in range(n):
            if state == 0:
                if self.active[i] == 1 and count < self.param.max_retries:
                    if self.perfect_order[i] == 1:
                        signal[i] = self.LONG
                        state = self.LONG
                        count += 1
                    elif self.perfect_order[i] == -1:
                        signal[i] = self.SHORT
                        state = self.SHORT
                        count += 1
            else:
                if self.active[i] == 0:
                    signal[i] = self.CLOSE
                    state = 0
                    count = 0
                else:
                    if (state == 1 and self.trend[i] == -1) or (state == -1 and self.trend[i] == 1):
                        signal[i] = self.CLOSE
                        state = 0
        return signal
    
    
    # beans.py （Beansクラス内に追加）
    def _build_trend_signal(self, ema_mid: np.ndarray, mid_slope: np.ndarray,
                        slope_abs_th: float, retrace_pct: float):
        """
        トレンド開始条件: |slope| >= slope_abs_th
        上昇トレンド中 : ema_mid% (= (ema_mid/base-1)*100) の running-peak から retrace_pct% 逆行で反転
        下降トレンド中 : running-trough から retrace_pct% 反発で反転
        """
        n = len(ema_mid)
        trend   = np.zeros(n, dtype=int)      # +1/-1/0（現在のトレンド方向）
        ret_arr   = np.full(n, np.nan)          # 開始基準からのEMA-mid変化率[%]
        peak_arr  = np.full(n, np.nan)          # running-peak（上昇時）
        trough_arr= np.full(n, np.nan)          # running-trough（下降時）
        flip_arr  = np.zeros(n, dtype=int)      # 反転イベント（上昇→1 / 下降→-1）
        active_arr= np.zeros(n, dtype=int)      # トレンド中=1

        state = 0         # 0:なし, 1:上昇, -1:下降
        base  = np.nan    # トレンド開始時の ema_mid
        peak  = 0.0
        trough= 0.0

        for i in range(n):
            s = mid_slope[i]

            # まだトレンドなし → スタート判定
            if state == 0:
                if s >= slope_abs_th:
                    state = 1
                    base = ema_mid[i]
                    peak = trough = 0.0
                    ret = 0.0
                elif s <= -slope_abs_th:
                    state = -1
                    base = ema_mid[i]
                    peak = trough = 0.0
                    ret = 0.0
                else:
                    ret_arr[i] = np.nan
                    continue
            else:
                # 進行中は基準からの%変化を更新
                ret = 100.0 * (ema_mid[i] - base) / base if base != 0 else 0.0

            # 上昇トレンドの管理
            if state > 0:
                peak = max(peak, ret)
                drawdown = peak - ret
                if drawdown >= retrace_pct:
                    # 反転シグナル
                    flip_arr[i] = 1
                    state = 0
                else:
                    active_arr[i] = 1

            # 下降トレンドの管理
            elif state < 0:
                trough = min(trough, ret)
                rebound = ret - trough
                if rebound >= retrace_pct:
                    flip_arr[i] = -1
                    state = 0
                else:
                    active_arr[i] = 1

            trend[i] = state
            ret_arr[i] = ret
            peak_arr[i] = peak
            trough_arr[i] = trough

            # 傾きがしぼんだら“終了”（反転扱いではない）
            if state != 0 and abs(s) < slope_abs_th:
                state = 0

        return trend

                                      
    def simulate_trades_from_signals(   self,
                                        k_sl: float = 1.2,           # ハードSL = k_sl * ATRP(entry)
                                        use_trailing: bool = False,  # トレーリングを使うなら True
                                        m_trail: float = 1.5,        # トレーリング幅 = m_trail * ATRP(entry)
                                        fees_pct: float = 0.0,       # 往復コスト（%）
                                    ) -> pd.DataFrame:
    
    
        #前提：self.trade_signal が生成済み
        #  - エントリー:  LONG(=1), SHORT(=-1)
        #  - クローズ  :  LONG_CLOSE(=2), SHORT_CLOSE(=-2)

        #退出条件の優先順位:
        #  1) ハードSL（ATRP基準）
        #  2) トレーリング（任意）
        #  3) 信号によるクローズ（LONG_CLOSE/SHORT_CLOSE）
        

        price = np.asarray(self.cl, dtype=float)
        atrp  = np.asarray(self.atrp, dtype=float)          # % 表現
        sig   = np.asarray(self.trade_signal, dtype=int)
        ts    = np.asarray(self.timestamp)

        n = len(price)
        position = 0                    # 0/±1
        entry_i  = None
        mfe_pct  = 0.0                  # Maximum Favorable Excursion
        curr_id  = 1

        # 可視化用
        entries    = np.zeros(n, dtype=int)
        entry_dir  = np.zeros(n, dtype=int)
        exits      = np.zeros(n, dtype=int)
        exit_reason= np.zeros(n, dtype=int)

        rows = []

        for i in range(n):
            s = sig[i]

            # --- エントリー ---
            if position == 0:
                if s == self.LONG:
                    position = self.LONG
                    entry_i  = i
                    mfe_pct  = 0.0
                    entries[i] = curr_id
                    entry_dir[i] = self.LONG
                elif s == self.SHORT:
                    position = self.SHORT
                    entry_i  = i
                    mfe_pct  = 0.0
                    entries[i] = curr_id
                    entry_dir[i] = self.SHORT
                continue

            # --- 建玉あり：損益とATRPを更新 ---
            ret     = (price[i] - price[entry_i]) / price[entry_i]    # +:上昇率
            fav_now = ret if position > 0 else -ret                   # 有利方向％
            mfe_pct = max(mfe_pct, fav_now)
            atrp0   = max(1e-8, atrp[entry_i] / 100.0)                # 0除算保険

            # 1) ハードSL（ATRP基準）
            if fav_now <= -k_sl * atrp0:
                pnl_pct = ret * position - fees_pct
                pnl_abs = (price[i] - price[entry_i]) * position
                exits[i], exit_reason[i] = curr_id, self.STOP_LOSS
                rows.append((ts[entry_i], ts[i], curr_id, position,
                            price[entry_i], price[i], pnl_pct, pnl_abs, mfe_pct, "STOP_LOSS"))
                position = 0; curr_id += 1
                continue

            # 2) トレイリング（任意）
            if use_trailing:
                trail = m_trail * atrp0
                if (mfe_pct - fav_now) >= trail:
                    pnl_pct = ret * position - fees_pct
                    pnl_abs = (price[i] - price[entry_i]) * position
                    exits[i], exit_reason[i] = curr_id, self.TRAILING_STOP
                    rows.append((ts[entry_i], ts[i], curr_id, position,
                                price[entry_i], price[i], pnl_pct, pnl_abs, mfe_pct, "TRAILING_STOP"))
                    position = 0; curr_id += 1
                    continue

            # 3) 信号によるクローズ
            if (position == self.LONG and s == self.CLOSE) or \
            (position == self.SHORT and s == self.CLOSE):
                pnl_pct = ret * position - fees_pct
                pnl_abs = (price[i] - price[entry_i]) * position
                exits[i], exit_reason[i] = curr_id, self.TECHNICAL_CLOSE
                rows.append((ts[entry_i], ts[i], curr_id, position,
                            price[entry_i], price[i], pnl_pct, pnl_abs, mfe_pct, self.reason[self.TECHNICAL_CLOSE]))
                position = 0; curr_id += 1
                continue

        # 可視化用に保持
        self.entry_direction = entry_dir
        self.exit_reason     = exit_reason

        if not rows:
            return pd.DataFrame(columns=[
                "entry_time","exit_time","id","side",
                "entry_px","exit_px","pnl_pct","pnl_abs","mfe_pct","exit_reason"
            ])

        return pd.DataFrame(rows, columns=[
            "entry_time","exit_time","id","side",
            "entry_px","exit_px","pnl_pct","pnl_abs","mfe_pct","exit_reason"
        ])

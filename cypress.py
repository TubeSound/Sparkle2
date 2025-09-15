import numpy as np
from technical import calc_sma, calc_ema, ATRP, slopes, detect_pivots, is_nans, slice_upper_abs, detect_perfect_order
from common import Columns, Indicators
import pandas as pd

class CypressParam:
    long_term = 50
    mid_term = 25
    short_term = 13
    atr_term = 20

    trend_slope_threshold = 0.2


class Cypress:
    TECHNICAL_CLOSE = 2
    STOP_LOSS = 2
    TRAILING_STOP = 3
    
    LONG = 1
    SHORT = -1
    CLOSE = 2
    
    reason = {TECHNICAL_CLOSE: 'TECH', STOP_LOSS:'SL', TRAILING_STOP: 'TRAIL'}
    
    
    
    
    
    def __init__(self, param: CypressParam):
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
    
    
        self.atrp = ATRP(dic, self.param.atr_term)        
        short_slope = slopes(self.ema_short, 10)
        mid_slope = slopes(self.ema_mid, 10)
        long_slope = slopes(self.ema_long, 10)
        self.short_slope = short_slope
        self.mid_slope = mid_slope
        self.long_slope = long_slope
        
        self.trend = self.detect_trend(long_slope, self.param.trend_slope_threshold)
        
        self.signal = self.detect_signal()
    
    def detect_signal(self):
        n = len(self.cl)
        signal = np.full(n, 0)
        for i in range(1, n):
            if self.trend[i] == 1:
                if self.cl[i - 1] <= self.ema_short[i - 1] and self.cl[i] >= self.ema_short[i]:
                    signal[i] = self.SHORT 
            elif self.trend[i] == -1:
                if self.cl[i - 1] >= self.ema_short[i - 1] and self.cl[i] <= self.ema_short[i]:
                    signal[i] = self.LONG                
        return signal

                                      
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

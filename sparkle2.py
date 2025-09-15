import numpy as np
from technical import calc_sma, ATRP, emas, calc_atr, slopes, majority_filter, ema_diff, detect_pivots, detect_near
from common import Columns, Indicators
import pandas as pd


class Sparkle2:
    STOP_LOSS = 1
    TRAILING_STOP = 2
    EMA_TOUCH = 3
    GROWTH_FAIL = 4
    OPPOSITE_PIVOT = 5
    
    LONG = 1
    SHORT = -1
    LONG_CLOSE = 2
    SHORT_CLOSE = -2
    
    reason = {STOP_LOSS:'SL', TRAILING_STOP: 'TRAIL', EMA_TOUCH: 'TOUCH', GROWTH_FAIL: 'FAIL', OPPOSITE_PIVOT: 'OPPST'}
    
    def __init__(self, short_term, mid_term, long_term, mid_slope_lower):
        self.short_term = short_term
        self.mid_term = mid_term
        self.long_term = long_term
        self.mid_slope_lower = mid_slope_lower
            
        self.pivot_term = 20
        self.pivot_depth_min = 0.2
        
    def calc(self, dic):
        cl = dic[Columns.CLOSE]
        self.cl = cl
        self.op = dic[Columns.OPEN]
        self.hi = dic[Columns.HIGH]
        self.lo = dic[Columns.LOW]
        self.timestamp = dic[Columns.JST]        
        self.short, self.mid, self.long = emas(self.timestamp, cl, 60 * self.short_term, 60 * self.mid_term, 60 * self.long_term)
        
        self.atrp = ATRP(dic, 14)        
        short_slope = slopes(self.short, 10)
        mid_slope = slopes(self.mid, 10)
        long_slope = slopes(self.long, 10)
        self.short_slope = short_slope
        self.mid_slope = mid_slope
        self.long_slope = long_slope
        
        short_mid_diff_pct, mid_long_diff_pct = ema_diff(self.cl, self.short, self.mid, self.long)
        self.short_mid_diff_pct =  short_mid_diff_pct
        self.mid_long_diff_pct = mid_long_diff_pct
        
        self.sign, self.entries, self.exits = self.detect_signal(mid_slope, self.mid_slope_lower)
        
        self.squeeze = detect_near(short_mid_diff_pct, mid_long_diff_pct, 0.001)
        self.pivot = detect_pivots(mid_long_diff_pct, self.pivot_term, self.pivot_depth_min)
        
        
        
    def detect_signal(self, slope, threshold):
        n = len(slope)
        sign = np.full(n, 0)
        for i in range(n):
            if slope[i] > threshold:
                sign[i] = self.LONG
            elif slope[i] < -threshold:
                sign[i] = self.SHORT
                
        entries = np.full(n, 0)
        exits = np.full(n, 0)      
        state = 0
        for i in range(n):
            if state == 0:
                if sign[i] == self.LONG:
                    entries[i] = self.LONG
                    state = self.LONG
                elif sign[i] == self.SHORT:
                    entries[i] = self.SHORT
                    state = self.SHORT
            else:
                if sign[i] == 0:
                    exits[i] = 1
                    state = 0
                else:
                    if sign[i] != state:
                        exits[i] = 1
                        entries[i] = sign[i] 
                        state = sign[i]
        return sign, entries, exits

      
    def simulate_trades(
            self,
            touch_exit: bool = True,
            k_sl: float = 1.2,
            m_trail: float = 1.5,
            T: int = 60,
            r: float = 1.2,
            fees_pct: float = 0.0,   # 片道/往復どちらでも。ここでは“往復”想定で1トレードごとに引く
        ) -> pd.DataFrame:
            price = np.asarray(self.cl, dtype=float)
            mid   = np.asarray(self.mid, dtype=float)
            pivot = np.asarray(self.pivot)
            atrp  = np.asarray(self.atrp, dtype=float)   # %表現を想定
            ts    = np.asarray(self.timestamp)

            n = len(price)
            position = 0   # 0/±1
            entry_i = None
            mfe_pct = 0.0  # Maximum Favorable Excursion in %
            curr_id = 1

            # plot用
            entries = np.zeros(n, dtype=int)
            entry_dir = np.zeros(n, dtype=int)
            exits  = np.zeros(n, dtype=int)
            reason = np.zeros(n, dtype=int)

            rows = []

            # 差分（EMAタッチ判定で使う）
            def crossed(prev, curr):  # 符号反転＝クロス
                return (prev == 0) or (np.sign(prev) != np.sign(curr))

            for i in range(n):
                # ---- ENTRY（確定ピボットのみ）
                if position == 0:
                    if pivot[i] == -2:  # Buy
                        position = self.LONG
                        entry_i = i
                        mfe_pct = 0.0
                        entries[i] = curr_id
                        entry_dir[i] = self.LONG
                    elif pivot[i] ==  2:  # Sell
                        position = self.SHORT
                        entry_i = i
                        mfe_pct = 0.0
                        entries[i] = curr_id
                        entry_dir[i] = self.SHORT
                    continue

                # ---- 以降は建玉あり
                ret = (price[i] - price[entry_i]) / price[entry_i]   # +:上昇率
                fav_now = ret if position > 0 else -ret              # 有利方向の％
                mfe_pct = max(mfe_pct, fav_now)

                atrp0 = max(1e-8, atrp[entry_i] / 100.0)  # 0除算保険

                # 1) ハードSL（ATRP基準）
                if fav_now <= -k_sl * atrp0:
                    pnl_pct = ret * position - fees_pct
                    pnl_abs = (price[i] - price[entry_i]) * position
                    exits[i], reason[i] = curr_id, self.STOP_LOSS
                    rows.append((ts[entry_i], ts[i], curr_id, position,
                                price[entry_i], price[i],
                                pnl_pct, pnl_abs, mfe_pct, "STOP_LOSS"))
                    position = 0; curr_id += 1
                    continue  # 同バーで二重発火防止

                # 2) トレイリング（ATRP基準）
                trail = m_trail * atrp0
                if (mfe_pct - fav_now) >= trail:
                    pnl_pct = ret * position - fees_pct
                    pnl_abs = (price[i] - price[entry_i]) * position
                    exits[i], reason[i] = curr_id, self.TRAILING_STOP
                    rows.append((ts[entry_i], ts[i], curr_id, position,
                                price[entry_i], price[i],
                                pnl_pct, pnl_abs, mfe_pct, "TRAILING_STOP"))
                    position = 0; curr_id += 1
                    continue

                # 3) EMA-Mid タッチ or クロス
                if touch_exit:
                    d_prev = (price[i-1] - mid[i-1]) if i > 0 else 0.0
                    d_curr = (price[i]   - mid[i])
                    touched = abs(d_curr) / max(1e-8, price[i]) < 1e-4
                    if touched or crossed(d_prev, d_curr):
                        pnl_pct = ret * position - fees_pct
                        pnl_abs = (price[i] - price[entry_i]) * position
                        exits[i], reason[i] = curr_id, self.EMA_TOUCH
                        rows.append((ts[entry_i], ts[i], curr_id, position,
                                    price[entry_i], price[i],
                                    pnl_pct, pnl_abs, mfe_pct, "EMA_TOUCH"))
                        position = 0; curr_id += 1
                        continue

                # 4) 成長不全（T本）
                if (i - entry_i) >= T:
                    depth0 = abs(self.mid_long_diff_pct[entry_i])
                    grown  = abs(self.mid_long_diff_pct[i]) >= r * depth0
                    slope_ok = (mid[i] - mid[i-1]) > 0 if position > 0 else (mid[i] - mid[i-1]) < 0
                    if not (grown and slope_ok):
                        pnl_pct = ret * position - fees_pct
                        pnl_abs = (price[i] - price[entry_i]) * position
                        exits[i], reason[i] = curr_id, self.GROWTH_FAIL
                        rows.append((ts[entry_i], ts[i], curr_id, position,
                                    price[entry_i], price[i],
                                    pnl_pct, pnl_abs, mfe_pct, "GROWTH_FAIL"))
                        position = 0; curr_id += 1
                        continue

                # 5) 逆ピボット
                if (position > 0 and pivot[i] == 2) or (position < 0 and pivot[i] == -2):
                    pnl_pct = ret * position - fees_pct
                    pnl_abs = (price[i] - price[entry_i]) * position
                    exits[i], reason[i] = curr_id, self.OPPOSITE_PIVOT
                    rows.append((ts[entry_i], ts[i], curr_id, position,
                                price[entry_i], price[i],
                                pnl_pct, pnl_abs, mfe_pct, "OPPOSITE_PIVOT"))
                    position = 0; curr_id += 1
                    continue

            # 可視化用
            #self.entries = entries
            self.entry_direction = entry_dir
            #self.exits = exits
            self.exit_reason = reason

            if not rows:
                return pd.DataFrame(columns=[
                    "entry_time","exit_time","id","side",
                    "entry_px","exit_px","pnl_pct","pnl_abs","mfe_pct","exit_reason"
                ])

            df = pd.DataFrame(rows, columns=[
                "entry_time","exit_time","id","side",
                "entry_px","exit_px","pnl_pct","pnl_abs","mfe_pct","exit_reason"
            ])
            return df
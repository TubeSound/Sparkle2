import numpy as np
from technical import calc_sma, calc_ema, ATRP, slopes, detect_pivots, is_nans, slice_upper_abs, detect_perfect_order
from common import Columns, Indicators
import pandas as pd

class CypressParam:
    long_term = 50
    mid_term = 25
    short_term = 13
    trend_slope_th = 0.2
    atr_term = 20
    sl = 20
    tp = 20

    def to_dict(self):
        dic = {
                'long_term': self.long_term,
                'mid_term': self.mid_term,
                'short_term': self.short_term,
                'trend_slope_th': self.trend_slope_th,
                'atr_term': self.atr_term,
                'sl': self.sl,
                'tp': self.tp
               }
        return dic

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
        
        self.trend = self.detect_trend(long_slope, self.param.trend_slope_th)
        
        self.entries, self.exits = self.detect_signals()
    
    def detect_signals(self):
        n = len(self.cl)
        entries = np.full(n, 0)
        exits = np.full(n, 0)
        state = 0
        last_price = 0
        for i in range(1, n):
            if self.trend[i] == -1:
                if self.cl[i - 1] <= self.ema_short[i - 1] and self.cl[i] >= self.ema_short[i]:
                    # Short
                    if state == 0:
                        entries[i] = self.SHORT
                        last_price = self.cl[i]
                        state = self.SHORT
                    elif state == self.SHORT:                    
                        if last_price >= self.cl[i]: 
                            entries[i] = self.SHORT
                            last_price = self.cl[i]
                            state = self.SHORT 
                        else:
                            exits[i] = self.CLOSE
                            state = 0
                    elif state == self.LONG:
                        entries[i] = self.SHORT
                        exits[i] = self.CLOSE
                        last_price = self.cl[i]
                        state = self.SHORT                           
            elif self.trend[i] == 1:
                if self.cl[i - 1] >= self.ema_short[i - 1] and self.cl[i] <= self.ema_short[i]:
                    # Long
                    if state == 0:
                        entries[i] = self.LONG
                        last_price = self.cl[i]
                        state = self.LONG
                    elif state == self.LONG:
                        if last_price <= self.cl[i]:
                            entries[i] = self.LONG
                            last_price = self.cl[i]
                            state = self.LONG
                        else:
                            exits[i] = self.CLOSE
                            state = 0
                    elif state == self.SHORT:
                        entries[i] = self.LONG
                        exits[i] = self.CLOSE
                        last_price = self.cl[i]
                        state = self.LONG
        return entries, exits

                                      
    
    # ── 汎用：シンボル→pip_size 解決（必要なら外部から渡してもOK）
    def resolve_pip_size(self, symbol: str) -> float:
        """
        代表例:
        - USDJPY, EURJPY など: 0.01
        - XAUUSD: 0.1  (ブローカー定義に合わせて調整)
        - 日経225/ダウ/ナスダックなど指数CFD: 1.0（=1ポイント）
        """
        s = symbol.upper()
        if s.endswith("JPY"): return 0.01
        if s in ("XAUUSD","GOLD","XAU"): return 0.1
        if s in ("NIKKEI","JP225","US30","US100","DOW","NSDQ","NAS100"): return 1.0
        # FXの多く: 小数第4位 ＝ 0.0001
        return 0.0001


    def _append_row(self, rows, ts, entry_i, i, trade_id, side, price, fees_pips, pip_size, reason):
        diff = (price[i] - price[entry_i]) * (1 if side > 0 else -1)
        pnl_pips = (diff / pip_size) - fees_pips
        rows.append((
            ts[entry_i], ts[i], trade_id, side,
            float(price[entry_i]), float(price[i]),
            float(pnl_pips), reason
        ))


    def simulate_scalping_pips_multi(
        self,
        priority: str = "TP_FIRST",        # "SL_FIRST" or "TP_FIRST"
        check_from_next_bar: bool = True,  # True: エントリーバーは判定しない
    ):
        """
        スキャルピング（pips≒価格差固定・複数ポジション可）
        - entries の 1/-1 を見るたびに新規建て
        - 各ポジション独立に High/Low 到達で TP/SL
        - 両到達の同時判定は priority で制御（既定: SL優先）
        - 価格は“ポイント差”で扱う（必要なら呼び出し側で pip_size を掛けて渡す）
        """
        cl = np.asarray(self.cl, dtype=float)
        hi = np.asarray(self.hi, dtype=float)
        lo = np.asarray(self.lo, dtype=float)
        sig = np.asarray(self.entries, dtype=int)
        ts  = np.asarray(self.timestamp)

        rows = []
        positions = []  # {id, side, entry_i}
        tid = 1
        n = len(cl)

        for i in range(n):
            to_close = []

            # 1) 既存ポジションの TP/SL 判定（High/Low 使用）
            for pos in positions:
                i0   = pos["entry_i"]
                side = pos["side"]
                ep   = cl[i0]

                # ルックアヘッド回避：次バー以降のみ判定
                if check_from_next_bar and i <= i0:
                    continue

                if side == self.LONG:
                    tp = ep + self.param.tp
                    sl = ep - self.param.sl
                    hit_tp = (hi[i] >= tp)
                    hit_sl = (lo[i] <= sl)
                    reason = None
                    ex_px  = None

                    if hit_tp and hit_sl:
                        if priority == "TP_FIRST":
                            reason, ex_px = "TP", tp
                        else:  # "SL_FIRST"
                            reason, ex_px = "SL", sl
                    elif hit_tp:
                        reason, ex_px = "TP", tp
                    elif hit_sl:
                        reason, ex_px = "SL", sl

                    if reason is not None:
                        pnl = (ex_px - ep)  # long
                        rows.append((ts[i0], ts[i], pos["id"], side, float(ep), float(ex_px), float(pnl), reason))
                        to_close.append(pos)

                else:  # SHORT
                    tp = ep - self.param.tp
                    sl = ep + self.param.sl
                    hit_tp = (lo[i] <= tp)
                    hit_sl = (hi[i] >= sl)
                    reason = None
                    ex_px  = None

                    if hit_tp and hit_sl:
                        if priority == "TP_FIRST":
                            reason, ex_px = "TP", tp
                        else:
                            reason, ex_px = "SL", sl
                    elif hit_tp:
                        reason, ex_px = "TP", tp
                    elif hit_sl:
                        reason, ex_px = "SL", sl

                    if reason is not None:
                        pnl = (ep - ex_px)  # short
                        rows + [ts[i0], ts[i], pos["id"], side, float(ep), float(ex_px), float(pnl), reason]
                        to_close.append(pos)

            if to_close:
                positions = [p for p in positions if p not in to_close]

            # 2) 新規エントリー（複数可）
            if sig[i] == self.LONG or sig[i] == self.SHORT:
                positions.append({"id": tid, "side": sig[i], "entry_i": i})
                tid += 1

        return rows, ["time_entry","time_exit","id","side","price_entry","price_exit","profit","reason"]

    def simulate_doten_pips(
        self,
        sl_pips: float = 15.0,
        pip_size: float = 0.01,
        fees_pips: float = 0.0,
    ):
        """
        ドテン（常に1方向のみ保持）
        - 反対シグナルで“必ず”クローズ→即ドテン
        - 損切りは pips 基準
        """
        price = np.asarray(self.cl, dtype=float)
        sig   = np.asarray(self.entries, dtype=int)  # entriesを“エントリー信号”として扱う
        ts    = np.asarray(self.timestamp)

        rows = []
        side = 0
        entry_i = None
        trade_id = 1

        n = len(price)
        for i in range(n):
            s = sig[i]

            # 建玉ありなら SL 判定
            if side != 0:
                diff = (price[i] - price[entry_i]) * (1 if side > 0 else -1)
                diff_pips = diff / pip_size
                if diff_pips <= -sl_pips:
                    self._append_row(rows, ts, entry_i, i, trade_id, side, price, fees_pips, pip_size, "SL")
                    side = 0
                    trade_id += 1

            # 反対信号でドテン
            if (side == self.LONG and s == self.SHORT) or (side == self.SHORT and s == self.LONG):
                # まずクローズ
                self._append_row(rows, ts, entry_i, i, trade_id, side, price, fees_pips, pip_size, "REVERSE")
                trade_id += 1
                # 即ドテン
                side = s
                entry_i = i
                continue

            # 0→新規
            if side == 0 and (s == self.LONG or s == self.SHORT):
                side = s
                entry_i = i

        return pd.DataFrame(rows, columns=[
            "entry_time","exit_time","id","side","entry_px","exit_px","pnl_pips","exit_reason"
        ])

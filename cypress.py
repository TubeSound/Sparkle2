import numpy as np
import pandas as pd
import MetaTrader5 as mt5api

from technical import calc_sma, calc_ema, ATRP, slopes, detect_pivots, is_nans, slice_upper_abs, detect_perfect_order
from common import Columns, Indicators
from trade_manager import TradeManager, Signal, PositionInfo

class CypressParam:
    long_term = 50
    mid_term = 25
    short_term = 13
    trend_slope_th = 0.2
    atr_term = 20
    sl = 20
    tp = 20
    trade_max = 10
    volume = 0.01

    def to_dict(self):
        dic = {
                'long_term': self.long_term,
                'mid_term': self.mid_term,
                'short_term': self.short_term,
                'trend_slope_th': self.trend_slope_th,
                'atr_term': self.atr_term,
                'sl': self.sl,
                'tp': self.tp,
                'trade_max': self.trade_max,
                'volume': self.volume
               }
        return dic

class Cypress:
    def __init__(self, symbol, param: CypressParam):
        self.symbol = symbol
        self.param = param
        
        
    def result_df(self):
        dic = {
                'jst': self.timestamp,
                'open': self.op, 
                'high': self.hi,
                'low': self.lo,
                'close': self.cl, 
                'entry_signal': self.entries,
                'exit_signal': self.exits,
                'trend': self.trend,
                'atrp': self.atrp,
                'mid_slope': self.mid_slope,
                'long_slope': self.short_slope,
                'long': self.buy_signal, 
                'short': self.sell_signal,
                'exit': self.exit_signal, 
                'reason': self.reason,
                'profit': self.profits
                } 
        
        df = pd.DataFrame(dic)
        return df
        
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
                        entries[i] = Signal.SHORT
                        last_price = self.cl[i]
                        state = Signal.SHORT
                    elif state == Signal.SHORT:                    
                        if last_price >= self.cl[i]: 
                            entries[i] = Signal.SHORT
                            last_price = self.cl[i]
                            state = Signal.SHORT 
                        else:
                            exits[i] = Signal.CLOSE
                            state = 0
                    elif state == Signal.LONG:
                        entries[i] = Signal.SHORT
                        exits[i] = Signal.CLOSE
                        last_price = self.cl[i]
                        state = Signal.SHORT                           
            elif self.trend[i] == 1:
                if self.cl[i - 1] >= self.ema_short[i - 1] and self.cl[i] <= self.ema_short[i]:
                    # Long
                    if state == 0:
                        entries[i] = Signal.LONG
                        last_price = self.cl[i]
                        state = Signal.LONG
                    elif state == Signal.LONG:
                        if last_price <= self.cl[i]:
                            entries[i] = Signal.LONG
                            last_price = self.cl[i]
                            state = Signal.LONG
                        else:
                            exits[i] = Signal.CLOSE
                            state = 0
                    elif state == Signal.SHORT:
                        entries[i] = Signal.LONG
                        exits[i] = Signal.CLOSE
                        last_price = self.cl[i]
                        state = Signal.LONG
        return entries, exits

                                      
    

    def simulate_scalping(self, priory: str = "SL_FIRST"):
        def cleanup(i, h, l):
            close_tickets = []
            for ticket, position in manager.positions.items():
                if position.is_sl(l, h):
                    position.profit = - position.sl
                    position.exit_time = jst[i]
                    position.exit_price = position.sl_price
                    position.reason = PositionInfo.STOP_LOSS
                    close_tickets.append(ticket)
                    reason[i] = PositionInfo.STOP_LOSS
                    exit_signal[i] = position.ticket
                    profits[i] = - position.sl
                elif position.is_tp(l, h):
                    position.profit = position.tp 
                    position.exit_time = jst[i]
                    position.exit_price = position.tp_price
                    position.reason = PositionInfo.TAKE_PROFIT
                    reason[i] = PositionInfo.TAKE_PROFIT
                    exit_signal[i] = ticket
                    profits[i] = position.tp
                    close_tickets.append(ticket)
            manager.remove_positions(close_tickets)
            
        n = len(self.cl)
        jst = self.timestamp
        manager = TradeManager(self.symbol, 'M1')    
        ticket = 1
        buy_signal = np.full(n, 0)
        sell_signal = np.full(n, 0)
        exit_signal = np.full(n, 0)
        reason = np.full(n, 0)
        profits = np.full(n, 0)
        for i in range(n):
            cleanup(i, self.hi[i], self.lo[i])
            entry = self.entries[i]    
            if entry == 0:
                continue
            elif entry == Signal.LONG:
                typ =  mt5api.ORDER_TYPE_BUY_STOP_LIMIT
                buy_signal[i] = ticket
            elif entry == Signal.SHORT:
                typ =  mt5api.ORDER_TYPE_SELL_STOP_LIMIT
                sell_signal[i] = ticket
            pos = PositionInfo(self.symbol, typ, jst[i], self.param.volume, ticket, self.cl[i], self.param.sl, self.param.tp)
            manager.add_position(pos)
            ticket += 1
            
        close_tickets = []
        for ticket, position in manager.positions.items():
            if position.order_signal == Signal.LONG:
                position.profit = self.cl[-1] - position.entry_price
                position.exit_time = jst[-1]
                position.exit_price = self.cl[-1]
                position.reason = PositionInfo.TIMEUP
                exit_signal[-1] = ticket
                reason[-1] = PositionInfo.TIMEUP
                profits[-1] = position.profit
                close_tickets.append(ticket)
            elif position.order_signal == Signal.SHORT:
                position.profit = position.entry_price - self.cl[-1] 
                position.exit_time = jst[-1]
                position.exit_price = self.cl[i]
                position.reason = PositionInfo.TIMEUP
                exit_signal[-1] = ticket
                reason[-1] = PositionInfo.TIMEUP
                profits[-1] = position.profit
                close_tickets.append(ticket)
        manager.remove_positions(close_tickets)    

        self.buy_signal = buy_signal
        self.sell_signal = sell_signal
        self.exit_signal = exit_signal
        self.reason = reason
        self.profits = profits
        return manager



    def simulate_scalping_pips_multi(
        self,
        priority: str = "SL_FIRST",        # "SL_FIRST" or "TP_FIRST"
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
        buy = np.full(n, 0)
        sell = np.full(n, 0)
        cls = np.full(n, 0)
        profits = np.full(n, 0)

        for i in range(n):
            to_close = []

            # 1) 既存ポジションの TP/SL 判定（High/Low 使用）
            for pos in positions:
                i0   = pos["entry_i"]
                side = pos["side"]
                price0   = cl[i0]

                # ルックアヘッド回避：次バー以降のみ判定
                if check_from_next_bar and i <= i0:
                    continue
                
                # Timeup
                if i == n - 1:
                    reason = self.TIMEUP
                    if side == Signal.LONG:
                        profit = cl[i] - price0
                        sell[i] = pos['id']        
                    elif side == Signal.SHORT:
                        profit = price0 - cl[i]
                        buy[i] = pos['id']
                    else:
                        break
                    rows.append([ts[i0], ts[i], pos["id"], side, float(price0), float(cl[i]), profit, self.close_reason[reason]])                    
                    break

                if side == Signal.LONG:
                    tp = price0 + self.param.tp
                    sl = price0 - self.param.sl
                    hit_tp = (hi[i] >= tp)
                    hit_sl = (lo[i] <= sl)
                    reason = None
                    price1 = None
                    profit = None

                    if hit_tp and hit_sl:
                        if priority == "TP_FIRST":
                            reason = self.TAKE_PROFIT
                            profit = self.param.tp
                            price1 = tp
                        else:  # "SL_FIRST"
                            reason = self.STOP_LOSS
                            profit = - self.param.sl
                            price1 = sl
                    elif hit_tp:
                        reason = self.TAKE_PROFIT
                        profit = self.param.tp
                        price1 = tp
                    elif hit_sl:
                        reason = self.STOP_LOSS
                        profit = - self.param.sl
                        price1 = sl

                    if reason is not None:
                        rows.append((ts[i0], ts[i], pos["id"], side, float(price0), float(price1), float(profit), self.close_reason[reason]))
                        sell[i] = pos['id']
                        cls[i] = reason
                        profits[i] = profit
                        to_close.append(pos)

                else:  # SHORT
                    tp = price0 - self.param.tp
                    sl = price0 + self.param.sl
                    hit_tp = (lo[i] <= tp)
                    hit_sl = (hi[i] >= sl)
                    reason = None
                    profit = None
                    price1  = None

                    if hit_tp and hit_sl:
                        if priority == "TP_FIRST":
                            reason = self.TAKE_PROFIT
                            price1 = tp
                            profit = self.param.tp
                        else:
                            reason = self.STOP_LOSS
                            price1 = sl
                            profit = - self.param.sl
                    elif hit_tp:
                        reason = self.TAKE_PROFIT
                        price1 = tp
                        profit = self.param.tp
                    elif hit_sl:
                        reason = self.STOP_LOSS
                        profit = - sl
                        price1 = sl
                    if reason is not None:
                        rows.append([ts[i0], ts[i], pos["id"], side, float(price0), float(price1), float(profit), self.close_reason[reason]])
                        buy[i] = pos['id']
                        cls[i] = reason
                        profits[i] = profit
                        to_close.append(pos)

            if to_close:
                positions = [p for p in positions if p not in to_close]
                    
            
            # 2) 新規エントリー（複数可）
            if sig[i] == Signal.LONG or sig[i] == Signal.SHORT:
                positions.append({"id": tid, "side": sig[i], "entry_i": i})
                if sig[i] == Signal.LONG:
                    buy[i] = tid
                elif sig[i] == Signal.SHORT:
                    sell[i] = tid                
                tid += 1
                
        self.trade_buy = buy
        self.trade_sell = sell
        self.trade_close_reason = cls
        self.trade_profits = profits
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
            if (side == Signal.LONG and s == Signal.SHORT) or (side == Signal.SHORT and s == Signal.LONG):
                # まずクローズ
                self._append_row(rows, ts, entry_i, i, trade_id, side, price, fees_pips, pip_size, "REVERSE")
                trade_id += 1
                # 即ドテン
                side = s
                entry_i = i
                continue

            # 0→新規
            if side == 0 and (s == Signal.LONG or s == Signal.SHORT):
                side = s
                entry_i = i

        return pd.DataFrame(rows, columns=[
            "entry_time","exit_time","id","side","entry_px","exit_px","pnl_pips","exit_reason"
        ])

import numpy as np
import pandas as pd
import MetaTrader5 as mt5api

from technical import calc_sma, calc_ema, calc_atr, is_nans, trend_heikin
from common import Columns, Indicators
from trade_manager import TradeManager, Signal, PositionInfo

class MaronParam:
    ma_term = 13
    ma_method = 'sma'
    atr_term = 14
    atr_shift_multiply = 2.0
    trend_threshold = 0.05
    upper_timeframe = 10
    sl = 20
    tp = 20
    sl_loose = None
    position_max = 20
    volume = 0.01

    def to_dict(self):
        dic = {
                'ma_term': self.ma_term,
                'ma_method': self.ma_method,
                'atr_term': self.atr_term,
                'atr_shift_multiply': self.atr_shift_multiply,
                'trend_threshold': self.trend_threshold,
                'upper_timeframe': self.upper_timeframe,
                'sl': self.sl,
                'tp': self.tp,
                'sl_loose': self.sl_loose,
                'position_max': self.position_max,
                'volume': self.volume
               }
        return dic
    
    @staticmethod
    def load_from_dic(dic: dict):
        param = MaronParam()
        param.ma_term = int(dic['ma_term']) 
        param.ma_method = dic['ma_method']          
        param.atr_term = int(dic['atr_term'])
        param.atr_shift_multiply = float(dic['atr_shift_multiply'])
        param.trend_threshold = float(dic['trend_threshold'])
        param.upper_timeframe = int(dic['upper_timeframe'])
        param.sl = float(dic['sl'])
        param.tp = float(dic['tp'])
        if dic['sl_loose'] == None:
            param.sl_loose = None
        else:
            param.sl_loose = float(dic['sl_loose'])
        param.position_max = int(dic['position_max'])
        param.volume = int(dic['volume'])
        return param    
        
class Maron:
    def __init__(self, symbol, param: MaronParam):
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
                'atr': self.atr,
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
            
    def mask_with_trend(self, signal, trend):
        n = len(signal)
        out = np.full(n, 0)
        for i in range(n):
            if signal[i] == trend[i]:
                out[i] = signal[i]
        return out

        
    def calc(self, df):
        self.timestamp = df['jst'].tolist()
        self.op = df[Columns.OPEN].to_numpy()
        self.hi = df[Columns.HIGH].to_numpy()
        self.lo = df[Columns.LOW].to_numpy()
        self.cl = df[Columns.CLOSE].to_numpy()
        if self.param.ma_method.lower() == 'sma':
            self.ma = calc_sma(self.cl, self.param.ma_term)
        else:
            self.ma = calc_ema(self.cl, self.param.ma_term)
        self.atr = calc_atr(self.hi, self.lo, self.cl, self.param.atr_term, how=self.param.ma_method)
        self.ma_upper = self.ma + self.atr * self.param.atr_shift_multiply
        self.ma_lower = self.ma - self.atr * self.param.atr_shift_multiply
        self.trend, _ = trend_heikin(df, self.param.upper_timeframe, self.param.trend_threshold)
        self.entries, self.exits = self.detect_signals()
    
    def detect_signals(self):
        n = len(self.cl)
        entries = np.full(n, 0)
        exits = np.full(n, 0)
        state = 0
        last_price = 0
        for i in range(1, n):
            if self.trend[i] == -1:
                if self.cl[i - 1] <= self.ma_upper[i - 1] and self.cl[i] > self.ma_upper[i]:
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
                if self.cl[i - 1] >= self.ma_lower[i - 1] and self.cl[i] < self.ma_lower[i]:
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

                                    
    def simulate_scalping(self, tbegin, tend, priory: str = "SL_FIRST"):
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
            if jst[i] < tbegin or jst[i] >= tend:
                continue
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
                
        return manager.summary()



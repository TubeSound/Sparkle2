import numpy as np
import pandas as pd
import MetaTrader5 as mt5api

from technical import calc_sma, calc_ema, calc_atr, is_nans, trend_heikin, super_trend
from common import Columns, Indicators
from trade_manager import TradeManager, Signal, PositionInfo

class MaronPieParam:
    ma_term = 14
    ma_method = 'ema'
    atr_term = 6
    atr_shift_multiply = 0.6
    supertrend_atr_term = 20
    supertrend_minutes = 5
    supertrend_multiply = 2.0
    heikin_minutes = 60
    heikin_threshold = 0.04
    sl = 0.5
    sl_loose = None
    position_max = 20
    volume = 0.01

    def to_dict(self):
        dic = {
                'ma_term': self.ma_term,
                'ma_method': self.ma_method,
                'atr_term': self.atr_term,
                'atr_shift_multiply': self.atr_shift_multiply,
                'supertrend_atr_term': self.supertrend_atr_term, 
                'supertrend_minutes': self.supertrend_minutes,
                'supertrend_multiply':self.supertrend_multiply,
                'heikin_minutes': self.heikin_minutes,
                'heikin_threshold': self.heikin_threshold,
                'sl': self.sl,
                'sl_loose': self.sl_loose,
                'position_max': self.position_max,
                'volume': self.volume
               }
        return dic
    
    @staticmethod
    def load_from_dic(dic: dict):
        param = MaronPieParam()
        param.ma_term = int(dic['ma_term']) 
        param.ma_method = dic['ma_method']          
        param.atr_term = int(dic['atr_term'])
        param.atr_shift_multiply = float(dic['atr_shift_multiply'])
        param.supertrend_atr_term = int(dic['supertrend_atr_term'])
        param.supertrend_minutes = int(dic['supertrend_minutes'])
        param.supertrend_multiply = float(dic['supertrend_multiply'])
        param.heikin_minutes = int(dic['heikin_minutes'])
        param.heikin_threshold = float(dic['heikin_threshold'])
        param.heikin_minutes = int(dic['heikin_minutes'])
        param.sl = float(dic['sl'])
        if dic['sl_loose'] == None:
            param.sl_loose = None
        else:
            param.sl_loose = float(dic['sl_loose'])
        param.position_max = int(dic['position_max'])
        param.volume = float(dic['volume'])
        return param    
        
class MaronPie:
    def __init__(self, symbol, param: MaronPieParam):
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
                'reversal': self.reversal,
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
        
        trend, reversal, upper_line, lower_line = super_trend(df, self.param.supertrend_minutes, self.param.supertrend_atr_term, self.param.supertrend_multiply)
        self.supertrend_upper = upper_line
        self.supertrend_lower = lower_line
        _, no_trend = trend_heikin(df, self.param.heikin_minutes, self.param.heikin_threshold)
        self.trend = trend
        self.no_trend = no_trend
        self.reversal = reversal
        self.entries, self.exits = self.detect_signals()
    
    def detect_signals(self):
        n = len(self.cl)
        entries = np.full(n, 0)
        exits = np.full(n, 0)
        current = 0
        for i in range(1, n):
            if current == 1 and self.reversal[i] == -1:
                exits[i] = Signal.CLOSE
            elif current == -1 and self.reversal[i] == 1:
                exits[i] = Signal.CLOSE
            if self.no_trend[i] == 1:
                continue
            if self.trend[i] == -1:
                if self.cl[i - 1] <= self.ma_upper[i - 1] and self.cl[i] > self.ma_upper[i]:
                    entries[i] = Signal.SHORT
                    current = -1                        
            elif self.trend[i] == 1:
                if self.cl[i - 1] >= self.ma_lower[i - 1] and self.cl[i] < self.ma_lower[i]:
                    entries[i] = Signal.LONG
                    current = 1   
        return entries, exits
     
    def simulate_doten(self, tbegin, tend):
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
            manager.remove_positions(close_tickets)
            
        def close_all(time, price, reason):
            close_tickets = []
            for ticket, position in manager.positions.items():
                position.exit_price = price
                position.exit_time = time
                position.reason = reason
                if position.order_signal == Signal.LONG:
                    position.profit = price - position.entry_price
                else:
                    position.profit = position.entry_price - price                
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
        profits = {'jst': jst, 'total_profit': np.full(n, 0.0), 'current_profit': np.full(n, 0.0), 'closed_profit': np.full(n, 0.0), 'trade_count': np.full(n, 0), 'win_rate': np.full(n, 0.0)}
        for i in range(n):
            if jst[i] < tbegin or jst[i] >= tend:
                continue
            
            total, current, closed, count, win_rate = manager.calc_profit(self.cl[i])
            profits['total_profit'][i] = total
            profits['current_profit'][i] = current
            profits['closed_profit'][i] = closed
            profits['trade_count'][i] = count
            profits['win_rate'][i] = win_rate
            if self.exits[i] == Signal.CLOSE:
                # doten
                close_all(jst[i], self.cl[i], PositionInfo.REVERSAL)
            else:
                # loss cut
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
            pos = PositionInfo(self.symbol, typ, jst[i], self.param.volume, ticket, self.cl[i], self.param.sl, 0)
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
                close_tickets.append(ticket)
            elif position.order_signal == Signal.SHORT:
                position.profit = position.entry_price - self.cl[-1] 
                position.exit_time = jst[-1]
                position.exit_price = self.cl[i]
                position.reason = PositionInfo.TIMEUP
                exit_signal[-1] = ticket
                reason[-1] = PositionInfo.TIMEUP
                close_tickets.append(ticket)
        manager.remove_positions(close_tickets)    

        self.buy_signal = buy_signal
        self.sell_signal = sell_signal
        self.exit_signal = exit_signal
        self.reason = reason
        self.profits = profits
        df_profits = pd.DataFrame(profits)
        return manager.summary(), df_profits



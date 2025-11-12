import numpy as np
import pandas as pd
import MetaTrader5 as mt5api

from technical import calc_sma, calc_ema, calc_atr, is_nans, trend_heikin, super_trend
from common import Columns, Indicators
from trade_manager import TradeManager, Signal, PositionInfo

class MontblancParam:
    atr_term = 10
    trend_minutes = 5
    trend_multiply = 2.0
    trend_micro_minutes = 1
    trend_micro_multiply = 2.0 
    sl = 0.5
    sl_loose = None
    position_max = 5
    volume = 0.01

    def to_dict(self):
        dic = {
                'atr_term': self.atr_term,
                'trend_minutes': self.trend_minutes,
                'trend_multiply': self.trend_multiply,
                'trend_micro_minutes': self.trend_micro_minutes,
                'trend_micro_multiply': self.trend_micro_multiply,
                'sl': self.sl,
                'sl_loose': self.sl_loose,
                'position_max': self.position_max,
                'volume': self.volume
               }
        return dic
    
    @staticmethod
    def load_from_dic(dic: dict):
        param = MontblancParam()   
        param.atr_term = int(dic['atr_term'])
        param.trend_minutes = int(dic['trend_minutes'])
        param.trend_multiply = float(dic['trend_multiply'])
        param.trend_micro_minutes = int(dic['trend_micro_minutes'])
        param.trend_micro_multiply = float(dic['trend_micro_multiply'])
        param.sl = float(dic['sl'])
        if dic['sl_loose'] == None:
            param.sl_loose = None
        else:
            param.sl_loose = float(dic['sl_loose'])
        param.position_max = int(dic['position_max'])
        param.volume = float(dic['volume'])
        return param    
        
class Montblanc:
    def __init__(self, symbol, param: MontblancParam):
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
        self.atr = calc_atr(self.hi, self.lo, self.cl, self.param.atr_term)
        trend, reversal, upper_line, lower_line, counts = super_trend(df, self.param.trend_minutes, self.param.atr_term, self.param.trend_multiply)
        trend_micro, reversal_micro, ul, ll, counts_micro = super_trend(df, self.param.trend_micro_minutes, self.param.atr_term, self.param.trend_micro_multiply)
        self.upper_line = upper_line
        self.lower_line = lower_line
        self.micro_upper_line = ul
        self.micro_lower_line = ll
        self.trend = trend
        self.update_counts = counts
        self.reversal = reversal
        self.trend_micro = trend_micro
        self.reversal_micro = reversal_micro
        self.entries, self.exits = self.detect_signals()
    
    def detect_signals(self):
        n = len(self.cl)
        entries = np.full(n, 0)
        exits = np.full(n, 0)
        for i in range(1, n):
            # trendが変化したときにクローズ
            if self.reversal[i] != 0:
                exits[i] = self.reversal[i]
            # microトレンドが変化したところでエントリー
            if self.trend[i] == 1 and self.reversal_micro[i] == 1:
                entries[i] = Signal.LONG
            elif self.trend[i] == -1 and self.reversal_micro[i] == -1:
                entries[i] = Signal.SHORT
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
            if self.exits[i] != 0:
                # doten
                close_all(jst[i], self.cl[i], PositionInfo.REVERSAL)
            else:
                # loss cut
                cleanup(i, self.hi[i], self.lo[i])
            entry = self.entries[i]    
            if entry == 0 or (len(manager.open_positions()) > self.param.position_max):
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




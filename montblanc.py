import numpy as np
import pandas as pd
import MetaTrader5 as mt5api

from technical import calc_sma, calc_ema, calc_atr, is_nans, trend_heikin, super_trend, ema_reversal, slope_trend
from common import Columns, Indicators
from trade_manager import TradeManager, Param, Signal, PositionInfo

class MontblancParam(Param):
    reversal_mode = 'reversal_major' # 'slope' / 'reversal_major' / 'reversal_minor' 
    ema_term_entry = 12
    filter_term_exit = 48
    atr_term = 14
    trend_major_minutes = 60
    trend_major_multiply = 2.0
    trend_minutes = 15
    trend_multiply = 2.0

    def to_dict(self):
        dic = {
                'sl_mode': self.sl_mode,
                'stop_cond': self.stop_cond,
                'sl_value': self.sl_value,
                'reversal_mode': self.reversal_mode,
                'ema_term_entry': self.ema_term_entry,
                'filter_term_exit': self.filter_term_exit,
                'atr_term': self.atr_term,
                'trend_major_minutes': self.trend_major_minutes,
                'trend_major_multiply': self.trend_major_multiply,
                'trend_minutes': self.trend_minutes,
                'trend_multiply': self.trend_multiply,
                'position_max': self.position_max,
                'volume': self.volume
               }
        return dic
    
    @staticmethod
    def load_from_dic(dic: dict):
        param = MontblancParam()   
        param.sl_mode = dic['sl_mode'].lower()
        param.stop_cond = dic['stop_cond'].lower()
        param.sl_value = float(dic['sl_value'])
        param.reversal_mode = dic['reversal_mode'].lower()
        param.ema_term_entry = int(dic['ema_term_entry'])
        param.filter_term_exit = int(dic['filter_term_exit'])
        param.atr_term = int(dic['atr_term'])
        param.trend_major_minutes = int(dic['trend_major_minutes'])
        param.trend_major_multiply = float(dic['trend_major_multiply'])
        param.trend_minutes = int(dic['trend_minutes'])
        param.trend_multiply = float(dic['trend_multiply'])
        param.position_max = int(dic['position_max'])
        param.volume = float(dic['volume'])
        return param    
        
class Montblanc:
    def __init__(self, symbol, param: MontblancParam):
        self.symbol = symbol
        self.param = param
           
    def result_df(self):
        dic = {
                'jst': self.jst,
                'open': self.op, 
                'high': self.hi,
                'low': self.lo,
                'close': self.cl, 
                'entries': self.entries,
                'exits': self.exits,
                'trend': self.trend,
                'trend_minor': self.trend_minor,
                'trend_major': self.trend_major,
                'trend_micro': self.trend_micro,
                'reversal_major': self.reversal_major,
                'reversal_minor': self.reversal_minor,
                'reversal_micro': self.reversal_micro,
                'atr': self.atr,
                'upper_minor': self.upper_minor,
                'lower_minor': self.lower_minor,
                'upper_major': self.upper_major,
                'lower_major': self.lower_major,
                'ema_entry': self.ema_entry,
                'slope_exit': self.slope_exit,
                } 
        
        df = pd.DataFrame(dic)
        return df   
        
    def count_value(self, array, value):
        n = 0
        for v in array:
            if v == value:
                n += 1
        return n                 
        
    def calc_ema(self, cl, term, trend):
        def search_term(trend, j):
            begin = None
            for i in range(j, n):
                if begin == None:
                    if trend[i] == 1 or trend[i] == -1:
                        begin = i
                        status = trend[i]
                else:
                    if trend[i] != status:
                        end = i - 1   
                        return (begin, end), False               
            if begin is not None:
                end = len(trend) - 1
                return (begin, len(trend) - 1), True
            else:
                return (None, None), True
                  
        n = len(cl)
        ma = np.full(n, np.nan)
        i = 0
        while True :
            (begin, end), finish = search_term(trend, i)
            if begin is None:
                break
            d = cl[begin: end + 1]
            ma[begin : end + 1] = calc_ema(d, term)
            if finish:
                break
            i = end + 1
        return ma
        
    def mask(self, signal, trend):
        out = signal.copy()
        n = len(signal)
        for i in range(n):
            if trend[i] != 1 and trend[i] != -1:
                out[i] = np.nan
        return out

    def calc(self, df):
        self.jst = df['jst'].tolist()
        self.op = df[Columns.OPEN].to_numpy()
        self.hi = df[Columns.HIGH].to_numpy()
        self.lo = df[Columns.LOW].to_numpy()
        self.cl = df[Columns.CLOSE].to_numpy()
        self.atr = calc_atr(self.hi, self.lo, self.cl, self.param.atr_term)
        # major trend
        self.trend_major, self.reversal_major, um, lm, _ = super_trend(df, self.param.trend_major_minutes, self.param.atr_term, self.param.trend_major_multiply)
        self.upper_major = um
        self.lower_major = lm
        
        # minor trend
        self.trend_minor, self.reversal_minor, upper_minor, lower_minor, counts = super_trend(df, self.param.trend_minutes, self.param.atr_term, self.param.trend_multiply)
        self.upper_minor = upper_minor
        self.lower_minor = lower_minor
        self.update_counts = counts
        
        # micro trend
        self.reversal_micro, self.ema_entry = ema_reversal(df, self.param.ema_term_entry)
        self.trend_micro, slope = slope_trend(self.ema_entry, 5) 
        
        # total trend
        self.trend = self.make_trend([self.trend_major, self.trend_minor, self.trend_micro])
                
        # entry
        self.entries = self.make_entry(self.trend, self.reversal_micro)
 
        # exit
        self.trend_exit, self.slope_exit = slope_trend(self.ema_entry, self.param.filter_term_exit)
        self.exits = self.make_exit()
        
            
    def make_trend(self, trends):
        def is_match(vector):
            for i in range(1, len(vector)):
                if vector[0] != vector[i]:
                    return False
            return True
        n = len(trends[0])
        trend = np.full(n, 0)
        for i in range(n):
            d = [t[i] for t in trends]
            if is_match(d):
                trend[i] = d[0]
        return trend
    
    def make_entry(self, trend, reversal_micro):
        n = len(self.cl)
        entries = np.full(n, 0)
        for i in range(n):
            if reversal_micro[i] == trend[i]:
                entries[i] = trend[i]
        return entries
    
    def make_exit(self):
        slope = self.slope_exit
        n = len(slope)
        exits = np.full(n, 0)
        if self.param.reversal_mode == 'slope':
            for i in range(1, n):
                if slope[i - 1] >= 0 and slope[i] < 0:
                    exits[i] = -1
                if slope[i - 1] <= 0 and  slope[i] > 0:
                    exits[i] = 1
            return exits
        elif self.param.reversal_mode == 'reversal_major':
            return self.reversal_major
        elif self.param.reversal_mode == 'reversal_minor':
            return self.reversal_minor                
        return exits
        
                        
    def calc_sl(self, i, is_long):
        if self.param.sl_mode == 'fix':
            if is_long:
               return self.cl[i] - self.param.sl_value
            else:
                return self.cl[i] + self.param.sl_value
        elif self.param.sl_mode == 'atr':
            if is_long:
                return self.lower_minor[i] - self.param.sl_value
            else:
                return self.upper_minor[i] + self.param.sl_value
            
                        
    def simulate_doten(self, tbegin, tend):
        manager = TradeManager(self.symbol, self.param)
            
        n = len(self.cl)
        jst = self.jst
        ticket = 1
        buy_signal = np.full(n, 0)
        sell_signal = np.full(n, 0)
        exit_signal = np.full(n, 0)
        reason = np.full(n, 0)
        profits = {'jst': jst, 'total_profit': np.full(n, 0.0), 'current_profit': np.full(n, 0.0), 'closed_profit': np.full(n, 0.0), 'trade_count': np.full(n, 0), 'win_rate': np.full(n, 0.0)}
        time = None
        for i in range(n):
            if jst[i] < tbegin or jst[i] >= tend:
                continue
            time = self.jst[i]
            price = [self.op[i], self.hi[i], self.lo[i], self.cl[i]]
            total, current, closed, count, win_rate = manager.calc_profit_sum()
            profits['total_profit'][i] = total
            profits['current_profit'][i] = current
            profits['closed_profit'][i] = closed
            profits['trade_count'][i] = count
            profits['win_rate'][i] = win_rate
            
            # 既存ポジションの更新
            manager.update(time, price, self.exits[i])
            if self.param.sl_mode == 'atr':
                manager.update_sl(self.upper_minor[i], self.lower_minor[i])
            entry = self.entries[i]    
            if entry == 0:
                continue
            
            num = len(manager.open_positions())
            if num == 5:
                debug = 1
            if num >= self.param.position_max:
                continue
            elif entry == Signal.LONG:
                typ =  mt5api.ORDER_TYPE_BUY_STOP_LIMIT
                buy_signal[i] = ticket
            elif entry == Signal.SHORT:
                typ =  mt5api.ORDER_TYPE_SELL_STOP_LIMIT
                sell_signal[i] = ticket
            sl = self.calc_sl(i, entry == Signal.LONG)
            pos = PositionInfo(self.symbol, typ, jst[i], self.param.volume, ticket, self.cl[i], self.param)
            manager.add_position(pos)
            ticket += 1
       
        # シミュレーション終了時にオープンポジションを強制決済     
        if time is not None:
            manager.timeup(time, price)

        self.buy_signal = buy_signal
        self.sell_signal = sell_signal
        self.exit_signal = exit_signal
        self.reason = reason
        self.profits = profits
        df_profits = pd.DataFrame(profits)
        return manager.summary(), df_profits




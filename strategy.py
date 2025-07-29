import numpy as np 
import pandas as pd
import math
import statistics as stat
from common import Indicators, Signal, Columns, UP, DOWN, HIGH, LOW, HOLD
from datetime import datetime, timedelta
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc') 
from time_utils import TimeFilter
    
def nans(length):
    return [np.nan for _ in range(length)]

def full(value, length):
    return [value for _ in range(length)]

class Position:
    def __init__(self, trade_param, signal: Signal, index: int, time: datetime, price, volume, sl):
        self.id = None
        self.sl = sl
        self.target = trade_param['trail_target']
        self.trail_stop = trade_param['trail_stop']
  
        self.signal = signal
        self.entry_index = index
        self.entry_time = time
        self.entry_price = price
        self.volume = volume
        self.exit_index = None
        self.exit_time = None       
        self.exit_price = None
        self.profit = None
        self.fired = False
        self.profit_max = None
        self.closed = False
        self.losscutted = False
        self.trail_stopped = False
        self.timelimit = False
        self.doten = False
        
    # return  True: Closed,  False: Not Closed
    def update(self, index, time, o, h, l, c, timefilter: TimeFilter):
        if self.sl == 0:
            return False
        
        if timefilter is not None:
            if timefilter.over(time):
                self.exit(index, time, c, timelimit=True)
                return True
        
        # check stoploss
        if self.signal == Signal.LONG:
            if l <= self.sl:
                 self.exit(index, time, self.sl)
                 self.losscutted = True
                 return True
            profit = c - self.entry_price
        else:
            if h >= self.sl:
                self.exit(index, time, self.sl)
                self.losscutted = True
                return True
            profit = c - self.entry_price            
        
        if self.target == 0:
            return False
        
        if self.fired:
            if profit > self.profit_max:
                self.profit_max = profit
            else:
                if self.profit_max - profit < self.trail_stop:
                    self.exit(index, time, c)
                    self.trail_stopped = True         
                    return True
        else:
            if profit >= self.target:
                self.profit_max = profit
                self.fired = True        
        return False
    
    def exit(self, index, time, price, doten=False, timelimit=False):
        self.exit_index = index
        self.exit_time = time
        self.exit_price = price
        self.closed = True
        self.profit = price - self.entry_price
        if self.signal == Signal.SHORT:
            self.profit *= -1
        self.doten=doten
        self.timelimit = timelimit

class Positions:
    
    def __init__(self, timefilter: TimeFilter):
        self.timefilter = timefilter
        self.positions = []
        self.closed_positions = []
        self.current_id = 0
                
    def num(self):
        return len(self.positions) 
    
    def total_volume(self):
        v = 0
        for position in self.positions:
            v += position.volume
        return v
    
    def add(self, position: Position):
        position.id = self.current_id
        self.positions.append(position)
        self.current_id += 1
        
    def update(self, index, time, op, hl, lo, cl):
        closed = []
        for position in self.positions:
            if position.update(index, time, op, hl, lo, cl, self.timefilter):
                closed.append(position.id)
        for id in closed:
            for i, position in enumerate(self.positions):
                if position.id == id:
                    pos = self.positions.pop(i)        
                    self.closed_positions.append(pos)
                    
    def exit_all(self, index, time, price, doten=False):
        for position in self.positions:
            position.exit(index, time, price, doten=doten)            
        self.closed_positions += self.positions
        self.positions = []
    
    def exit_all_signal(self, signal, index, time, price, doten=False):
        removed = []
        remain = []
        for i, position in enumerate(self.positions):
            if index == position.entry_index:
                continue
            if (signal is None) or (position.signal == signal):
                position.exit(index, time, price, doten=doten)         
                removed.append(position)
            else:
                remain.append(position)         
        self.positions = remain
        self.closed_positions += removed
        self.last_entry_price = None
        self.last_entry_signal = None
        
    def summary(self):
        profit_sum = 0
        win = 0
        profits = []
        tacc = []
        acc = []
        time = []
        for positions in [self.closed_positions, self.positions]:
            for position in positions:
                profits.append(position.profit)
                prf = position.profit
                if prf is not None:
                    if not np.isnan(prf):
                        profit_sum += prf
                        if prf > 0:
                            win += 1
                time.append(position.entry_time)
                tacc.append(position.exit_time)
                acc.append(profit_sum)
        
        n = len(self.closed_positions)
        if n > 0:
            win_rate = float(win) / float(n)
        else:
            win_rate = 0
        return [n, profit_sum, win_rate], (tacc, acc)
    
    def to_dataFrame(self, strategy: str):
        def bool2str(v):
            s = 'true' if v else 'false'
            return s
            
        data = []
        count = 0
        for positions in [self.closed_positions, self.positions]:
            for position in positions:
                d = [count, strategy, position.signal, position.entry_index, str(position.entry_time), position.entry_price]
                d += [position.exit_index, str(position.exit_time), position.exit_price, position.sl, position.profit]
                d += [bool2str(position.closed), bool2str(position.losscutted),  bool2str(position.trail_stopped)]
                d += [bool2str(position.doten), bool2str(position.timelimit)]
                data.append(d)
        columns = ['no', 'Mode', 'signal', 'entry_index', 'entry_time', 'entry_price']
        columns += ['exit_index', 'exit_time', 'exit_price', 'sl', 'profit']
        columns += ['closed', 'losscuted', 'trail_stopped', 'doten', 'timelimit']
        df = pd.DataFrame(data=data, columns=columns)
        return df 
    
class Simulation:
    SL_NONE = 0
    SL_FIX = 1
    SL_RANGE = 2
    SL_DATA = 3
    
    def __init__(self, data, trade_param:dict):
        self.data = data
        self.trade_param = trade_param
        self.strategy = self.trade_param['strategy'].upper()
        self.volume = trade_param['volume']
        self.position_num_max = trade_param['position_max']
        self.sl_method = self.trade_param['sl_method']
        self.sl_value = self.trade_param['sl_value']
        self.last_entry_price = None
        self.last_entry_signal = None
        try :
            begin_hour = self.trade_param['begin_hour']
            begin_minute = self.trade_param['begin_minute']
            hours = self.trade_param['hours']
            if hours <= 0 or hours == 24:
                self.timefilter = None
            else:
                self.timefilter = TimeFilter(JST, begin_hour, begin_minute, hours)
        except:
            self.timefilter = None                
        self.positions = Positions(self.timefilter)
        
    def calc_sl(self, signal, index, sl_method, sl_value, data):
        if sl_method == self.SL_NONE:
            return None
        op = data[Columns.OPEN]
        hi = data[Columns.HIGH]
        lo = data[Columns.LOW]
        cl = data[Columns.CLOSE]
        if sl_method == self.SL_FIX:
            [value] = sl_value
            if signal == Signal.LONG:
                sl = cl[index] - value
            elif signal == Signal.SHORT:
                sl = cl[index] + value 
        elif sl_method == self.SL_RANGE:
            [length, value] = sl_value
            i0 = index - length + 1
            if i0 < 0:
                i0 = 0
            if signal == Signal.LONG:
                d = lo[i0: index + 1]
                sl = min(d) - value
            elif signal == Signal.SHORT:
                d = hi[i0: index + 1]
                sl = max(d) + value
        elif sl_method == self.SL_DATA:
            [column] = sl_value
            stop_signal = data[column]
            value = stop_signal[index]
            if signal == Signal.LONG:
                sl = cl[index] - value
            elif signal == Signal.SHORT:
                sl = cl[index] + value             
        return sl
        
    def run(self, data, entry_signal, exit_signal):
        self.data = data
        time = data[Columns.JST]
        op = data[Columns.OPEN]
        hi = data[Columns.HIGH]
        lo = data[Columns.LOW]
        cl = data[Columns.CLOSE]
        n = len(time)
        entry = data[entry_signal]
        ext = data[exit_signal]
    
        for i in range(1, n):
            t = time[i]
            if i == n - 1:
                self.positions.exit_all(i, time[i], cl[i])
                break
            self.positions.update(i, time[i], op[i], hi[i], lo[i], cl[i])
            if ext[i] != 0:
                self.positions.exit_all_signal(None, i, time[i], cl[i])
            if entry[i] != 0:
                sl = self.calc_sl(entry[i], i, self.sl_method, self.sl_value, data)
                self.entry(entry[i], i, time[i], cl[i], sl)
        summary, profit_curve = self.positions.summary()
        return self.positions.to_dataFrame(self.strategy), summary, profit_curve
           
    def entry(self, signal, index, time, price, sl):
        if self.timefilter is not None:
            if self.timefilter.on(time) == False:
                return
        
        if self.positions.num() < self.position_num_max:
            position = Position(self.trade_param, signal, index, time, price, self.volume, sl)
            self.positions.add(position)
            self.last_entry_price = price
            self.last_entry_signal = signal
        
        
    
    

        
    
    
    
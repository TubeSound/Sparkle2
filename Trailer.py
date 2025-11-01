import os
import sys
sys.path.append('../Libraries/trade')
import pickle

import time
import threading
import numpy as np
import pandas as pd
from dateutil import tz
from datetime import datetime, timedelta, timezone
from mt5_trade import Mt5Trade, Columns
from MetaTrader5 import TradePosition
import sched

from trade_manager import TradeManager, PositionInfo, Signal
from time_utils import TimeUtils
from utils import Utils
from common import Indicators

from maron_pie import MaronPie, MaronPieParam
from montblanc import Montblanc, MontblancParam

AXIORY_SUMMER_TIME_BEGIN_MONTH = 3
AXIORY_SUMMER_TIME_BEGIN_WEEK = 2
AXIORY_SUMMER_TIME_END_MONTH = 11
AXIORY_SUMMER_TIME_END_WEEK = 1
AXIORY_DELTA_TIME_TO_UTC = 3.0



JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')  

import logging
os.makedirs('./log', exist_ok=True)
log_path = './log/trade_' + datetime.now().strftime('%y%m%d_%H%M') + '.log'
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p"
)

# -----

scheduler = sched.scheduler()

# -----
def utcnow():
    utc = datetime.now(UTC)
    return utc

def utc2localize(aware_utc_time, timezone):
    t = aware_utc_time.astimezone(timezone)
    return t

def is_market_open(mt5, timezone, symbol='USDJPY'):
    now = utcnow()
    t = utc2localize(now, timezone)
    t -= timedelta(seconds=5)
    df = mt5.get_ticks_from(symbol, t, length=100)
    return (len(df) > 0)

class Position:
    def __init__(self):
        self.symbol = ""
        self.type = 0
        self.ticket = ""
        self.time = None
        self.price = 0
        self.volume = 0
        self.entry_time = None
        self.entry_price = None
        self.sl = 0.0
        self.tp = 0.0
        self.profit = 0
        self.profit_value = 0        
        
    def to_dict(self):
        d = {
                'symbol': self.symbol,
                'type': self.type,
                'ticket': self.ticket,
                'time': self.time,
                'price': self.price,
                'volume': self.volume,
                'entry_time': self.entry_time,
                'entry_price': self.entry_price,
                'sl': self.sl,
                'tp': self.tp,
                'profit': self.profit,
                'profit_value': self.profit_value            
        }
        return d

class Trailer:
    def __init__(self, symbol, strategy):
        self.symbol = symbol
        self.strategy = strategy
        self.history = []
        mt5 = Mt5Trade( AXIORY_SUMMER_TIME_BEGIN_MONTH,
                        AXIORY_SUMMER_TIME_BEGIN_WEEK,
                        AXIORY_SUMMER_TIME_END_MONTH,
                        AXIORY_SUMMER_TIME_END_WEEK,
                        AXIORY_DELTA_TIME_TO_UTC)
         
        mt5.set_symbol(symbol)
        self.mt5 = mt5
        # === Trailing-stop configuration (current P/L only) ===
        self.trail_enabled = True
        self.trail_mode = 'pct'          # 'abs' or 'pct'
        self.trail_start_trigger = 4000.0  # activate when current_unreal >= this
        self.trail_distance = 30.0        # allowed drawdown (abs or pct)
        self.trail_step_lock = None      # e.g., 2.0 to ratchet lock line by steps
        self.trail_min_positions = 1
        # Negative region handling (耐える設定)
        self.neg_grace_bars = 0          # tolerate first N negative bars
        self.neg_hard_stop = 2000        # force close if current_unreal <= -value
        self.activate_from = 'breakeven' # 'breakeven' or 'rebound'
        self.rebound_from_trough = 0.0   # needed improvement to activate (abs or pct)

        # === Trailing-stop state ===
        self.trailing_activated = False
        self.equity_peak = 0.0           # peak of current_unreal since activation
        self.equity_peak_time = None
        self.lock_line = None            # close if current_unreal <= lock_line
        self.neg_bars = 0                # consecutive negative bars
        self.unreal_trough = 0.0         # min current_unreal while negative
        
        
    def append_to_history(self, time, judge, metrics):
        v =  [self.symbol, time, judge] + metrics + [self.trailing_activated, self.equity_peak, self.equity_peak_time, self.lock_line, self.neg_bars, self.unreal_trough]
        self.history.append(v)
        
    def hisotry_df(self):
        columns = ['symbol', 'jst',  'judge', 'count', 'win_rate', 'profit', 'profit_value', 'activated', 'peak', 'peak_time', 'lock', 'neg_bars', 'through']
        df = pd.DataFrame(data= self.history, columns=columns)
        return df
        
    def timestamp2jst(self, ts):
        t = pd.to_datetime(ts, unit='s')
        utc = self.mt5.severtime2utc([t])
        jst = utc[0].astimezone(JST)
        return jst
    
    def backup_dir(self):
        dir_path = f'./trading/{self.strategy}/{self.symbol}'
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    def get_positions(self):
        mt5_positions = self.mt5.get_positions(self.symbol)
        positions = {}
        for p in mt5_positions:
            pos = Position()
            pos.symbol = p.symbol
            pos.type = p.type
            pos.ticket = p.ticket
            pos.time = datetime.now().astimezone(JST)
            pos.price = p.price_current
            pos.entry_time = self.timestamp2jst(p.time)
            pos.entry_price = p.price_open
            pos.volume = p.volume
            pos.sl = p.sl
            pos.tp = p.tp 
            pos.profit = p.profit
            pos.profit_value = p.price_current - p.price_open
            if self.mt5.is_short(p.type):
                pos.profit_value *= -1
            positions[p.ticket] = pos
        return positions
        
    def calc_metrics(self, position: dict):
        n = len(position.items())
        profit_sum = 0
        profit_value_sum = 0
        win_count = 0
        for ticket, pos in position.items():
            profit_sum += pos.profit
            profit_value_sum += pos.profit_value
            if pos.profit > 0:
                win_count += 1
        if n > 0:
            win_rate = win_count / n
        else:
            win_rate = 0
        return [n, win_rate, profit_sum, profit_value_sum]
    
    def get_path(self):
        now = datetime.now().astimezone(JST)
        filename = f"{self.strategy}_{self.symbol}_profits_history_{now.year}-{now.month}-{now.day}.csv"
        return os.path.join(self.backup_dir(), filename)    
    
    def update(self):
        dic = self.get_positions()
        metrics = self.calc_metrics(dic)
        r = self.judge_trailing_stop(metrics)
        if r > 0:
            pass
            #self.mt5.close_all_position(self.symbol)
        self.append_to_history(datetime.now().astimezone(JST), r, metrics)
        try:
            df = self.hisotry_df()
            df.to_csv(self.get_path(), index=False)
        except:
            pass
            
            
    def judge_trailing_stop(self, metrics):
        """
        Decide whether to close all positions based on current (unrealized) P/L only.
        Returns True to close, False to continue.
        """
        if not self.trail_enabled:
            return 0
        
        n, win_rate, profit_sum, profit_value_sum = metrics
        
        if n < self.trail_min_positions:
            self.neg_bars = 0
            self.unreal_trough = 0.0
            self.trailing_activated = False
            self.lock_line = None
            self.equity_peak = 0.0
            return 0
       
        current_unreal = float(profit_sum)
        now = datetime.now().astimezone(JST)
        # Negative region handling
        if current_unreal < 0:
            self.neg_bars += 1
            if (self.neg_bars == 1) or (current_unreal < self.unreal_trough):
                self.unreal_trough = current_unreal
            if self.neg_hard_stop is not None and current_unreal <= -abs(self.neg_hard_stop):
                self._reset_trailing_negative_state(current_unreal, now)
                print('Trailing Hard Stop', self.symbol, 'Profit', current_unreal, 'setting:', self.neg_hard_stop)
                return 2
            if self.neg_bars <= max(0, self.neg_grace_bars):
                return 0
            if self.activate_from == 'breakeven':
                return 0
            else:
                if self.trail_mode == 'pct':
                    need = abs(self.unreal_trough) * (self.rebound_from_trough / 100.0)
                else:
                    need = self.rebound_from_trough
                improved = current_unreal - self.unreal_trough
                if improved < need:
                    return 0
                self.trailing_activated = True
        else:
            self.neg_bars = 0
            self.unreal_trough = 0.0
            if self.activate_from == 'breakeven' and current_unreal >= self.trail_start_trigger:
                self.trailing_activated = True
        if not self.trailing_activated:
            return 0
        if current_unreal < self.trail_start_trigger:
            return 0
        if current_unreal > self.equity_peak:
            self.equity_peak = current_unreal
            self.equity_peak_time = now
        dd_allow = self.equity_peak * (self.trail_distance / 100.0) if self.trail_mode=='pct' else self.trail_distance
        new_lock = self.equity_peak - dd_allow
        if self.lock_line is None:
            self.lock_line = new_lock
        else:
            if new_lock > self.lock_line:
                self.lock_line = new_lock
            if self.trail_step_lock is not None:
                step_val = (self.equity_peak * (self.trail_step_lock/100.0)) if self.trail_mode=='pct' else self.trail_step_lock
                while (self.equity_peak - self.lock_line) > (dd_allow + step_val):
                    self.lock_line += step_val
        if current_unreal <= (self.lock_line if self.lock_line is not None else -1e18):
            self.lock_line = None
            self.equity_peak = current_unreal
            self.equity_peak_time = now
            self.trailing_activated = False
            self.neg_bars = 0
            self.unreal_trough = 0.0
            print('Trailing Stop fired:', self.symbol, 'Profit:', current_unreal, self.lock_line)
            return 1
        return 0

    def _reset_trailing_negative_state(self, current_unreal: float, now: datetime):
        self.lock_line = None
        self.equity_peak = current_unreal
        self.equity_peak_time = now
        self.trailing_activated = False
        self.neg_bars = 0
        self.unreal_trough = 0.0


def execute(strategy, symbols):
    trailers = {}
    for i, symbol in enumerate(symbols):
        trailer = Trailer(symbol, strategy)
        trailers[symbol ] = trailer
        if i == 0:
            Mt5Trade.connect()
    while True:
        for symbol in symbols:
            scheduler.enter(5, 1 + 1, trailers[symbol].update)
            scheduler.run()
        
        
def test():
    trailer = Trailer('JP225')
    dic = trailer.get_positions()
    for ticket, pos in dic.items():
        print(pos.to_dict())
        
    r = trailer.metric(dic)
    n, win_rate, profit, profit_value = r
    print('metric', r)
     
if __name__ == "__main__":
    symbols = ['JP225', 'XAUUSD']
    execute('Montblanc', symbols)
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
    def __init__(self, symbol):
        self.symbol = symbol
        self.history = []
        mt5 = Mt5Trade( AXIORY_SUMMER_TIME_BEGIN_MONTH,
                        AXIORY_SUMMER_TIME_BEGIN_WEEK,
                        AXIORY_SUMMER_TIME_END_MONTH,
                        AXIORY_SUMMER_TIME_END_WEEK,
                        AXIORY_DELTA_TIME_TO_UTC)
         
        mt5.set_symbol(symbol)
        self.mt5 = mt5
        self.mt5.connect()
        
    def hisotry_df(self):
        df = pd.DataFrame(data= self.history, columns=['symbol', 'jst', 'count', 'win_rate', 'profit', 'profit_value'])
        return df
        
    def timestamp2jst(self, ts):
        t = pd.to_datetime(ts, unit='s')
        utc = self.mt5.severtime2utc([t])
        jst = utc[0].astimezone(JST)
        return jst
    
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
    
    def update(self):
        dic = self.get_positions()
        metrics = self.calc_metrics(dic)
        h = [self.symbol, datetime.now().astimezone(JST)] + metrics
        print(h)
        self.history.append(h)
        r = self.judge_trailing_stop()
        if r:
            self.mt5.close_all_position(self.symbol)
            
    def judge_trailing_stop(self):
        return False

def execute(symbols):
    trailers = {}
    for i, symbol in enumerate(symbols):
        trailer = Trailer(symbol)
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
    execute(symbols)
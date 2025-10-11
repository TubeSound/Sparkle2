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
import sched

from trade_manager import TradeManager, PositionInfo, Signal
from time_utils import TimeUtils
from utils import Utils
from common import Indicators

from maron_pie import MaronPie, MaronPieParam


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
    #utc1 = datetime.utcnow()
    #utc1 = utc1.replace(tzinfo=UTC)
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

def save(data, path):
    d = data.copy()
    time = d[Columns.TIME] 
    d[Columns.TIME] = [str(t) for t in time]
    jst = d[Columns.JST]
    d[Columns.JST] = [str(t) for t in jst]
    df = pd.DataFrame(d)
    df.to_excel(path, index=False)
    

class TradeBot:
    def __init__(self, strategy: str, symbol:str, param):
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = 'M1'
        self.data_length =  2 * 24 * 60
        self.invterval_seconds = 10
        self.param = param
        if strategy == 'MaronPie':
            self.act = MaronPie(symbol, param)
        mt5 = Mt5Trade(3, 2, 11, 1, 3.0) 
        mt5.set_symbol(symbol)
        self.trade_manager = self.load_trade_manager()
        
        self.last_time = None
        self.mt5 = mt5
        self.delta_hour_from_gmt = None
        self.server_timezone = None
        
        
    def debug_print(self, *args):
        utc = utcnow()
        jst = utc2localize(utc, JST)
        t_server = utc2localize(utc, self.server_timezone)  
        s = 'JST*' + jst.strftime('%Y-%m-%d_%H:%M:%S') + ' (ServerTime:' +  t_server.strftime('%Y-%m-%d_%H:%M:%S') +')'
        for arg in args:
            s += ' '
            s += str(arg) 
        print(s)    
                
        
    def set_sever_time(self, begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer):
        now = datetime.now(JST)
        dt, tz = TimeUtils.delta_hour_from_gmt(now, begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer)
        self.delta_hour_from_gmt  = dt
        self.server_timezone = tz
        print('SeverTime GMT+', dt, tz)
        
        
    def backup_path(self):
        dir_path = f'./trading/{self.strategy}/tmp/{self.symbol}/{self.timeframe}'
        path = os.path.join(dir_path, f'{self.strategy}_{self.symbol}_{self.timeframe}_trade_manager.pkl')
        return path, dir_path
        
    def load_trade_manager(self):
        path, dir_path = self.backup_path()
        try:
            with open(path, 'rb') as f:
                trade_manager = pickle.load(f)
            print(self.symbol, self.timeframe, ' loaded Trade_manager positions num: ', len(self.trade_manager.positions))
        except:
            trade_manager = TradeManager(self.symbol, self.timeframe)
        return trade_manager
            
    def save_trade_manager(self):
        path, dir_path = self.backup_path()
        os.makedirs(dir_path, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.trade_manager, f)     
            
    def df2dic(self, df):
        jst = df[Columns.JST].to_numpy()
        op = df[Columns.OPEN].to_numpy()
        hi = df[Columns.HIGH].to_numpy()
        lo = df[Columns.LOW].to_numpy()
        cl = df[Columns.CLOSE].to_numpy()
        dic = {Columns.JST: jst, Columns.OPEN: op, Columns.HIGH: hi, Columns.LOW: lo, Columns.CLOSE: cl}
        return dic    
            
    def run(self):
        df = self.mt5.get_rates(self.symbol, self.timeframe, self.data_length)
        if len(df) < self.data_length:
            raise Exception('Error in initial data loading')
        if is_market_open(self.mt5, self.server_timezone):
            self.update()
            os.makedirs('./debug', exist_ok=True)
            #save(buffer.data, './debug/initial_' + self.symbol + '_' + datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + '.xlsx')
            return True            
        else:
            print(f'*マーケットクローズ: {self.symbol}')
            return False
    
    def update(self):
        t0 = datetime.now()
        self.remove_closed_positions()
        df = self.mt5.get_rates(self.symbol, self.timeframe, self.data_length)
        # remove invalid data
        df = df.iloc[:-1, :]
        jst = df[Columns.JST].to_list()
        if self.last_time is None:
            self.last_time = jst[-1]
        else:
            #print(self.symbol, 'Last:', self.last_time, 'Now: ', jst[-1])
            if jst[-1] <= self.last_time:
                #print(self.symbol, 'Pass')
                return
            else:
                self.last_time = jst[-1]            
        self.update_sl()
        t1 = datetime.now()
        self.act.calc(df)
        t2 = datetime.now()
        #print(t2, ' ... Elapsed time: ', t1 - t0, t2 - t1, 'total:', t2 - t0)
        
        # ドテン
        ext = self.act.exits[-1]
        self.doten(ext)
        ent = self.act.entries[-1]     
        if ent == Signal.LONG:
            self.entry(Signal.LONG, jst[-1])
            self.save_trade_manager()
        elif ent == Signal.SHORT:
            self.entry(Signal.SHORT, jst[-1])
            self.save_trade_manager()
        
    def doten(self, signal):
        if signal == 0:
            return
        for ticket, position in self.trade_manager.positions.items():
            if signal == 1:
                # down -> up
                if position.order_signal == Signal.SHORT:
                    self.close_position(position)
            elif signal == -1:
                # up ->down
                if position.order_signal == Signal.LONG:
                    self.close_position(position)
        
    def close_position(self, position: PositionInfo):
        ret, _ = self.mt5.close_by_position_info(position)
        if ret:
            self.debug_print('<Closed> Success', self.symbol, position.desc())
            self.trade_manager.remove_positions([position.ticket])
        else:
            self.debug_print('<Closed> Fail', self.symbol, position.desc())           
        return ret
    
    def close_all_positions(self):
        for ticket, position in self.trade_manager.positions.items():
            self.close_position(position)
                
    def update_sl(self):
        for ticket, position in self.trade_manager.open_positions().items():
            if (self.param.sl_loose is not None) and (position.sl_updated is False):
                if position.order_signal == Signal.LONG:
                    sl = position.entry_price - self.param.sl
                elif position.order_signal == Signal.SHORT:
                    sl = position.entry_price + self.param.sl
                else:
                    continue
                self.mt5.modify_sl(self.symbol, position.ticket, sl)
                position.sl_updated = True        
        
    def mt5_position_num(self):
        positions = self.mt5.get_positions(self.symbol)
        count = 0
        for position in positions:
            if position.symbol == self.symbol:
                count += 1
        return count
        
    def entry(self, signal, time):
        volume = self.param.volume
        if self.param.sl_loose is None:
            sl = self.param.sl
        else:
            sl = self.param.sl * self.param.sl_loose
        tp = self.param.tp                     
        position_max = int(self.param.position_max)
        num =  self.mt5_position_num()
        if num >= position_max:
            self.debug_print('<Entry> Request Canceled ', self.symbol, time,  'Position num', num)
            return
        try:
            ret, position_info = self.mt5.entry(self.symbol, signal, time, volume, stoploss=sl, takeprofit=tp)
            if ret:
                self.trade_manager.add_position(position_info)
                self.debug_print('<Entry> signal', position_info.order_signal, position_info.symbol, position_info.entry_time)
            else:
                self.debug_print('<Entry Error> signal', signal, sl, tp)
        except Exception as e:
            print(' ... Entry Error', e)
            print(position_info)
            


def load_params(strategy, symbol, volume, position_max):
    def array_str2int(s):
        i = s.find('[')
        j = s.find(']')
        v = s[i + 1: j]
        return float(v)
    
    print( os.getcwd())
    path = f'./{strategy}/{strategy}_best_trade_params.xlsx'
    df = pd.read_excel(path)
    df = df[df['symbol'] == symbol]
    params = []
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        if strategy == 'MaronPie':
            param = MaronPieParam.load_from_dic(row)
            params.append(param)      
        else:
            raise Exception('No definition ' + strategy)
    return params

def create_bot(strategy, symbol, lot):
    params = load_params(strategy, symbol, lot, 10)
    print(symbol, 'parameter num', len(params))
    param = params[0]
    param.volume = lot
    bot = TradeBot(strategy, symbol, param)    
    bot.set_sever_time(3, 2, 11, 1, 3.0)
    return bot
         
def is_trade_time():
    now = datetime.now()
    begin = datetime(now.year, now.month, now.day, hour=7)
    end = datetime(now.year, now.month, now.day, hour=8) 
    return not (now >= begin and now < end)
         
def is_close_time():
    hours = [0, 7, 16]
    now = datetime.now()
    for hour in hours:
        begin = datetime(now.year, now.month, now.day, hour=hour, minute=0)
        end = datetime(now.year, now.month, now.day, hour=hour, minute=2)
        if (now >= begin and now < end):
            return True
    return False
         
def execute(strategy, items):
    bots = {}
    for i, (symbol, lot) in enumerate(items):
        bot = create_bot(strategy, symbol, lot)
        if i == 0:
            Mt5Trade.connect()
        bot.run()
        bots[symbol ] = bot
        
    while True:
        for i, (symbol, lot) in enumerate(items):
            if is_close_time():
                bots[symbol].close_all_positions()
            else:   
                scheduler.enter(5, 1 + 1, bots[symbol].update)
                scheduler.run()
            
def test():
    
    params = load_params('NSDQ', 0.01, 20)
    print(params)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))    
    strategy = 'MaronPie'
    items = [    # [symbol, volume, sl_loose]
                ['XAUUSD', 0.01],
                ['USDJPY', 0.1],
                ['JP225', 10], 
                ['US100', 1],
            ]
    execute(strategy, items)
    #test()
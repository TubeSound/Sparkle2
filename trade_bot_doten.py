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
from montblanc import Montblanc, MontblancParam


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
    def __init__(self, strategy: str, symbol:str, param: MontblancParam):
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = 'M1'
        self.data_length =  4 * 24 * 60
        self.invterval_seconds = 10
        self.param = param
        if strategy.lower() == 'maronPie':
            self.act = MaronPie(symbol, param)
        elif strategy.lower() == 'montblanc':
            self.act = Montblanc(symbol, param)
        mt5 = Mt5Trade(3, 2, 11, 1, 3.0) 
        mt5.set_symbol(symbol)
        self.trade_manager = self.load_trade_manager()
        self.set_trailing_stop()
        
        self.last_time = None
        self.mt5 = mt5
        self.delta_hour_from_gmt = None
        self.server_timezone = None
        
        
    def debug_print(self, *args):
        utc = utcnow()
        jst = utc2localize(utc, JST)
        t_server = utc2localize(utc, self.server_timezone)  
        s = f'[{self.symbol}] ' + jst.strftime('%m-%d_%H:%M')
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
        
        
    def backup_dir(self):
        dir_path = f'./trading/{self.strategy}/{self.symbol}/{self.timeframe}'
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    def trade_manager_path(self):
        return os.path.join(self.backup_dir(), f'{self.strategy}_{self.symbol}_{self.timeframe}_trade_manager.pkl')
        
    def load_trade_manager(self):
        try:
            with open(self.trade_manager_path(), 'rb') as f:
                trade_manager = pickle.load(f)
            print(self.symbol, self.timeframe, ' loaded Trade_manager positions num: ', len(trade_manager.positions))
        except:
            trade_manager = TradeManager(self.symbol, self.timeframe)
        return trade_manager
    
    def set_trailing_stop(self):
        self.trade_manager.set_trailing(    
                                            enabled=False,
                                            mode = "abs",
                                            start_trigger=100,
                                            distance=50,
                                            step_lock=50,
                                            min_positions=1,
                                            neg_grace_bars=0,
                                            neg_hard_stop=100,
                                        )
            
    def save_trade_manager(self):
        with open(self.trade_manager_path(), 'wb') as f:
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
        
    def now_str(self):
        t = datetime.now()
        return datetime.strftime(t, '%Y-%m-%d_%H')
    
    
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
                
        cl = df[Columns.CLOSE].to_list()        
        judge = self.trade_manager.judge_stop(cl[-1])
        try:
            df_profit = self.trade_manager.history_df()
            path = os.path.join(self.backup_dir(), f'{self.strategy}_{self.symbol}_profit_history.csv')
            df_profit.to_csv(path, index=False)
        except:
            pass
        if judge:
            self.debug_print('Trailing Stop:')
            self.close_all_positions()
            return
        t1 = datetime.now()
        self.act.calc(df)
        try:
            df_save = self.act.signals_df()
            dirpath = f'./trading/{self.strategy}'
            os.makedirs(dirpath, exist_ok=True)
            filename = f'{self.symbol}_signals_{self.now_str()}.csv'
            filepath = os.path.join(dirpath, filename)
            df_save.to_csv(filepath, index=False)
        except:
            pass
        t2 = datetime.now()
        #print(t2, ' ... Elapsed time: ', t1 - t0, t2 - t1, 'total:', t2 - t0)
        
        if self.param.sl_mode == 'atr':
            self.update_sl(self.act.upper_minor, self.act.lower_minor, self.param.sl_value)
        
        # ドテン
        ext = self.act.exits[-1]
        self.doten(ext)
        ent = self.act.entries[-1]     
        if ent == Signal.LONG:
            sl = self.calc_sl_price(True)
            self.entry(Signal.LONG, jst[-1], sl)
            self.save_trade_manager()
        elif ent == Signal.SHORT:
            sl = self.calc_sl_price(False)
            self.entry(Signal.SHORT, jst[-1], sl)
            self.save_trade_manager()
        
    def update_sl(self, upper_line, lower_line, offset):
        mt5_positions = self.mt5.get_positions(self.symbol)
        for p in mt5_positions:
            if p.symbol != self.symbol:
                continue
            if self.mt5.is_long(p.type):
                if np.isnan(lower_line[-2]):
                    continue
                if np.isnan(lower_line[-1]):
                    #self.debug_print('Bad lower_line for stop')
                    continue
                if lower_line[-1] != lower_line[-2]:
                    # change sl
                    sl = lower_line[-1] - offset
                    self.mt5.modify_sl(self.symbol, p.ticket, sl)
                    #self.debug_print('Changed Stoploss', p.sl, '->', sl)
            elif self.mt5.is_short(p.type):
                if np.isnan(upper_line[-2]):
                    continue
                if np.isnan(upper_line[-1]):
                    #self.debug_print('Bad lower_line for stop')
                    continue
                if upper_line[-1] != upper_line[-2]:
                    sl = upper_line[-1] + offset
                    self.mt5.modify_sl(self.symbol, p.ticket, sl)
                    #self.debug_print('Changed Stoploss', p.sl, '->', sl)
        
    def doten(self, signal):
        if signal == 0:
            return
        #self.debug_print('doten', signal)
        positions = self.trade_manager.positions.copy()
        for ticket, position in positions.items():
            if signal == 1:
                # down -> up
                
                if position.order_signal == Signal.SHORT:
                    self.close_position(position)
                    #self.debug_print('close: down->up ', position.ticket)
            elif signal == -1:
                # up ->down
                
                if position.order_signal == Signal.LONG:
                    self.close_position(position)
                    #self.debug_print('trend: up->down', position.ticket)
        
    def remove_closed_positions(self):
        positions = self.mt5.get_positions(self.symbol)
        self.trade_manager.remove_position_auto(positions)    
        
    def close_position(self, position: PositionInfo):
        ret, _ = self.mt5.close_by_position_info(position)
        if ret:
            self.debug_print('<Exit> ', position.desc())
            self.trade_manager.remove_positions([position.ticket])
        else:
            self.debug_print('<Exit Fail>', position.desc())           
        return ret
    
    def close_all_positions(self):
        positions = self.trade_manager.positions.copy()
        for ticket, position in positions.items():
            self.close_position(position)
                    
    def mt5_position_num(self):
        positions = self.mt5.get_positions(self.symbol)
        count = 0
        for position in positions:
            if position.symbol == self.symbol:
                count += 1
        return count
        
    def calc_sl_price(self, is_long):
        if self.param.sl_mode.lower() == 'fix':
            if is_long:
                return self.act.cl[-1] - self.param.sl
            else:
                return self.act.cl[-1] + self.param.sl
        elif self.param.sl_mode.lower() == 'atr':
            if is_long:
                return self.act.lower_minor[-1] - self.param.sl
            else:
                return self.act.upper_minor[-1] + self.param.sl
        raise Exception('Bad sl calc')
        
    def entry(self, signal, time, sl):
        volume = self.param.volume
        #tp = self.param.tp                     
        position_max = int(self.param.position_max)
        num =  self.mt5_position_num()
        if num >= position_max:
            self.debug_print('<Entry> Request Canceled ', 'Position num', num)
            return
        try:
            ret, position_info = self.mt5.entry(self.symbol, signal, time, volume, stoploss=sl) #, takeprofit=tp)
            if ret:
                self.trade_manager.add_position(position_info)
                self.debug_print(f'<Entry> signal:{position_info.order_signal}  ticket: {position_info.ticket} price: {position_info.entry_price}')
            else:
                self.debug_print('<Entry Error> signal', signal, sl)
        except Exception as e:
            self.debug_print(' ... Entry Error', e, position_info)

def load_params(strategy, symbol, ver, volume, position_max):
    def array_str2int(s):
        i = s.find('[')
        j = s.find(']')
        v = s[i + 1: j]
        return float(v)
    
    print( os.getcwd())
    path = f'./{strategy}/v{ver}/{strategy}_v{ver}_best_trade_params.xlsx'
    df = pd.read_excel(path)
    df = df[df['symbol'] == symbol]
    params = []
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        if strategy == 'MaronPie':
            param = MaronPieParam.load_from_dic(row)
            param.volume = volume
            param.position_max = position_max
            params.append(param)      
        elif strategy == 'Montblanc':
            param = MontblancParam.load_from_dic(row)
            param.volume = volume
            param.position_max = position_max
            params.append(param)  
        else:
            raise Exception('No definition ' + strategy)
    return params

def create_bot(strategy, symbol, ver, lot):
    params = load_params(strategy, symbol, ver, lot, 10)
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
    return False
    hours = [[23, 58], [6, 50], [16, 30]]
    now = datetime.now()
    for hour, minute in hours:        
        begin = datetime(now.year, now.month, now.day, hour=hour, minute=minute)
        end = begin + timedelta(minutes=1)
        if (now >= begin and now < end):
            return True
    return False
         
def is_trade_time(symbol):
    if symbol == 'USDJPY':
        return True    
    # 休憩時間　# 07:00 ～ 08:15, 16:30 ～ 21:15はトレードしない
    rest = [
                [[7, 0], [8, 15]], 
                [[16, 30], [17, 00]]
            ]
    now = datetime.now()
    for [hour0, min0], [hour1, min1] in rest:
        begin = datetime(now.year, now.month, now.day, hour=hour0, minute=min0)
        end = datetime(now.year, now.month, now.day, hour=hour1, minute=min1)
        if (now >= begin and now < end):
            return False
    return True
         
def execute(strategy, ver, items):
    bots = {}
    for i, (symbol, lot) in enumerate(items):
        bot = create_bot(strategy, symbol, ver, lot)
        if i == 0:
            Mt5Trade.connect()
        bot.run()
        bots[symbol ] = bot
        
    while True:
        for i, (symbol, lot) in enumerate(items):
            if is_trade_time(symbol):
                if is_close_time():
                    bots[symbol].close_all_positions()
                else:   
                    scheduler.enter(5, 1 + 1, bots[symbol].update)
                    scheduler.run()
            
def test():
    
    params = load_params('NSDQ', 0.01, 20)
    print(params)

def maronpie():
    strategy = 'MaronPie'
    items = [    # [symbol, volume, sl_loose]
                #['XAUUSD', 0.01],
                ['USDJPY', 0.1],
                ['JP225', 5], 
                #['US30', 0.1],
                #['US100', 0.1]
            ]
    execute(strategy, '4.1', items)    


def montblanc():
    strategy = 'Montblanc'
    items = [    # [symbol, volume, sl_loose]
                ['XAUUSD', 0.01],
                ['USDJPY', 0.1],
                ['JP225', 10], 
                ['US30', 0.1],
                ['US100', 0.1]
            ]
    execute(strategy, 3, items)    
    
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))    
    montblanc()
    #test()
import os
import shutil
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc') 
from common import Columns

def from_pickle(symbol, timeframe, source='Axiory'):
    import pickle
    symbol = symbol.lower()
    timeframe = timeframe.upper()
    if source=='Axiory':
        filepath = f'./data/{source}/{symbol}_{timeframe}.pkl'
    elif source=='225labo' and symbol == 'nikkei' and timeframe == 'H1':
        filepath = f'./data/{source}/nikkei225f_h1.pkl'
    else:
        filepath = f'./data/BacktestMarket/BM_{symbol}_{timeframe}.pkl'
    with open(filepath, 'rb') as f:
        data0 = pickle.load(f)
    return data0

class DataLoader:
    def __init__(self, dealer):
        self.dealer = dealer
        
    def server_time_str_2_datetime(self, server_time_str_list, server_timezone, format='%Y-%m-%d %H:%M:%S'):
        t_utc = []
        t_jst = []
        for time_str in server_time_str_list:
            i = time_str.find('+')
            if i > 0:
                time_str = time_str[:i]
            t = datetime.strptime(time_str, format)
            t = t.replace(tzinfo=server_timezone)
            utc = t.astimezone(UTC)
            t_utc.append(utc)
            jst = t.astimezone(JST)        
            t_jst.append(jst)
        return t_utc, t_jst


    def str2datetime(self, times, format, tz=UTC):
        out = []
        for t in times:
            d = datetime.strptime(t[:19], format)
            d1 = d.replace(tzinfo=tz)
            out.append(d1)
        return out
    
    def data_filepath(self, symbol, timeframe, year, month):
        path = f'../MarketData/{self.dealer}/'
        dir_path = os.path.join(path, symbol, timeframe)
        name = symbol + '_' + timeframe + '_' + str(year) + '_' + str(month).zfill(2) + '.csv'
        filepath = os.path.join(dir_path, name)
        if os.path.isfile(filepath):
            return filepath 
        else:
            return None
        
    def load_data(self, symbol, timeframe, from_year, from_month, to_year, to_month):
        dfs = []
        year = from_year
        month = from_month
        while True:
            filepath = self.data_filepath(symbol, timeframe, year, month)
            if filepath is not None:
                df = pd.read_csv(filepath)
                if len(df) > 0:
                    dfs.append(df)
            if year == to_year and month == to_month:
                break
            month += 1
            if month > 12:
                year += 1
                month = 1
                
        if len(dfs) == 0:
            return 0, {}
        df = pd.concat(dfs, ignore_index=True)
        n = len(df)
        dic = {}
        for column in df.columns:
            dic[column] = list(df[column].values)
        tzone = timezone(timedelta(hours=2))
        #if timeframe.upper() == 'D1'or timeframe.upper() == 'W1':
        #    format='%Y-%m-%d'
        #else:
        format='%Y-%m-%d %H:%M:%S'
        time =  self.str2datetime(df[Columns.TIME], format)
        jst = self.str2datetime(df[Columns.JST], format, tz=JST)
        dic[Columns.TIME] = time
        dic[Columns.JST] = jst
        print(symbol, timeframe, 'Data size:', len(jst), jst[0], '-', jst[-1])
        self.size = n
        return n, dic
    
    def data(self):
        return self.dic
    
    
def save(filepath, obj):
    import pickle
    with open(filepath, mode='wb') as f:
        pickle.dump(obj, f)
        
        
def arrange(data):
    time = data['time']
    jst = []
    utc = []
    for t in time:
        tj = t + timedelta(hours=14) 
        jst.append(tj.astimezone(JST))
        tu = t + timedelta(hours=2)
        utc.append(tu.astimezone(UTC))
    data['jst'] = jst
    data['utc'] = utc   
    
    
def load_backtest_market():
    file = './data/BacktestMarket/csv/spi500_h1.csv'
    df = pd.read_csv(file)
    date = df['date']
    time0 = df['time']
    
    time = []
    for d, t in zip(date, time0):
        tim = datetime.strptime(d + ' ' + t, '%d/%m/%Y %H:%M:%S')
        time.append(tim)
    
    data = {}    
    data['time'] = time
    data['open'] = df['open'].to_list()
    data['high'] = df['high'].to_list()
    data['low'] = df['low'].to_list()
    data['close']  = df['close'].to_list()
    arrange(data)
    save('./data/BacktestMarket/BM_spi500_H1.pkl', data)
    

def to_float(array):
    return [float(v) for v in array]
    
    
def load_225labo():
    dirpath = './data/225labo'
    files = glob.glob(os.path.join(dirpath, 'csv/*.csv'))
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    df = pd.concat(dfs)
    
    df = df.sort_values('jst')
    
    time = df['jst'].to_list()
    jst = [datetime.strptime(tstr, '%Y/%m/%d %H:%M').astimezone(JST) for tstr in time]
    
    data = {}    
    data['jst'] = jst
    data['open'] = to_float(df['open'])
    data['high'] = to_float(df['high'])
    data['low'] = to_float(df['low'])
    data['close'] = to_float(df['close'])
    data['volume'] = to_float(df['volume'])

    save(os.path.join(dirpath, 'nikkei225f_h1.pkl'), data)    
    
if __name__ == '__main__':
    #load_backtest_market()
    load_225labo()   
        
        
        
        
        
        
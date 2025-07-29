# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:44:13 2023

@author: docs9
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from mt5_api import Mt5Api
from common import TimeFrame
from datetime import datetime, timedelta
from dateutil import tz
from time_utils import TimeFilter, TimeUtils
from data_loader import DataLoader

JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc') 

AXIORY = 'axiory'
FXGT = 'fxgt'

def all_symbols():
    symbols = ['NIKKEI', 'DOW', 'NSDQ', 'SP', 'HK50', 'DAX', 'FTSE', 'XAUUSD']
    symbols += ['USDJPY', 'GBPJPY', 'NVDA', 'TSLA']
    symbols += ['CL', 'NGAS', 'XAGUSD', 'XPDUSD', 'XPTUSD']
    return symbols

def download(symbols, save_holder):
    api = Mt5Api()
    api.connect()
    for symbol in symbols:
        for tf in [TimeFrame.M5, TimeFrame.M15, TimeFrame.M30, TimeFrame.H1, TimeFrame.H4, TimeFrame.D1, TimeFrame.M1]:
            for year in range(2025, 2026):
                for month in range(7, 8):
                    t0 = datetime(year, month, 1, 0)
                    t0 = t0.replace(tzinfo=JST)
                    t1 = t0 + relativedelta(months=1) - timedelta(seconds=1)
                    if tf == 'TICK':
                        rates = api.get_ticks(t0, t1)
                    else:
                        rates = api.get_rates_jst(symbol, tf, t0, t1)
                    path = os.path.join(save_holder, symbol, tf)
                    os.makedirs(path, exist_ok=True)
                    path = os.path.join(path, symbol + '_' + tf + '_' + str(year) + '_' + str(month).zfill(2) + '.csv')
                    df = pd.DataFrame(rates)
                    if len(df) > 10:
                        df.to_csv(path, index=False)
                    print(symbol, tf, year, '-', month, 'size: ', len(df))
    
    pass


def download_tick(symbols, year, month, h, save_holder):
    api = Mt5Api()
    api.connect()
    for symbol in symbols:
        tf = 'TICK'
        day = 7 * int(h - 1) + 1
            
        t0 = datetime(year, month, day).replace(tzinfo=JST)
        if h == 4:
            if month == 12:
                y = year + 1
                m = 1
            else:
                y = year
                m = month + 1
            t1 = datetime(y, m, 1).replace(tzinfo=JST)
        else:
            t1 = t0 + timedelta(days=7)
        df = api.get_ticks(symbol, t0, t1)
        path = os.path.join(save_holder, symbol, 'Tick')
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f"{symbol}_{tf}_{year}-{str(month).zfill(2)}-{h}.csv")
        df = df[['jst', 'bid', 'ask', 'volume', 'volume_real', 'flags']]
        df.to_csv(path, index=False)
        print(path, symbol, tf, year, '-', month, 'size: ', len(df))

    pass


def dl1(dealer):
    if dealer == AXIORY:
        symbols = all_symbols()
    else:
        symbols = ['BTCUSDs']
    download(symbols, f'../MarketData/{dealer}/')
    
def dl2():
    symbols = ['SP', 'HK50', 'DAX', 'FTSE',  'XAGUSD', 'EURJPY', 'AUDJPY']
    symbols = ['NIKKEI', 'USDJPY']
    download(symbols, '../MarketData/Axiory/')
    
def save(filepath, obj):
    import pickle
    with open(filepath, mode='wb') as f:
        pickle.dump(obj, f)
    
def save_data(dealer, symbols):
    year_from = 2020
    month_from = 1
    year_to = 2025
    month_to = 7
    loader = DataLoader(dealer)
    tfs = ['M15', 'M30', 'H1', 'H4', 'D1']
    for symbol in symbols:
        for tf in tfs:
            n, data = loader.load_data(symbol, tf, year_from, month_from, year_to, month_to)
            os.makedirs(f'./data/{dealer}', exist_ok=True)
            save(f'./data/{dealer}/' + symbol + '_' + tf + ".pkl", data)
    
def main():
    dl1(AXIORY)
    save_data(AXIORY, all_symbols())
    from analyze_atrp import main5
    main5()
    #download_tick(['DOW', 'NIKKEI', 'NSDQ', 'XAUUSD'], './data/Axiory/tick')
 
def main2():
    symbols = ['NSDQ', 'XAUUSD', 'NIKKEI', 'CL', 'DOW']   
    root = '../MarketData/Axiory'
    year = 2025
    month = 7
    for i in range(1, 2):
        download_tick(symbols, year, month, i, root)
        
def main3():
    symbol = 'NSDQ'
    t0 = datetime(2025, 7, 15, 22).astimezone(tz=JST)
    t1 = datetime(2025, 7, 16, 2).astimezone(tz=JST)
    tf = 'M1'
    path = './debug/nsdq_m1.csv'
    api = Mt5Api()
    api.connect()
    rates = api.get_rates_jst(symbol, tf, t0, t1)
    df = pd.DataFrame(rates)
    if len(df) > 10:
        df.to_csv(path, index=False)
    

def main4(symbol, year, month, timeframe):
    mstr = str(month).zfill(2)
    dirpath = f'../DayTradeData/{timeframe}/{symbol}/{year}-{mstr}'
    os.makedirs(dirpath, exist_ok=True)
    api = Mt5Api()
    api.connect()
    for day in range(1, 31):
        for i in range(1, 5):
            t0 = datetime(year, month, day, 8).astimezone(tz=JST)
            try:
                t1 = t0 + timedelta(hours= (i - 1) * 4)
                t2 = t1 + timedelta(hours=4) 
            except:
                continue
            dstr = str(day).zfill(2)
            filename = f'{symbol}_{timeframe}_{year}-{mstr}-{dstr}-{i}.csv'
            if timeframe == 'tick':
                df = api.get_ticks(symbol, t1, t2)    
                df = df[['jst', 'bid', 'ask', 'volume', 'volume_real', 'flags']]
            else:
                dic = api.get_rates_jst(symbol, timeframe, t1, t2)
                df = pd.DataFrame(dic)
            if len(df) > 10:
                df.to_csv(os.path.join(dirpath, filename), index=False)

if __name__ == '__main__':
    symbol = 'NSDQ'
    timeframe = 'tick'
    for m in range(4, 8):
        main4(symbol, 2025, m, timeframe)
    


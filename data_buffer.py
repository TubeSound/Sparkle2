import pandas as pd
import numpy as np
from mt5_trade import Mt5Trade, TimeFrame, Columns, nptimestamp2pydatetime
from datetime import datetime, timedelta
from common import Signal, Indicators
from utils import Utils


from dateutil import tz

JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc') 

COLUMNS = [Columns.TIME, Columns.OPEN, Columns.HIGH, Columns.LOW, Columns.CLOSE, 'tick_volume']
TICK_COLUMNS = [Columns.TIME, Columns.ASK, Columns.BID]

def nans(length):
    return [np.nan for _ in range(length)]

def nones(length):
    return [None for _ in range(length)]

def jst2utc(jst: datetime): 
    return jst.astimezone(UTC)

def utcstr2datetime(utc_str: str, format='%Y-%m-%d %H:%M:%S'):
    utc = datetime.strptime(utc_str, format)
    utc = utc.replace(tzinfo=UTC)
    return utc

def utc2jst(utc: datetime):
    jst = utc.astimezone(JST)       
    return jst

def to_pydatetime(times, utc_from: datetime, delta_hour_from_gmt):
    if utc_from is None:
        i_begin = 0
    else:
        i_begin = -1
    out = []
    for i, time in enumerate(times):
        if type(time) is str:
            utc = utcstr2datetime(time) - delta_hour_from_gmt
        else:
            server_time = nptimestamp2pydatetime(time)
            utc = server_time - delta_hour_from_gmt
            utc = utc.replace(tzinfo=UTC)
        if utc_from is None:
            out.append(utc)
        else:
            if utc > utc_from:
                if i_begin == -1:
                    i_begin = i
                out.append(utc)
    return (i_begin, len(out), out)
    
def df2dic(df: pd.DataFrame, time_column: str, columns, utc_from: datetime,  delta_hour_from_gmt:timedelta):
    if type(df) == pd.Series:
        return df2dic_one(df, time_column, columns, utc_from, delta_hour_from_gmt)
    i_from, n, utc = to_pydatetime(df[time_column], utc_from, delta_hour_from_gmt)
    if n == 0:
        return (0, {})    
    dic = {}
    dic[time_column] = utc
    jst = [utc2jst(t) for t in utc]
    dic[Columns.JST] = jst
    for column in columns:
        if column != time_column:
            array = list(df[column].values)  
            dic[column] = array[i_from:]
    return (n, dic)

def df2dic_one(df: pd.DataFrame, time_column: str, columns, utc_from: datetime, delta_hour_from_gmt):
    time = df[time_column]
    i_from, n, utc = to_pydatetime([time], utc_from, delta_hour_from_gmt)
    if n == 0:
        return (0, {})    
    dic = {}
    dic[time_column] = utc
    jst = [utc2jst(t) for t in utc]
    dic[Columns.JST] = jst
    for column in columns:
        if column != time_column:
            dic[column] = [df[column]]
    return (n, dic)

class DataBuffer:
    def __init__(self, symbol: str, timeframe: str, length, df: pd.DataFrame, indicator_function, param, delta_hour_from_gmt):
        self.symbol = symbol
        self.timeframe = timeframe        
        self.delta_hour_from_gmt  =  delta_hour_from_gmt 
        n, data = df2dic(df, Columns.TIME, COLUMNS, None, self.delta_hour_from_gmt)
        if n == 0:
            raise Exception('Error cannot get initail data')
        indicator_function(timeframe, data, param)
        data = Utils.sliceDictLast(data, length)
        self.data = data
        self.param = param
        self.indicator_function = indicator_function

    def last_time(self):
        t_utc = self.data[Columns.TIME][-1]
        return t_utc
    
    def last_index(self):
        time = self.data[Columns.TIME]
        return len(time) - 1
    
    def to_int32(self, array):
        n = len(array)
        out = []
        for i in range(n):
            out.append(np.int32(array[i]))
        return out  
    
    def to_int64(self, array):
        n = len(array)
        out = []
        for i in range(n):
            out.append(np.int64(array[i]))
        return out  

    def to_float64(self, array):
        n = len(array)
        out = []
        for i in range(n):
            out.append(np.float64(array[i]))
        return out    
    
    
    def update(self, df: pd.DataFrame):
        last = self.last_time()
        n, dic = df2dic(df, Columns.TIME, COLUMNS, last, self.delta_hour_from_gmt)
        if n == 0:
            return 0
        for key, value in self.data.items():
            if key in dic.keys():
                d = dic[key]
                if isinstance(value[0],  np.int64):
                    d = self.to_int64(d)
                elif isinstance(value[0], np.int32):
                    d = self.to_int32(d)
                elif isinstance(value[0], np.float64):
                    d = self.to_float64(d)
                value += d
            else:
                try:
                    if isinstance(value[0], np.int64):
                        value += np.full(n, 0, dtype=np.int64 )
                    elif isinstance(value[0], np.uint64):
                        value += np.full(n, 0, dtype=np.uint64)
                    elif isinstance(value[0], np.int32):
                        value += np.full(n, 0, dtype=np.int32)
                    else:
                        value += np.full(n, np.nan)
                except Exception as e:
                    print('update()', e)
                    print(value)
        self.indicator_function(self.timeframe, self.data, self.param)
        return n
                
                
def save(data: dict, path: str):
    d = data.copy()
    d[Columns.TIME] = [str(t) for t in d[Columns.TIME]]
    d[Columns.JST] = [str(t) for t in d[Columns.JST]]
    df = pd.DataFrame(d)
    df.to_excel(path, index=False)   
    
    
def test():
    path = '../MarketData/Axiory/NIKKEI/M30/NIKKEI_M30_2023_06.csv'
    df = pd.read_csv(path)

    df1 = df.iloc[:1003, :]
    df2 = df.iloc[1003:, :]
    
    print(len(df))
    print(len(df1))
    print(len(df2))
    params= {'MA':{'window':60}, 'ATR': {'window': 9, 'multiply': 3.0}}
    buffer = DataBuffer('', 'M30', df1, params)
    buffer.update(df2)
    save(buffer.data, './debug/divided.xlsx')
    


if __name__ == '__main__':
    test()
    
       

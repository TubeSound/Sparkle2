# -*- coding: utf-8 -*-
"""
Created: 2022/12/04 22:37:16
Revised: 2023/07/28 08:25
@author: IKU-Trader
"""

import calendar
from dateutil import tz
from datetime import datetime, timedelta, timezone, tzinfo

UTC = tz.gettz('UTC')
JST = tz.gettz('Asia/TOKYO')

class TimeFilter:
    def __init__(self, timezone: tzinfo, begin_hour:int, begin_minute:int, hours:int):
        self.timezone = timezone
        self.begin_hour = begin_hour
        self.begin_minute = begin_minute
        self.hours = hours
        
    def from_to(self, time):
        t = TimeUtils.pyTime(time.year, time.month, time.day, self.begin_hour, self.begin_minute, 0, self.timezone)
        t = t.astimezone(time.tzinfo)
        t = datetime(time.year, time.month, time.day, t.hour, t.minute)
        t = t.replace(tzinfo=time.tzinfo)
        if time < t:
            t -= timedelta(days=1)    
        t1 = t + timedelta(hours=self.hours)
        return t, t1
    
    def on(self, time) :
        t1, t2 = self.from_to(time)
        return (time >= t1 and time <= t2)

    def over(self, time):
        t1, t2 = self.from_to(time)
        return (time > t2)
        
class TimeUtils:
    @staticmethod
    def dayOfLastSunday(year, month):
        '''dow: Monday(0) - Sunday(6)'''
        dow = 6
        n = calendar.monthrange(year, month)[1]
        l = range(n - 6, n + 1)
        w = calendar.weekday(year, month, l[0])
        w_l = [i % 7 for i in range(w, w + 7)]
        return l[w_l.index(dow)]
    
    @staticmethod 
    def dayOfSunday(year, month, num):
        first = datetime(year, month, 1).weekday()
        day = 7 * num - first
        return day
    
    @staticmethod
    def utcnow():
        now = datetime.utcnow()
        now = now.replace(tzinfo=UTC)
        return now
    
    @staticmethod
    def jstnow():
        now = TimeUtils.utcnow()
        return now.astimezone(JST) 
    
    @staticmethod
    def now(tzinfo):
        utc = TimeUtils.utcnow()
        return utc.astimezone(tzinfo)
       
    @staticmethod
    def pyTime(year, month, day, hour, minute, second, tzinfo):
        t = datetime(year, month, day, hour, minute, second)
        return t.astimezone(tzinfo)
           
    @staticmethod
    def utcTime(year, month, day, hour, minute, second):
        t = datetime(year, month, day, hour, minute, second)   
        t = t.replace(tzinfo=UTC)
        return t
    
    @staticmethod
    def awarePytime2naive(time):
        naive = datetime(time.year, time.month, time.day, time.hour, time.minute, time.second)
        return naive

    @staticmethod
    def delta_hour_from_gmt(date_time, begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer):
        if TimeUtils.isSummerTime(date_time, begin_month, begin_sunday, end_month, end_sunday):
            dt = timedelta(hours=delta_hour_from_gmt_in_summer)
        else:
            dt = timedelta(hours=delta_hour_from_gmt_in_summer- 1)
            
        tzinfo = timezone(dt)
        return (dt, tzinfo)
    
    @staticmethod
    def isSummerTime(date_time, begin_month, begin_sunday, end_month, end_sunday):
        day0 = TimeUtils.dayOfSunday(date_time.year, begin_month, begin_sunday)
        tsummer0 = TimeUtils.utcTime(date_time.year, begin_month, day0, 0, 0, 0)
        day1 = TimeUtils.dayOfSunday(date_time.year, end_month, end_sunday)
        tsummer1 = TimeUtils.utcTime(date_time.year, end_month, day1, 0, 0, 0)
        if date_time > tsummer0 and date_time < tsummer1:
            return True
        else:
            return False
    
    @staticmethod
    def isSummerTime2(date_time):
        day0 = TimeUtils.dayOfLastSunday(date_time.year, 3)
        tsummer0 = TimeUtils.utcTime(date_time.year, 3, day0, 0, 0, 0)
        day1 = TimeUtils.dayOfLastSunday(date_time.year, 10)
        tsummer1 = TimeUtils.utcTime(date_time.year, 10, day1, 0, 0, 0)
        if date_time > tsummer0 and date_time < tsummer1:
            return True
        else:
            return False
        
    @staticmethod
    def timestamp2localtime(utc_server, tzinfo=None, adjust_summer_time=True):
        if tzinfo is None:
            t = datetime.fromtimestamp(utc_server)
        else:
            t = datetime.fromtimestamp(utc_server, tzinfo)
        if tzinfo is None or adjust_summer_time == False:
            return t
            
        if TimeUtils.isSummerTime(t):
            dt = 3
        else:
            dt = 2
        t -= timedelta(hours=dt)
        return t
    
    @staticmethod    
    def jst2timestamp(jst):
        timestamp = []
        for t in jst:
            timestamp.append(t.timestamp())
        return timestamp
    
    @staticmethod    
    def jst2utc(jst):
        utc = []
        for t in jst:
            utc.append(t.astimezone(timezone.utc))
        return utc
    
    @staticmethod        
    def numpyDateTime2pyDatetime(np_time):
        py_time = datetime.fromtimestamp(np_time.astype(datetime) * 1e-9)
        return py_time
    
    @staticmethod                
    def sliceTime(pytime_array: list, time_from, time_to):
        begin = None
        end = None
        for i in range(len(pytime_array)):
            t = pytime_array[i]
            if begin is None:
                if t >= time_from:
                    begin = i
            else:
                if t >= time_to:
                    end = i - 1
                    return (end - begin + 1, begin, end)
        if begin is not None:
            end = len(pytime_array) - 1
            return (end - begin + 1, begin, end)
        else:
            return (0, None, None)
        
    @staticmethod
    def slice(dic, time_column, time_from, time_to):
        time = dic[time_column]
        n, begin, end = TimeUtils.sliceTime(time, time_from, time_to)
        if n == 0:
            return n, None
        out = {}
        for key, value in dic.items():
            out[key] = value[begin: end + 1]
        return n, out
            
        
        
def test():
    print(TimeUtils.now(JST), TimeUtils.utcnow())
    jst = TimeUtils.pyTime(2024, 3, 1, 21, 10, 0, JST)
    utc = TimeUtils.utcTime(2024, 3, 1, 12, 10, 0)
    if jst == utc:
        print(jst, utc)
    
    
    
    tzinfo = timezone(timedelta(hours=2))
    now = TimeUtils.now(JST)
    print(now)
    
    now = TimeUtils.now(tzinfo)
    print(now)
    
    t = TimeUtils.pyTime(2023, 2, 1, 13, 0, 5, JST)
    print(t)
    pass

if __name__ == '__main__':
    test()
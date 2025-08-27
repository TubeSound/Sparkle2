import numpy as np
from technical import calc_sma, ATRP, emas, calc_atr, slopes, majority_filter, ema_diff, detect_pivots
from common import Columns, Indicators


class Sparkle2:
    def __init__(self, short_term, mid_term, long_term, pivot_term, pivot_depth_min):
        self.short_term = short_term
        self.mid_term = mid_term
        self.long_term = long_term    
        self.pivot_term = pivot_term
        self.pivot_depth_min = pivot_depth_min
        
    def calc(self, dic):
        cl = dic[Columns.CLOSE]
        self.cl = cl
        self.op = dic[Columns.OPEN]
        self.hi = dic[Columns.HIGH]
        self.lo = dic[Columns.LOW]
        timestamps = dic[Columns.JST]        
        self.short, self.mid, self.long = emas(timestamps, cl, 60 * self.short_term, 60 * self.mid_term, 60 * self.long_term)
        
        self.atrp = ATRP(dic, 14)        
        short_slope = slopes(self.short, 10)
        mid_slope = slopes(self.mid, 10)
        long_slope = slopes(self.long, 10)
        self.short_slope = short_slope
        self.mid_slope = mid_slope
        self.long_slope = long_slope
        
        short_mid_diff_pct, mid_long_diff_pct = ema_diff(self.cl, self.short, self.mid, self.long)
        self.pivot = detect_pivots(mid_long_diff_pct, self.pivot_term, self.pivot_depth_min)
        self.mid_long_diff_pct = mid_long_diff_pct
        
        

        
        

        
        
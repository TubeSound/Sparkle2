import numpy as np
from technical import calc_sma, ATRP, emas, calc_atr, slopes, majority_filter, ema_diff, detect_pivots
from common import Columns, Indicators


class PPP:
    def __init__(self, short_term, mid_term, long_term, slope_th, gap_min):
        self.short_term = short_term
        self.mid_term = mid_term
        self.long_term = long_term    
        self.slope_th = slope_th
        self.gap_min = gap_min
        
    def calc(self, dic):
        cl = dic[Columns.CLOSE]
        self.cl = cl
        self.op = dic[Columns.OPEN]
        self.hi = dic[Columns.HIGH]
        self.lo = dic[Columns.LOW]
        timestamps = dic[Columns.JST]        
        self.short, self.mid, self.long = emas(timestamps, cl, 60 * self.short_term, 60 * self.mid_term, 60 * self.long_term)
        
        self.atrp = ATRP(dic, 14)        
        self.ppp = self.detect_ppp()
        
    def detect_ppp(self):
        short_slope = slopes(self.short, 10)
        mid_slope = slopes(self.mid, 10)
        long_slope = slopes(self.long, 10)
        self.short_slope = short_slope
        self.mid_slope = mid_slope
        self.long_slope = long_slope
        
        short_mid_diff_pct, mid_long_diff_pct = ema_diff(self.cl, self.short, self.mid, self.long)
        self.pivot = detect_pivots(mid_long_diff_pct, 15)
        self.mid_long_diff_pct = mid_long_diff_pct
        
        n = len(self.short)        
        state = np.full(n, 0)
        for i in range(n):
            if (self.short[i] > self.mid[i]) and (self.mid[i] > self.long[i]):
                if short_slope[i] > self.slope_th and mid_slope[i] > self.slope_th and long_slope[i] > self.slope_th:
                    if abs(mid_long_diff_pct[i]) > self.gap_min: 
                        state[i] = 1
            elif (self.short[i] < self.mid[i]) and (self.mid[i] < self.long[i]):
                if short_slope[i] < - self.slope_th and mid_slope[i] < -self.slope_th and long_slope[i] < - self.slope_th:
                    if abs(mid_long_diff_pct[i]) > self.gap_min:
                        state[i] = -1
        out = majority_filter(state, [1, -1], 10)
        return out              
        
        
        

        
        
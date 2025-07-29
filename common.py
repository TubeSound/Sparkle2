# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 19:59:00 2023

@author: docs9
"""
import MetaTrader5 as mt5


HOLD = 0
DOWN = -1
UP = 1
DOWN_TO_UP = 2
UP_TO_DOWN = 3
LOW = -1
HIGH = 1

class Columns:
    TIME = 'time'
    JST = 'jst'
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'    
    ASK = 'ask'
    BID = 'bid'
    MID = 'mid'
    HL2 = 'hl2'
    VOLUME = 'volume'
    TIMESTAMP = 'timestamp'

class Signal:
    LONG = 1
    SHORT = -1    
    
class TimeFrame:
    TICK = 'TICK'
    M1 = 'M1'
    M5 = 'M5'
    M15 = 'M15'
    M30 = 'M30'
    H1 = 'H1'
    H4 = 'H4'
    D1 = 'D1'
    W1 = 'W1'
    
    timeframes = {  M1: mt5.TIMEFRAME_M1, 
                    M5: mt5.TIMEFRAME_M5,
                    M15: mt5.TIMEFRAME_M15,
                    M30: mt5.TIMEFRAME_M30,
                    H1: mt5.TIMEFRAME_H1,
                    H4: mt5.TIMEFRAME_H4,
                    D1: mt5.TIMEFRAME_D1,
                    W1: mt5.TIMEFRAME_W1}
            
    @staticmethod 
    def const(timeframe_str: str):
        return TimeFrame.timeframes[timeframe_str]

class Indicators:
    PROFITS = 'PROFITS'
    PROFITS_MA = 'PROFITS_MA'
    PROFITS_CLOSE = 'PROFITS_CLOSE'
    PROFITS_PEAKS = 'PROFITS_PEAKS'
    
    MA = 'MA'
    MA_SHORT = 'MA_SHORT'
    MA_MID = 'MA_MID'
    MA_LONG = 'MA_LONG'
    MA_LONG_HIGH = 'MA_LONG_HIGH'
    MA_LONG_LOW = 'MA_LONG_LOW'
    
    MA_LONG_SLOPE = 'MA_LONG_SLOPE'
    MA_MID_SLOPE = 'MA_MID_SLOPE'
    MA_SHORT_SLOPE = 'MA_SHORT_SLOPE'
    MA_LONG_TREND = 'MA_LONG_TREND'

    MABAND = 'MABAND'
    MABAND_LONG = 'MABAND_LONG'
    MABAND_SHORT = 'MABAND_SHORT'    
    
    MAGAP = 'MAGAP'
    MAGAP_SLOPE= 'MAGAP_SLOPE'
    MAGAP_ENTRY = 'MAGAP_ENTRY'
    MAGAP_EXIT = 'MAGAP_EXIT'
    
    TR = 'TR'
    ATR = 'ATR'
    ATR_LONG = 'ATR_LONG'
    ATR_UPPER = 'ATR_UPPER'
    ATR_LOWER = 'ATR_LOWER'
    ATRP = 'ATRP'
    ATR_H1 = 'ATR_H1'
    ATRP_H1 = 'ATRP_H1'
    
    DX = 'DX'
    ADX = 'ADX'
    ADX_LONG = 'ADX_LONG'
    DI_PLUS = 'DI_PLUS'
    DI_MINUS = 'DI_MINUS'
    POLARITY = 'POLARITY'
    
    ATR_TRAIL = 'ATR_TRAIL'
    ATR_TRAIL_SIGNAL = 'ATR_TRAIL_SIGNAL'
    ATR_TRAIL_U = 'ATR_TRAIL_U'
    ATR_TRAIL_L = 'ATR_TRAIL_L'
    
    SUPERTREND_MA = 'SUPERTREND_MA'
    SUPERTREND_U = 'SUPERTREND_U'
    SUPERTREND_L = 'SUPERTREND_L'
    SUPERTREND = 'SUPERTREND'
    SUPERTREND_ENTRY = 'SUPERTREND_ENTRY'
    SUPERTREND_EXIT = 'SUPERTREND_EXIT'
    SUPERTREND_UPDATE = 'SUPERTREND_UPDATE'
    
    BB = 'BB'
    BB_MA = 'BB_MA'
    BB_UPPER = 'BB_UPPER'
    BB_LOWER = 'BB_LOWER'
    BB_UP = 'BB_UP'
    BB_DOWN = 'BB_DOWN'
    BB_CROSS = 'BB_CROSS'
    BB_CROSS_UP = 'BB_CROSS_UP'
    BB_CROSS_DOWN = 'BB_CROSS_DOWN'
    
    TREND_ADX_DI ='TREND_ADX_DI'
    
    BBRATE = 'BBRATE'
    VWAP = 'VWAP'
    VWAP_RATE = 'VWAP_RATE'
    VWAP_SLOPE = 'VWAP_SLOPE'
    VWAP_U = 'VWAP_U'
    VWAP_L = 'VWAP_L'
    VWAP_PROB = 'VWAP_PROB'
    VWAP_DOWN = 'VWAP_DOWN'
    VWAP_CROSS_DOWN = 'VWAP_CROSS_DOWN'
    VWAP_RATE_SIGNAL = 'VWAP_RATE_SIGNAL'
    VWAP_PROB_SIGNAL = 'VWAP_PROB_SIGNAL'
    
    RCI = 'RCI'
    RCI_SIGNAL = 'RCI_SIGNAL'
    
    FILTER_MA = 'FILTER_MA'
    
    PPP_ENTRY = 'PPP_ENTRY'
    PPP_EXIT = 'PPP_EXIT'
    PPP_UP = 'PPP_UP'
    PPP_DOWN = 'PPP_DOWN'
    
    MA_GOLDEN_CROSS = 'MA_GOLDEN_CROSS'
    
    BREAKOUT = 'BREAKOUT'
    BREAKOUT_ENTRY = 'BREAKOUT_ENTRY'
    BREAKOUT_EXIT = 'BREAKOUT_EXIT' 
    BREAKOUT_SL = 'BREAKOUT_SL' 
    
    RALLY = 'RALLY'
    RALLY_ENTRY = 'RALLY_ENTRY'
    RALLY_EXIT = 'RALLY_EXIT'
    
    SQUEEZER = 'SQUEEZER'
    SQUEEZER_STD = 'SQUEEZER_STD'
    SQUEEZER_ATR = 'SQUEEZER_ATR'
    SQUEEZER_ENTRY = 'SQUEEZER_ENTRY'
    SQUEEZER_UPPER = 'SQUEEZER_UPPER'
    SQUEEZER_UPPER_SL = 'SQUEEZER_UPPER_SL'
    SQUEEZER_LOWER = 'SQUEEZER_LOWER'
    SQUEEZER_LOWER_SL = 'SQUEEZER_LOWER_SL'
    SQUEEZER_STATUS = 'SQUEEZER_STATUS'
    SQUEEZER_SIGNAL = 'SQUEEZER_SIGNAL'
    SQUEEZER_ENTRY = 'SQUEEZER_ENTRY'
    SQUEEZER_EXIT = 'SQUEEZER_EXIT'
    
    ANKO = 'ANKO'
    ANKO_ENTRY = 'ANKO_ENTRY'
    ANKO_EXIT = 'ANKO_EXIT'
    ANKO_PROFIT = 'ANKO_PROFIT'
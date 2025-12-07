from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Dict
import pandas as pd
import numpy as np
import MetaTrader5 as mt5api

class Signal:
    LONG = 1
    SHORT = -1
    CLOSE = 2

    order_types = {
        mt5api.ORDER_TYPE_BUY: 'Market Buy order',
        mt5api.ORDER_TYPE_SELL: 'Market Sell order',
        mt5api.ORDER_TYPE_BUY_LIMIT: 'Buy Limit pending order',
        mt5api.ORDER_TYPE_SELL_LIMIT: 'Sell Limit pending order',
        mt5api.ORDER_TYPE_BUY_STOP: 'Buy Stop pending order',
        mt5api.ORDER_TYPE_SELL_STOP: 'Sell Stop pending order',
        mt5api.ORDER_TYPE_BUY_STOP_LIMIT: 'Upon reaching the order price, a pending Buy Limit order is placed at the StopLimit price',
        mt5api.ORDER_TYPE_SELL_STOP_LIMIT: 'Upon reaching the order price, a pending Sell Limit order is placed at the StopLimit price',
        mt5api.ORDER_TYPE_CLOSE_BY: 'Order to close a position by an opposite one'
    }

    @staticmethod
    def order_type2signal(type):
        if type == mt5api.ORDER_TYPE_BUY:
            return Signal.LONG
        elif type == mt5api.ORDER_TYPE_BUY_LIMIT:
            return Signal.LONG
        elif type == mt5api.ORDER_TYPE_BUY_STOP:
            return Signal.LONG
        elif type == mt5api.ORDER_TYPE_BUY_STOP_LIMIT:
            return Signal.LONG
        elif type == mt5api.ORDER_TYPE_SELL:
            return Signal.SHORT
        elif type == mt5api.ORDER_TYPE_SELL_LIMIT:
            return Signal.SHORT
        elif type == mt5api.ORDER_TYPE_SELL_STOP:
            return Signal.SHORT
        elif type == mt5api.ORDER_TYPE_SELL_STOP_LIMIT:
            return Signal.SHORT
        else:
            return None

class Param:
    sl_mode = 'fix' # fix/atr
    sl_value = None
    stop_cond = 'close' # close/moment
    trail_hit_price = None
    trail_stop_percent = None
    neg_bar_max = 0
    position_max = 5
    volume = 0.01
    
class PositionInfo:

    TAKE_PROFIT = 1
    STOP_LOSS = 2
    TIMEUP = 3
    REVERSAL = 4
    FORCE_CLOSE = 5
    TRAILING_STOP = 6
    TRAILING_HARD_STOP = 7

    close_reason = {TAKE_PROFIT: 'TP',
                    STOP_LOSS:'SL',
                    TIMEUP: 'TIMEUP',
                    REVERSAL: "REVERSAL",
                    FORCE_CLOSE:'FORCE',
                    TRAILING_STOP: 'TRAIL_STOP',
                    TRAILING_HARD_STOP: 'TRAIL_HARD',       
                    }

    def __init__(self, symbol, order_type: int, time: datetime, volume, ticket, price, sl, tp, param:Param):
        self.symbol = symbol
        self.order_type = order_type
        self.order_signal = Signal.order_type2signal(order_type)
        self.volume = volume
        self.ticket = ticket
        self.entry_time = time
        self.entry_price = float(price)
        self.sl = sl
        self.tp = tp
        self.param = param
        self.exit_time = None
        self.exit_price = None
        self.profit = None
        self.profit_max = None
        self.trail_hit = False
        self.neg_bar_count = 0
        self.closed = False
        self.reason = None


    def is_sl(self, price_high, price_low, price_close):
        if self.sl_price is None:
            return False
        if self.order_signal == Signal.LONG:
            if self.param.stop_cond == 'close':
                return (price_close < self.sl_price)               
            elif self.param.stop_cond == 'moment':
                return (self.sl_price < price_low)
            else:
                raise Exception('Bad sl cond')
        elif self.order_signal == Signal.SHORT:
            if self.param.stop_cond == 'close':
                return (price_close > self.sl_price)
            elif self.param.stop_cond == 'moment':
                return (self.sl_price > price_high)
            else:
                raise Exception('Bad sl_cond')
        return False
    
    def is_trail_stop(self, hi, lo, cl):
        if self.param.trail_hit_price is None:
            return False
        profit = cl - self.entry_price
        if self.order_signal == Signal.SHORT:
            profit *= -1
        if self.trail_hit:
            if profit > self.profit_max:
                self.profit_max = profit
            else:
                d = abs((profit - self.profit_max) / self.profit_max) * 100.0
                if d > self.param.trail_stop_percent:
                    return True
        else:
            if profit >= self.param.trail_hit_price:
                self.trail_hit = True
                self.profit_max = profit 
        return False   

    # 連続損出検出
    def neg_bars(self,  hi, lo, cl):
        if self.param.neg_bar_max is None:
            return False
        profit = cl - self.entry_price
        if self.order_signal == Signal.SHORT:
            profit *= -1
        if profit < 0:
            self.neg_bar_count += 1
            if self.neg_bar_count > self.param.neg_bar_max:
                return True
        else:
            self.neg_bar_count = 0
        return False
    
    def desc(self):
        type_str = Signal.order_types[self.order_type]
        s = 'symbol: ' + self.symbol + ' type: ' + type_str + ' volume: ' + str(self.volume) + ' ticket: ' + str(self.ticket)
        return s

    def array(self):
        data = [self.symbol, self.order_type, self.volume, self.ticket, self.sl, self.tp, self.entry_time, self.entry_price, self.exit_time, self.exit_price, self.profit, self.closed, self.reason]
        return data

    @staticmethod
    def array_columns():
        return ['symbol', 'type', 'volume', 'ticket', 'sl', 'tp', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'profit', 'closed', 'reason']

class TradeManager:
    def __init__(self, symbol, timeframe, param: Param):
        self.symbol = symbol
        self.timeframe = timeframe
        self.param = param
        self.positions: Dict[int, PositionInfo] = {}
        self.positions_closed: Dict[int, PositionInfo] = {}


    def history_df(self):
        df = pd.DataFrame(data=self.pnl_history, columns=['Time', 'Profit(Unreal)', 'Profit(Realized)', 'count', 'win_rate', 'reason'])
        return df

    def add_position(self, position: PositionInfo):
        self.positions[position.ticket] = position

    def move_to_closed(self, ticket):
        if ticket in self.positions.keys():
            pos = self.positions.pop(ticket)
            self.positions_closed[ticket] = pos
        else:
            print('move_to_closed, No tickt')

    def calc_profit(self, price):
        current = 0
        win_num = 0
        for ticket, position in self.positions.items():
            if position.order_signal == Signal.LONG:
                profit = price - position.entry_price
            else:
                profit = position.entry_price - price
            current += profit
            if profit > 0:
                win_num += 1
        count = len(self.positions.items())
        if count == 0:
            win_rate = 0.0
        else:
            win_rate = float(win_num) / float(count)
        closed = 0.0
        for ticket, position in self.positions_closed.items():
            if position.profit is not None:
                closed += position.profit
        return current + closed, current, closed, count, win_rate

    def summary(self):
        out = []
        columns = PositionInfo.array_columns()
        for ticket, pos in list(self.positions.items()) + list(self.positions_closed.items()):
            d = pos.array()
            out.append(d)
        return out, columns

    def remove_positions(self, tickets):
        for ticket in tickets:
            self.move_to_closed(ticket)

    def open_positions(self):
        return self.positions

    def untrail_positions(self):
        positions = {}
        for ticket, position in self.positions.items():
            if position.profit_max is None:
                positions[ticket] = position
        return positions

    def remove_position_auto(self, mt5_positions):
        remove_tickets = []
        for ticket, info in self.positions.items():
            found = False
            for position in mt5_positions:
                if position.ticket == ticket:
                    found = True
                    break
            if not found:
                remove_tickets.append(ticket)
        if len(remove_tickets):
            self.remove_positions(remove_tickets)
            print('<Closed by Meta Trader Stoploss or Takeprofit> ', self.symbol, 'tickets:', remove_tickets)

    def df_position(self):
        data = []
        columns = None
        for ticket, position in self.positions.items():
            r = position.array()
            columns = PositionInfo.array_columns()
            data.append(r)
        for ticket, position in self.positions_closed.items():
            r = position.array()
            columns = PositionInfo.array_columns()
            data.append(r)
        df = pd.DataFrame(data=data, columns=columns if columns else PositionInfo.array_columns())
        return df

    def update(self, time, op, hi, lo, cl):
        # check stop loss
        
        
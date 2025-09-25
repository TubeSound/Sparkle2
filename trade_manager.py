from datetime import datetime
import pandas as pd
import numpy as np
import MetaTrader5 as mt5api


class Signal:
    LONG = 1
    SHORT = -1    
    CLOSE = 2
    
    order_types = {
                mt5api.ORDER_TYPE_BUY:'Market Buy order',
                mt5api.ORDER_TYPE_SELL: 'Market Sell order',
                mt5api.ORDER_TYPE_BUY_LIMIT: 'Buy Limit pending order',
                mt5api.ORDER_TYPE_SELL_LIMIT: 'Sell Limit pending order',
                mt5api.ORDER_TYPE_BUY_STOP: 'Buy Stop pending order',
                mt5api.ORDER_TYPE_SELL_STOP:'Sell Stop pending order',
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
        

    
class PositionInfo:

    TAKE_PROFIT = 1
    STOP_LOSS = 2
    TRAILING_STOP = 3
    TIMEUP = 4
    FORCE_CLOSE = 5
    
    close_reason = {TAKE_PROFIT: 'TP', STOP_LOSS:'SL',  TRAILING_STOP: 'TRAIL', TIMEUP: 'TIMEUP', FORCE_CLOSE:'FORCE'}
        
    def __init__(self, symbol, order_type: int, time: datetime, volume, ticket, price, sl, tp, target_profit=0):
        self.symbol = symbol
        self.order_type = order_type
        self.order_signal = Signal.order_type2signal(order_type)
        self.volume = volume
        self.ticket = ticket
        self.entry_time = time
        self.entry_price = float(price)
        self.sl = sl        
        self.tp = tp
        if self.order_signal == Signal.LONG:
            self.sl_price =  None if sl is None else (price - sl)
            self.tp_price = None if tp is None else (price + tp)
        elif self.order_signal == Signal.SHORT:
            self.sl_price =  None if sl is None else (price + sl)
            self.tp_price = None if tp is None else (price - tp)
        self.target_profit = target_profit

        self.exit_time = None       
        self.exit_price = None
        self.profit = None
        self.profit_max = None
        self.closed = False
        self.reason = None
        
    def is_tp(self, price_low, price_high):
        if self.order_signal == Signal.LONG:
            if self.tp_price <=  price_high:
                return True
            else:
                return False
        elif self.order_signal == Signal.SHORT:
            if self.tp_price >= price_low:
                return True
            else:
                return False
            
    def is_sl(self, price_low, price_high):
        if self.order_signal == Signal.LONG:
            if self.sl_price > price_low:
                return True
            else:
                return False
        elif self.order_signal == Signal.SHORT:
            if self.sl_price < price_high:
                return True
            else:
                return False
 
    def desc(self):
        type_str = Signal.order_types[self.order_type]
        s = 'symbol: ' + self.symbol + ' type: ' + type_str + ' volume: ' + str(self.volume) + ' ticket: ' + str(self.ticket)
        return s
    
    def array(self):
        columns = ['symbol', 'type', 'volume', 'ticket', 'sl', 'tp', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'profit', 'closed', 'reason']
        data = [self.symbol, self.order_type, self.volume, self.ticket, self.sl, self.tp, self.entry_time, self.entry_price, self.exit_time, self.exit_price, self.profit, self.closed, self.reason]
        return data, columns


class TradeManager:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.positions = {}
        self.positions_closed = {}

    def add_position(self, position: PositionInfo):
        self.positions[position.ticket] = position
       
    def move_to_closed(self, ticket):
        if ticket in self.positions.keys():
            pos = self.positions.pop(ticket)
            self.positions_closed[ticket] = pos
        else:
            print('move_to_closed, No tickt')
        
    def summary(self):
        out = []
        for ticket, pos in list(self.positions.items()) + list(self.positions_closed.items()):
            d, columns = pos.array()
            out.append(d)
        df = pd.DataFrame(data=out, columns=columns)
        return df

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
            if found == False:
                remove_tickets.append(ticket)
        if len(remove_tickets):
            self.remove_positions(remove_tickets)    
            print('<Closed by Meta Trader Stoploss or Takeprofit> ', self.symbol, 'tickets:', remove_tickets)
            
            
    def df_position(self):
        data = []
        for ticket, position in self.positions.items():
            r, columns = position.array()
            data.append(r)
        for ticket, position in self.positions_closed.items():
            r, columns = position.array()
            data.append(r)
        df = pd.DataFrame(data=data, columns=columns)
        return df
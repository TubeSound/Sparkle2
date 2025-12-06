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

    def __init__(self, symbol, order_type: int, time: datetime, volume, ticket, price, sl, tp, target_profit=0):
        self.symbol = symbol
        self.order_type = order_type
        self.order_signal = Signal.order_type2signal(order_type)
        self.volume = volume
        self.ticket = ticket
        self.entry_time = time
        self.entry_price = float(price)
        self.sl = sl
        self.sl_updated = False
        self.tp = tp
        if self.order_signal == Signal.LONG:
            self.sl_price =  None if sl is None else (price - sl)
            self.tp_price = None if tp is None else (price + tp)
        elif self.order_signal == Signal.SHORT:
            self.sl_price =  None if sl is None else (price + sl)
            self.tp_price = None if tp is None else (price - tp)
        else:
            self.sl_price = None
            self.tp_price = None
        self.target_profit = target_profit

        self.exit_time = None
        self.exit_price = None
        self.profit = None
        self.profit_max = None
        self.closed = False
        self.reason = None

    def is_tp(self, price_low, price_high):
        if self.order_signal == Signal.LONG:
            return self.tp_price is not None and (self.tp_price <= price_high)
        elif self.order_signal == Signal.SHORT:
            return self.tp_price is not None and (self.tp_price >= price_low)
        return False

    def is_sl(self, price_high, price_low, price_close, how):
        if self.sl_price is None:
            return False
        if self.order_signal == Signal.LONG:
            if how == 'close':
                return (price_close < self.sl_price)               
            elif how == 'moment':
                return (self.sl_price < price_low)
            else:
                raise Exception('Bad sl cond')
        elif self.order_signal == Signal.SHORT:
            if how == 'close':
                return (price_close > self.sl_price)
            elif how == 'moment':
                return (self.sl_price > price_high)
            else:
                raise Exception('Bad sl_cond')
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

# ===== 追加: 合計(未実現のみ)トレーリング用設定 =====
@dataclass
class TrailingConfig:
    enabled: bool = True
    mode: str = "abs"                  # "abs" or "pct"
    start_trigger: float = 0.0         # トレール開始の基準（current_unrealがこれを超えるまで待つ）
    distance: float = 50.0             # ドローダウン許容（abs or pct）
    step_lock: Optional[float] = None  # ラチェット幅
    min_positions: int = 1
    close_all: bool = True

    # マイナス領域耐性
    neg_grace_bars: int = 0                 # current_unreal<0 の連続バー数までは“無条件に耐える”
    neg_hard_stop: Optional[float] = None   # current_unreal <= -この値 で強制クローズ（abs推奨）
    activate_from: str = "breakeven"        # "breakeven"（0超えたら発動） or "rebound"
    rebound_from_trough: float = 0.0        # activate_from="rebound"時: トラフからの改善量（abs or pct）

class TradeManager:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.positions: Dict[int, PositionInfo] = {}
        self.positions_closed: Dict[int, PositionInfo] = {}

        # === ここから追記: 合計(未実現)トレーリングの状態 ===
        self.trailing: Optional[TrailingConfig] = None
        # (time, current_unreal, realized, count, win_rate, reason)
        self.pnl_history: List[Tuple[datetime, float, float, int, float, str]] = []
        self.equity_peak: float = 0.0
        self.equity_peak_time: Optional[datetime] = None
        self.lock_line: Optional[float] = None  # ここを下回ったら一括クローズ

        # マイナス領域の状態
        self.neg_bars: int = 0
        self.unreal_trough: float = 0.0
        self.trailing_activated: bool = False

    def history_df(self):
        df = pd.DataFrame(data=self.pnl_history, columns=['Time', 'Profit(Unreal)', 'Profit(Realized)', 'count', 'win_rate', 'reason'])
        return df

    # ====== 既存メソッド ======
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

    # ====== ここから追記：トレーリング設定と評価 ======
    def set_trailing(self,
                     enabled: bool = True,
                     mode: str = "abs",
                     start_trigger: float = 0.0,
                     distance: float = 50.0,
                     step_lock: Optional[float] = None,
                     min_positions: int = 1,
                     close_all: bool = True,
                     neg_grace_bars: int = 0,
                     neg_hard_stop: Optional[float] = None,
                     activate_from: str = "breakeven",
                     rebound_from_trough: float = 0.0):
        """
        合計(未実現P/Lのみ)トレーリングの設定を更新。
        mode: "abs" or "pct"
        start_trigger: current_unreal がこの値を超えるとトレーリング開始（breakeven起動の場合）
        distance: ドローダウン許容幅（abs: 同一単位 / pct: %）
        step_lock: これを超えるたびにロックラインを引き上げる（abs/pct）
        neg_grace_bars: current_unreal<0 の連続バー数までは耐える
        neg_hard_stop: current_unreal が -値 に達したら強制撤退（安全網）
        activate_from: "breakeven" = 0超えたら発動, "rebound" = トラフからの戻りで発動
        rebound_from_trough: activate_from="rebound"時の必要戻り量（abs or pct）
        """
        self.trailing = TrailingConfig(
            enabled=enabled, mode=mode, start_trigger=start_trigger,
            distance=distance, step_lock=step_lock, min_positions=min_positions,
            close_all=close_all, neg_grace_bars=neg_grace_bars,
            neg_hard_stop=neg_hard_stop, activate_from=activate_from,
            rebound_from_trough=rebound_from_trough,
        )
        self.equity_peak = 0.0
        self.equity_peak_time = None
        self.lock_line = None
        self.pnl_history.clear()
        self.neg_bars = 0
        self.unreal_trough = 0.0
        self.trailing_activated = False

    def _distance_to_value(self, peak: float) -> float:
        """distance を実数に変換（pctなら%→実数化、absならそのまま）"""
        if self.trailing is None:
            return 0.0
        if self.trailing.mode == "pct":
            return peak * (self.trailing.distance / 100.0)
        return self.trailing.distance

    def judge_stop(self,
                  price: float,
                  now: Optional[datetime] = None,
                  close_fn: Optional[Callable[[dict], List[int]]] = None):
        """
        毎ティック/バーで呼び出し。
        - 未実現損益(current_unreal)の推移を見てトレーリング条件を評価
        - close_fn: 実クローズを行うコールバック
            * 引数: {"positions": Dict[ticket, PositionInfo], "symbol": str}
            * 戻り: 実際にクローズできた tickets(list)
        """
        if now is None:
            now = datetime.now()

        # P/L計測（currentのみ使用）
        total, current_unreal, realized, count, win_rate = self.calc_profit(price)
        history = [now, current_unreal, realized, count, win_rate, ""]

        if not self.trailing or not self.trailing.enabled:
            self.pnl_history.append(history)
            return False
        if count < self.trailing.min_positions:
            self.pnl_history.append(history)
            return False

        cfg = self.trailing

        # --- マイナス領域の処理 ---
        if current_unreal < 0:
            # 連続マイナス本数カウント
            self.neg_bars += 1
            # トラフ更新
            if (self.neg_bars == 1) or (current_unreal < self.unreal_trough):
                self.unreal_trough = current_unreal

            # ハードストップ（絶対額のみ想定）
            if cfg.neg_hard_stop is not None and current_unreal <= -abs(cfg.neg_hard_stop):
                self._reset_trailing_negative_state(current_unreal, now)
                history[-1] = PositionInfo.close_reason[PositionInfo.TRAILING_HARD_STOP]
                self.pnl_history.append(history)
                return True

            # 猶予：neg_grace_bars 以内は耐える
            if self.neg_bars <= max(0, cfg.neg_grace_bars):
                self.pnl_history.append(history)
                return False

            # 発動条件をチェック（activate_from）
            if cfg.activate_from == "breakeven":
                # 0を超えるまでトレーリングは発動しない
                self.pnl_history.append(history)
                return False
            else:
                # "rebound": トラフからの一定改善で発動
                if cfg.mode == "pct":
                    need = abs(self.unreal_trough) * (cfg.rebound_from_trough / 100.0)
                else:
                    need = cfg.rebound_from_trough
                improved = current_unreal - self.unreal_trough
                if improved < need:
                    self.pnl_history.append(history)
                    return False
                # 改善量クリア → 発動OK
                self.trailing_activated = True

        else:
            # 非マイナス帯に出たら連続マイナスをリセット
            self.neg_bars = 0
            self.unreal_trough = 0.0
            # breakeven起動の場合、0以上 かつ start_trigger クリアで発動
            if cfg.activate_from == "breakeven" and current_unreal >= cfg.start_trigger:
                self.trailing_activated = True

        # まだ発動していないなら何もしない
        if not self.trailing_activated:
            self.pnl_history.append(history)
            return False

        # ここから通常のトレーリング（current_unrealのみ）
        # start_trigger も満たしているか最終確認
        if current_unreal < cfg.start_trigger:
            self.pnl_history.append(history)
            return False

        # ピーク・ロックライン更新
        if current_unreal > self.equity_peak:
            self.equity_peak = current_unreal
            self.equity_peak_time = now

        dd_allow = self._distance_to_value(self.equity_peak)
        new_lock = self.equity_peak - dd_allow

        if self.lock_line is None:
            self.lock_line = new_lock
        else:
            if new_lock > self.lock_line:
                self.lock_line = new_lock
            if cfg.step_lock is not None:
                step_val = (self.equity_peak * (cfg.step_lock/100.0)) if cfg.mode=="pct" else cfg.step_lock
                while (self.equity_peak - self.lock_line) > (dd_allow + step_val):
                    self.lock_line += step_val

        # 発火：current_unreal がロックライン割れ
        if current_unreal <= (self.lock_line if self.lock_line is not None else -1e18):
            # 再起動に備えて軽くリセット
            self.lock_line = None
            self.equity_peak = current_unreal
            self.equity_peak_time = now
            self.trailing_activated = False
            self.neg_bars = 0
            self.unreal_trough = 0.0
            history[-1] = PositionInfo.close_reason[PositionInfo.TRAILING_STOP]
            self.pnl_history.append(history)
            return True
        else:
            self.pnl_history.append(history)
            return False

    def _reset_trailing_negative_state(self, current_unreal: float, now: datetime):
        self.lock_line = None
        self.equity_peak = current_unreal
        self.equity_peak_time = now
        self.trailing_activated = False
        self.neg_bars = 0
        self.unreal_trough = 0.0

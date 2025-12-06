import os
import sys
sys.path.append('../Libraries/trade')
import pickle

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import threading
import numpy as np
import pandas as pd
from dateutil import tz
from datetime import datetime, timedelta, timezone
from mt5_trade import Mt5Trade, Columns
from MetaTrader5 import TradePosition
import sched

from trade_manager import TradeManager, PositionInfo, Signal
from time_utils import TimeUtils
from utils import Utils
from common import Indicators

from maron_pie import MaronPie, MaronPieParam
from montblanc import Montblanc, MontblancParam

AXIORY_SUMMER_TIME_BEGIN_MONTH = 3
AXIORY_SUMMER_TIME_BEGIN_WEEK = 2
AXIORY_SUMMER_TIME_END_MONTH = 11
AXIORY_SUMMER_TIME_END_WEEK = 1
AXIORY_DELTA_TIME_TO_UTC = 3.0



JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')  

import logging
os.makedirs('./log', exist_ok=True)
log_path = './log/trade_' + datetime.now().strftime('%y%m%d_%H%M') + '.log'
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p"
)

# -----

scheduler = sched.scheduler()

# -----
def utcnow():
    utc = datetime.now(UTC)
    return utc

def utc2localize(aware_utc_time, timezone):
    t = aware_utc_time.astimezone(timezone)
    return t

def is_market_open(mt5, timezone, symbol='USDJPY'):
    now = utcnow()
    t = utc2localize(now, timezone)
    t -= timedelta(seconds=5)
    df = mt5.get_ticks_from(symbol, t, length=100)
    return (len(df) > 0)



@dataclass
class Metrics:
    n: int
    win_rate: float
    profit_sum: float       # 含みP/L合計（未実現）
    profit_value_sum: float # 将来的にロット換算などしたい場合の拡張用（今は profit_sum と同じでOK）


class Trailer:
    """
    チケット（取引）ごとにトレーリングストップを行うクラス。
    - 評価対象は「未実現P/L（position.profit）」のみ
    - 含み損の猶予、ハードストップ、トラフからのリバウンド起動などをサポート
    """

    def __init__(self, strategy: str, symbol: str):
        self.strategy = strategy
        self.symbol = symbol
        self.history = []
        mt5 = Mt5Trade( AXIORY_SUMMER_TIME_BEGIN_MONTH,
                        AXIORY_SUMMER_TIME_BEGIN_WEEK,
                        AXIORY_SUMMER_TIME_END_MONTH,
                        AXIORY_SUMMER_TIME_END_WEEK,
                        AXIORY_DELTA_TIME_TO_UTC)
         
        mt5.set_symbol(symbol)
        self.mt5 = mt5

        # 必要なら口座/サーバなどの接続情報をここで設定
        # self.mt5.login(...)

        # === トレーリング設定（全チケット共通） ===
        self.trail_enabled: bool = True
        self.trail_mode: str = "abs"          # "abs" or "pct"
        self.trail_start_trigger: float = 500  # 起動条件: 含み益がこの値以上
        self.trail_distance: float = 20.0      # ピークからの許容ドローダウン（abs or pct）
        self.trail_step_lock: Optional[float] = None  # ラチェット幅（abs or pct）
        self.trail_min_positions: int = 1     # 何ポジ以上あればトレール開始するか

        # 含み損ゾーンでの耐性
        self.neg_grace_bars: int = 0          # 含み損連続Nバーまでは無条件で耐える
        self.neg_hard_stop: Optional[float] = 500  # 含み損が -値 に達したら即クローズ

        # 起動モード
        # "breakeven": 含み益が 0 を超え、かつ start_trigger を超えたら起動
        # "rebound" : 含み損トラフから rebound_from_trough だけ戻したら起動
        self.activate_from: str = "breakeven"   # "breakeven" or "rebound"
        self.rebound_from_trough: float = 0.0   # reboud モード時、トラフからの戻り幅（abs or pct）

        # === 旧・合計トレール互換用（使わなくてもよい） ===
        self.trailing_activated: bool = False
        self.equity_peak: float = 0.0
        self.equity_peak_time: Optional[datetime] = None
        self.lock_line: Optional[float] = None
        self.neg_bars: int = 0
        self.unreal_trough: float = 0.0

        # --- チケット単位のトレーリング状態 ---
        # ticket -> {activated:bool, peak:float, lock:float|None, neg_bars:int, trough:float, peak_time:datetime|None}
        self.ticket_state: Dict[int, Dict[str, Any]] = {}

        # === ログ ===
        self.history: List[Dict[str, Any]] = []

    # ============================================================
    #  MT5関連ヘルパ
    # ============================================================
    def get_positions(self) -> Dict[int, Any]:
        """
        指定シンボルのポジションを取得し、 ticket -> position の辞書にして返す。
        position は MetaTrader5 の TradePosition 構造体を想定。
        """
        positions = self.mt5.get_positions(self.symbol)
        if positions is None:
            return {}
        return {p.ticket: p for p in positions}

    def calc_metrics(self, positions: Dict[int, Any]) -> Metrics:
        """
        ポジション全体の簡易メトリクスを計算。
        - profit_sum: 各ポジションの profit を単純合計（未実現）
        - win_rate : profit>0 のポジション割合
        """
        n = len(positions)
        if n == 0:
            return Metrics(n=0, win_rate=0.0, profit_sum=0.0, profit_value_sum=0.0)

        profits = [float(p.profit) for p in positions.values()]
        profit_sum = float(sum(profits))
        win_num = sum(1 for p in profits if p > 0)
        win_rate = win_num / n if n > 0 else 0.0

        # 拡張用に別名も用意（今は同じ値）
        profit_value_sum = profit_sum

        return Metrics(
            n=n,
            win_rate=win_rate,
            profit_sum=profit_sum,
            profit_value_sum=profit_value_sum,
        )

    def append_to_history(self, t: datetime, judge: int, metrics: Metrics):
        """
        シンプルな履歴: 日時・judgeフラグ・メトリクス。
        judge:
            0 = 何もせず
            1 = トレーリング決済した
        """
        self.history.append(
            {
                "time": t.isoformat(),
                "judge": judge,
                "n": metrics.n,
                "win_rate": metrics.win_rate,
                "profit_sum": metrics.profit_sum,
                "profit_value_sum": metrics.profit_value_sum,
            }
        )

    def hisotry_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def get_path(self) -> str:
        """
        ログCSVの保存先パス。必要に応じて変更してください。
        """
        return f"{self.strategy}_{self.symbol}_trailing_history.csv"

    # ============================================================
    #  トレーリング：チケット単位
    # ============================================================
    def update(self):
        """
        毎バー/毎ティックで呼び出し想定。
        - 現在のポジション取得
        - チケット単位トレール判定
        - 必要ならクローズ
        - 履歴ログ
        """
        dic = self.get_positions()
        metrics = self.calc_metrics(dic)
        if self.trail_enabled:
            pos_to_close = self.judge_trailing_stop(dic)
            for pos in pos_to_close:
                self.mt5.close_position(pos)

            judge_flag = 1 if pos_to_close else 0
            self.append_to_history(datetime.now(tz=JST), judge_flag, metrics)

        try:
            df = self.hisotry_df()
            df.to_csv(self.get_path(), index=False)
        except Exception:
            # ログ保存失敗してもトレールは継続
            pass



    # --- コアロジック：チケット単位トレーリング ---
    def judge_trailing_stop(self, positions: Dict[int, Any]) -> []:
        """
        チケット単位のトレーリングストップ判定。
        - 入力: ticket -> position (MetaTrader5 position)
        - 出力: 今回クローズすべき ticket のリスト
        """
        if not self.trail_enabled:
            return []

        tickets = list(positions.keys())
        if len(tickets) < self.trail_min_positions:
            # 旧・合計トレール状態は掃除しておく（互換用）
            self.neg_bars = 0
            self.unreal_trough = 0.0
            self.trailing_activated = False
            self.lock_line = None
            self.equity_peak = 0.0
            return []

        # 開いていないチケットの状態をGC
        open_set = set(tickets)
        for tk in list(self.ticket_state.keys()):
            if tk not in open_set:
                del self.ticket_state[tk]

        now = datetime.now(tz=JST)
        to_close = []

        for tk in tickets:
            pos = positions[tk]
            cur = float(pos.profit)  # チケット単位の未実現P/L

            st = self.ticket_state.get(
                tk,
                {
                    "activated": False,
                    "peak": 0.0,
                    "peak_time": None,
                    "lock": None,
                    "neg_bars": 0,
                    "trough": 0.0,
                },
            )

            # ---------- 含み損領域の処理 ----------
            if cur < 0:
                st["neg_bars"] += 1
                if st["neg_bars"] == 1 or cur < st["trough"]:
                    st["trough"] = cur

                # ハードストップ
                if self.neg_hard_stop is not None and cur <= -abs(self.neg_hard_stop):
                    st["activated"] = False
                    st["peak"] = cur
                    st["peak_time"] = now
                    st["lock"] = None
                    st["neg_bars"] = 0
                    st["trough"] = 0.0
                    to_close.append(pos)
                    self.ticket_state[tk] = st
                    continue

                # 猶予（耐える）
                if st["neg_bars"] <= max(0, self.neg_grace_bars):
                    self.ticket_state[tk] = st
                    continue

                # 起動条件
                if self.activate_from == "breakeven":
                    # 0を超えるまでは起動しない
                    self.ticket_state[tk] = st
                    continue
                else:
                    # "rebound" モード: トラフからの戻り量で判定
                    if self.trail_mode == "pct":
                        need = abs(st["trough"]) * (self.rebound_from_trough / 100.0)
                    else:
                        need = self.rebound_from_trough
                    improved = cur - st["trough"]
                    if improved < need:
                        self.ticket_state[tk] = st
                        continue
                    st["activated"] = True

            else:
                # 非マイナス帯：neg系はリセット、breakeven起動の判断
                st["neg_bars"] = 0
                st["trough"] = 0.0
                if self.activate_from == "breakeven" and cur >= self.trail_start_trigger:
                    st["activated"] = True

            # まだ起動していない
            if not st["activated"]:
                self.ticket_state[tk] = st
                continue

            # start_trigger も満たすこと（再起動などの整合）
            if cur < self.trail_start_trigger:
                self.ticket_state[tk] = st
                continue

            # ---------- ピーク・ロックライン更新 ----------
            if cur > st["peak"]:
                st["peak"] = cur
                st["peak_time"] = now

            if self.trail_mode == "pct":
                dd_allow = st["peak"] * (self.trail_distance / 100.0)
            else:
                dd_allow = self.trail_distance

            new_lock = st["peak"] - dd_allow

            if st["lock"] is None:
                st["lock"] = new_lock
            else:
                if new_lock > st["lock"]:
                    st["lock"] = new_lock
                if self.trail_step_lock is not None:
                    step_val = (
                        st["peak"] * (self.trail_step_lock / 100.0)
                        if self.trail_mode == "pct"
                        else self.trail_step_lock
                    )
                    while (st["peak"] - st["lock"]) > (dd_allow + step_val):
                        st["lock"] += step_val

            # ---------- 発火判定 ----------
            if cur <= (st["lock"] if st["lock"] is not None else -1e18):
                # 次のサイクル用に軽くリセット
                st["lock"] = None
                st["peak"] = cur
                st["peak_time"] = now
                st["activated"] = False
                st["neg_bars"] = 0
                st["trough"] = 0.0
                to_close.append(pos)

            # 状態を保存
            self.ticket_state[tk] = st

        return to_close

    # 旧仕様互換用：必要に応じて使っても良いし放置してもOK
    def _reset_trailing_negative_state(self, current_unreal: float, now: datetime):
        self.lock_line = None
        self.equity_peak = current_unreal
        self.equity_peak_time = now
        self.trailing_activated = False
        self.neg_bars = 0
        self.unreal_trough = 0.0


def execute(strategy, symbols):
    trailers = {}
    for i, symbol in enumerate(symbols):
        trailer = Trailer(strategy, symbol)
        trailers[symbol ] = trailer
        if i == 0:
            Mt5Trade.connect()
    while True:
        for symbol in symbols:
            scheduler.enter(60, 1, trailers[symbol].update)
            scheduler.run()
        
        
def test():
    trailer = Trailer('JP225')
    dic = trailer.get_positions()
    for ticket, pos in dic.items():
        print(pos.to_dict())
        
    r = trailer.metric(dic)
    n, win_rate, profit, profit_value = r
    print('metric', r)
     
if __name__ == "__main__":
    symbols = ['JP225', 'XAUUSD', 'USDJPY', 'US100', 'US30']
    execute('Montblanc', symbols)
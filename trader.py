# from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Any
import string
import json
import numpy as np


from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
# from optimiser import best


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()

class TraderDataStore:
    def __init__(self, raw_trader_data: str, max_midprice_history: int = 200):
        self.max_midprice_history = max_midprice_history
        self.data = self._parse(raw_trader_data)
        self.data.setdefault("products", {})

    def _parse(self, raw_trader_data: str) -> dict:
        if not raw_trader_data:
            return {}
        try:
            parsed = json.loads(raw_trader_data)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {}

    def _product_bucket(self, product_name: str) -> dict:
        products = self.data.setdefault("products", {})
        bucket = products.setdefault(product_name, {})
        return bucket

    def get_product_value(self, product_name: str, key: str, default=None):
        return self._product_bucket(product_name).get(key, default)

    def set_product_value(self, product_name: str, key: str, value: Any) -> None:
        self._product_bucket(product_name)[key] = value

    def add_midprice(self, product_name: str, midprice: int | None) -> list[int]:
        bucket = self._product_bucket(product_name)
        history = bucket.get("midprices", [])
        if not isinstance(history, list):
            history = []
        if midprice is not None:
            history.append(midprice)
        bucket["midprices"] = history[-self.max_midprice_history:]
        return bucket["midprices"]

    def to_json(self) -> str:
        try:
            return json.dumps(self.data, separators=(",", ":"))
        except Exception:
            return ""

class Product:
    def __init__(self, name, limit, state, trader_data_store: TraderDataStore):
        self.name = name
        self.position_limit = limit
        self.state = state
        self.orders = []
        self.initial_position = self.get_initial_position()
        self.buy_orders = self.get_buy_orders()
        self.sell_orders = self.get_sell_orders()
        self.max_buy_volume = self.get_max_buy_volume_allowed()
        self.max_sell_volume = self.get_max_sell_volume_allowed()
        self.trader_data_store = trader_data_store


    def get_initial_position(self):
        if self.name in self.state.position:
            return self.state.position[self.name]
        else:
            return 0

    def get_buy_orders(self):
        return self.state.order_depths[self.name].buy_orders

    def get_sell_orders(self):
        return self.state.order_depths[self.name].sell_orders

    def get_best_bid(self):
        if len(self.buy_orders) > 0:
            return max(self.buy_orders.keys())
        else:
            return None

    def get_best_ask(self):
        if len(self.sell_orders) > 0:
            return min(self.sell_orders.keys())
        else: 
            return None

    def get_max_buy_volume_allowed(self):
        return self.position_limit - self.initial_position

    def get_max_sell_volume_allowed(self):
        return self.position_limit + self.initial_position

    def bid(self, price, volume):
        bid_volume = min(volume, self.max_buy_volume)
        if bid_volume > 0:
            self.orders.append(Order(self.name, price, bid_volume))
            self.max_buy_volume -= bid_volume
    
    def ask(self, price, volume):
        ask_volume = min(volume, self.max_sell_volume)
        if ask_volume > 0:
            self.orders.append(Order(self.name, price, -ask_volume))
            self.max_sell_volume -= ask_volume

    def midprice(self):
        if self.get_best_bid() is not None and self.get_best_ask() is not None:
            return int((self.get_best_bid() + self.get_best_ask()) // 2)
        else:
            return None

    def volume_weighted_midprice(self):
        if len(self.buy_orders) > 0 and len(self.sell_orders) > 0:
            buy_volume = np.array(list(self.buy_orders.values()))
            buy_prices = np.array(list(self.buy_orders.keys()))

            sell_volume = np.array(list(self.sell_orders.values()))
            sell_prices = np.array(list(self.sell_orders.keys()))

            return int((np.sum(buy_volume * buy_prices) + np.sum(-sell_volume * sell_prices)) / (np.sum(buy_volume) + -np.sum(sell_volume)))
        else:
            return None

class INTARIAN_PEPPER_ROOT(Product):
    """
    Intarian_pepper_root has a linearly climbing fair price which we will market make around making sure that we keep a long position
    We will do this by taking at fair price to take a long position
    """
    def __init__(self, name, limit, state, trader_data_store):
        super().__init__(name, limit, state, trader_data_store)

    def get_orders(self):
        if self.get_best_bid() is None or self.get_best_ask() is None:
            return []

        fair = (self.get_best_bid() + self.get_best_ask()) / 2

        # ── Trend detection ───────────────────────────────────────────────────
        # Track midprice history to detect if the trend reverses.
        # Slope is computed as (avg of recent half - avg of older half) / half_window.
        # In a normal up-trend this is ~+0.1/tick; a clear reversal gives < NEG_THRESHOLD.
        history = self.trader_data_store.add_midprice(self.name, fair)

        TREND_WINDOW   = 60    # ticks of history to measure slope over
        NEG_THRESHOLD  = -0.02 # slope/tick that triggers exit (price clearly falling)
        POS_THRESHOLD  =  0.05 # slope/tick required to re-enter after being flat
        WAIT_TICKS     = 100   # ticks to sit flat before re-evaluating direction

        slope = None
        if len(history) >= TREND_WINDOW:
            half = TREND_WINDOW // 2
            recent_avg = sum(history[-half:]) / half
            older_avg  = sum(history[-TREND_WINDOW:-half]) / half
            slope = (recent_avg - older_avg) / half

        # ── State machine ─────────────────────────────────────────────────────
        # States: "LONG"  → buy max, ride the trend
        #         "EXITING" → sell to zero immediately
        #         "FLAT"  → hold zero, wait, then check for new trend
        mode = self.trader_data_store.get_product_value(self.name, "mode") or "LONG"
        wait = self.trader_data_store.get_product_value(self.name, "wait") or 0

        if mode == "LONG":
            if slope is not None and slope < NEG_THRESHOLD:
                mode = "EXITING"

        elif mode == "EXITING":
            if self.initial_position == 0:
                mode = "FLAT"
                wait = WAIT_TICKS

        elif mode == "FLAT":
            wait = max(0, wait - 1)
            if wait == 0 and slope is not None and slope > POS_THRESHOLD:
                mode = "LONG"

        self.trader_data_store.set_product_value(self.name, "mode", mode)
        self.trader_data_store.set_product_value(self.name, "wait", wait)

        # ── Orders ────────────────────────────────────────────────────────────
        if mode == "LONG":
            # Sweep all reasonably-priced sell orders by bidding above best ask.
            # No asks posted — selling would just lose drift income.
            self.bid(int(fair + 10), self.max_buy_volume)

        elif mode == "EXITING":
            # Hit all bids aggressively to reach zero as fast as possible.
            if self.initial_position > 0:
                self.ask(int(fair - 10), self.max_sell_volume)
            elif self.initial_position < 0:
                self.bid(int(fair + 10), self.max_buy_volume)

        # mode == "FLAT": do nothing, hold zero

        return self.orders

class ASH_COATED_OSMIUM(Product):
    def __init__(self, name, limit, state, trader_data_store):
        super().__init__(name, limit, state, trader_data_store)

    def find_mm_pairs(self):
        """
        Find symmetric MM quotes (same absolute volume on both bid and ask side).
        Returns a list of (bid_price, ask_price, volume) sorted by half-spread
        ascending (innermost pair first).
        """
        bid_by_vol = {}
        for p, v in self.buy_orders.items():
            # If multiple bids at same volume, take highest (best bid)
            bid_by_vol[v] = max(bid_by_vol.get(v, -1), p)

        ask_by_vol = {}
        for p, v in self.sell_orders.items():
            vol = abs(v)
            # If multiple asks at same volume, take lowest (best ask)
            ask_by_vol[vol] = min(ask_by_vol.get(vol, 1e9), p)

        pairs = []
        for vol, bp in bid_by_vol.items():
            if vol in ask_by_vol:
                ap = ask_by_vol[vol]
                if ap > bp:
                    pairs.append((bp, ap, vol))

        pairs.sort(key=lambda x: x[1] - x[0])  # innermost first
        return pairs

    def Z_score(self, fair):
        midprice_history = self.trader_data_store.add_midprice(self.name, fair)
        lookback = 100
        if len(midprice_history) < 2:
            z_score = 0.0
        else:
            window = np.array(midprice_history[-lookback:], dtype=float)
            mean = np.mean(window)
            std = np.std(window)
            z_score = 0.0 if std < 1e-9 else float((window[-1] - mean) / std)
        self.trader_data_store.set_product_value(self.name, "z_score", z_score)
        return z_score

    def get_orders(self):
        if self.get_best_bid() is None or self.get_best_ask() is None or self.midprice() is None:
            return []

        pairs = self.find_mm_pairs()
        n_pairs = len(pairs)

        # ── Fair value from symmetric pairs ───────────────────────────────────
        if n_pairs > 0:
            total_w = sum(p[2] for p in pairs)
            mm_fair = sum((p[0] + p[1]) / 2.0 * p[2] for p in pairs) / total_w
            self.trader_data_store.set_product_value(self.name, "mm_fair", mm_fair)
            self.trader_data_store.set_product_value(self.name, "prev_pairs", pairs)
        else:
            mm_fair = self.trader_data_store.get_product_value(self.name, "mm_fair")

        # Fallback: wall_mid
        bid_wall = min(self.buy_orders.keys())
        ask_wall = max(self.sell_orders.keys())
        wall_mid = (bid_wall + ask_wall) / 2
        fair = mm_fair if mm_fair is not None else wall_mid
        ref = fair if n_pairs > 0 else wall_mid  # use live wall_mid when MM absent

        # ── Phase 1: Taking ───────────────────────────────────────────────────
        for ap in sorted(self.sell_orders.keys()):
            if ap <= fair - 1:
                self.bid(int(ap), abs(self.sell_orders[ap]))
        for bp in sorted(self.buy_orders.keys(), reverse=True):
            if bp >= fair + 1:
                self.ask(int(bp), self.buy_orders[bp])

        # ── Lag-1 reversal signal (ASH has -0.50 lag-1 autocorrelation) ─────────
        prev_fair = self.trader_data_store.get_product_value(self.name, "prev_fair")
        if prev_fair is not None:
            delta = fair - prev_fair
            if delta > 0:
                self.ask(self.get_best_ask(), self.max_sell_volume)
            elif delta < 0:
                self.bid(self.get_best_bid(), self.max_buy_volume)
        self.trader_data_store.set_product_value(self.name, "prev_fair", fair)

        # ── Z-score mean reversion ─────────────────────────────────────────────
        z_thr = 0.0
        z = self.Z_score(fair)
        if z > z_thr:
            self.ask(self.get_best_ask(), self.max_sell_volume)
        if z < -z_thr:
            self.bid(self.get_best_bid(), self.max_buy_volume)

        # ── Order-book imbalance signal ───────────────────────────────────────
        total_bid = sum(self.buy_orders.values())
        total_ask = sum(abs(v) for v in self.sell_orders.values())
        total_vol = total_bid + total_ask
        imbalance = (total_bid - total_ask) / total_vol if total_vol > 0 else 0.0
        IMB_THR = 0.0
        if imbalance > IMB_THR:
            self.ask(self.get_best_ask(), self.max_sell_volume)
        if imbalance < -IMB_THR:
            self.bid(self.get_best_bid(), self.max_buy_volume)

        # ── Inside-spread order detection (informed trader signal) ────────────
        # If a non-MM order is placed inside the MM spread, it's a strong directional signal
        inside_bid_signal = False
        inside_ask_signal = False
        if n_pairs > 0:
            inner_bid, inner_ask = pairs[0][0], pairs[0][1]
            for bp in self.buy_orders:
                if bp > inner_bid:  # bid inside MM spread
                    inside_bid_signal = True
                    break
            for ap in self.sell_orders:
                if ap < inner_ask:  # ask inside MM spread
                    inside_ask_signal = True
                    break
        if inside_bid_signal:
            self.bid(self.get_best_bid(), self.max_buy_volume)
        if inside_ask_signal:
            self.ask(self.get_best_ask(), self.max_sell_volume)

        # ── Passive making anchored to MM fair ───────────────────────────────
        # Overbid the best existing bid below fair; undercut best ask above fair.
        best_bid_below = max((p for p in self.buy_orders if p < ref), default=None)
        best_ask_above = min((p for p in self.sell_orders if p > ref), default=None)
        if best_bid_below is not None:
            self.bid(int(best_bid_below) + 1, self.max_buy_volume)
        if best_ask_above is not None:
            self.ask(int(best_ask_above) - 1, self.max_sell_volume)

        return self.orders

class Trader:

    # def __init__(self):
    
    def bid(self):
        return None # ignored for now
    
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))


        # Orders to be placed on exchange matching engine
        result = {}
        trader_data_store = TraderDataStore(state.traderData)

        #Intaran_pepper_root
        #Intaran_pepper_root has a linear true price
        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            intaran_pepper_root = INTARIAN_PEPPER_ROOT("INTARIAN_PEPPER_ROOT", 80, state, trader_data_store)
            result[intaran_pepper_root.name] = intaran_pepper_root.get_orders()

        # Ash_coated_osmium
        # Ash_coated_osmium price fluctuates so we need an accurate predictor of true price
        if "ASH_COATED_OSMIUM" in state.order_depths:
            ash_coated_osmium = ASH_COATED_OSMIUM("ASH_COATED_OSMIUM", 80, state, trader_data_store)
            result[ash_coated_osmium.name] = ash_coated_osmium.get_orders()

        traderData = trader_data_store.to_json()
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
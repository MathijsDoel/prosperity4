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
        self.fair_price = int(10000 + self.state.timestamp * 1e-3)

    def get_orders(self):
        #Taking orders at fair price to balance inventory
        if self.get_best_bid() is None or self.get_best_ask() is None:
            return []
        
        #We want to stay around 60 to use the rising price
        if self.initial_position < 60:
            self.bid(self.get_best_ask(), 60-self.initial_position) 

        # if self.get_best_bid() == self.fair_price and self.initial_position > 0:
        #     self.ask(self.fair_price, min(self.initial_position, self.buy_orders[self.fair_price]))
        # if self.get_best_ask() == self.fair_price and self.initial_position < 0:
        #     self.bid(self.fair_price, min(-self.initial_position, -self.sell_orders[self.fair_price]))

        # # Market make by over/undercutting existing orders
        # # If position gets out of balance make it more attractive to rebalance 
        buy_edge = 1
        sell_edge = 1
        if self.get_best_bid() < self.fair_price - buy_edge:
            self.bid(self.get_best_bid() + buy_edge, self.max_buy_volume)
        if self.get_best_ask() > self.fair_price + sell_edge:
            self.ask(self.get_best_ask() - sell_edge, self.max_sell_volume)
        
        return self.orders

class ASH_COATED_OSMIUM(Product):
    def __init__(self, name, limit, state, trader_data_store):
        super().__init__(name, limit, state, trader_data_store)

    def get_EMA(self):
        N = 50
        alpha = 2 / (N+1)
        old_ema = self.trader_data_store.get_product_value(self.name, "EMA")
        if old_ema is None:
            new_EMA = self.midprice()
        else:
            new_EMA = old_ema * (1-alpha) + self.midprice() * alpha
        self.trader_data_store.set_product_value(self.name, "EMA", new_EMA)
        return new_EMA

    def Z_score(self):
        midprice_history = self.trader_data_store.add_midprice(self.name, self.midprice())
        lookback = 50
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
        # Market make by over/undercutting existing orders and skewing the orders based on inventory
        # if self.get_best_bid() == self.midprice() and self.initial_position > 0:
        #     self.ask(self.midprice(), min(self.initial_position, self.buy_orders[self.midprice()]))
        # if self.get_best_ask() == self.midprice() and self.initial_position < 0:
        #     self.bid(self.midprice(), min(-self.initial_position, -self.sell_orders[self.midprice()]))

        # Market make by over/undercutting existing orders
        # If position gets out of balance make it more attractive to rebalance 
        if self.Z_score() > 2:
            self.ask(self.get_best_ask(), self.max_sell_volume)
        if self.Z_score() < -2:
            self.bid(self.get_best_bid(), self.max_buy_volume)

        buy_edge = 1
        sell_edge = 1
        if self.get_best_bid() < self.midprice() - buy_edge:
            self.bid(self.get_best_bid() + buy_edge, self.max_buy_volume)
        if self.get_best_ask() > self.midprice() + sell_edge:
            self.ask(self.get_best_ask() - sell_edge, self.max_sell_volume)

        return self.orders

class Trader:

    # def __init__(self):
    
    def bid(self):
        return None # ignored for now
    
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))


        # Orders to be placed on exchange matching engine
        result = {}
        trader_data_store = TraderDataStore(state.traderData)

        #Intaran_pepper_root
        #Intaran_pepper_root has a linear true price
        intaran_pepper_root = INTARIAN_PEPPER_ROOT("INTARIAN_PEPPER_ROOT", 80, state, trader_data_store)
        result[intaran_pepper_root.name] = intaran_pepper_root.get_orders()

       # Ash_coated_osmium
       # Ash_coated_osmium price fluctuates so we need an accurate predictor of true price
        ash_coated_osmium = ASH_COATED_OSMIUM("ASH_COATED_OSMIUM", 80, state, trader_data_store)
        result[ash_coated_osmium.name] = ash_coated_osmium.get_orders()

        traderData = trader_data_store.to_json()
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
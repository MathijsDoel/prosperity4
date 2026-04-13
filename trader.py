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

class Product:
    def __init__(self, name, limit, state):
        self.name = name
        self.position_limit = limit
        self.state = state
        self.orders = []
        self.position = self.get_position()
        self.buy_orders = self.get_buy_orders()
        self.sell_orders = self.get_sell_orders()

    def get_position(self):
        if self.name in self.state.position:
            return self.state.position[self.name]
        else:
            return 0

    def get_buy_orders(self):
        return self.state.order_depths[self.name].buy_orders

    def get_sell_orders(self):
        return self.state.order_depths[self.name].sell_orders

    def get_best_bid(self):
        return max(self.buy_orders.keys())

    def get_best_ask(self):
        return min(self.sell_orders.keys())

    def max_buy_volume(self):
        return self.position_limit - self.position
    
    def max_sell_volume(self):
        return self.position_limit + self.position

    def bid(self, price, volume):
        bid_volume = min(volume, self.max_buy_volume())
        self.orders.append(Order(self.name, price, bid_volume))
    
    def ask(self, price, volume):
        ask_volume = min(volume, self.max_sell_volume())
        self.orders.append(Order(self.name, price, -ask_volume))

    def midprice(self):
        return int((self.get_best_bid() + self.get_best_ask()) // 2)

    def volume_weighted_midprice(self):
        buy_volume = np.array(list(self.buy_orders.values()))
        buy_prices = np.array(list(self.buy_orders.keys()))

        sell_volume = np.array(list(self.sell_orders.values()))
        sell_prices = np.array(list(self.sell_orders.keys()))

        return int((np.sum(buy_volume * buy_prices) + np.sum(-sell_volume * sell_prices)) / (np.sum(buy_volume) + -np.sum(sell_volume)))

class Emeralds(Product):
    def __init__(self, name, limit, state):
        super().__init__(name, limit, state)
        self.fair_price = 10000

    def get_orders(self):
        # Market make by over/undercutting existing orders

        if self.get_best_bid() == self.fair_price and self.position < 0:
            self.bid(self.fair_price, min(-self.position, self.buy_orders()[self.fair_price]))
        if self.get_best_ask() == self.fair_price and self.position > 0:
            self.ask(self.fair_price, min(self.position, -self.buy_orders()[self.fair_price]))
        if self.get_best_bid() < self.fair_price - 1:
            self.bid(self.get_best_bid() + 1, self.max_buy_volume())
           
        if self.get_best_ask() > self.fair_price + 1:
            self.ask(self.get_best_ask() - 1, self.max_sell_volume())
        
        #buy/sell at fair price if position get unbalanced
        if self.position > 40:
            self.ask(self.fair_price, self.position)
        elif self.position < 40:
            self.bid(self.fair_price, -self.position)
        
        return self.orders

class Tomatoes(Product):
    def __init__(self, name, limit, state):
        super().__init__(name, limit, state)
        self.balancing_force = 0.04

    def get_orders(self):
        # Market make by over/undercutting existing orders and skewing the orders based on inventory
        balance_factor = int(-self.position * self.balancing_force)
        if self.get_best_bid() < self.volume_weighted_midprice() - 1:
            self.bid(self.get_best_bid() + 1 + balance_factor, self.max_buy_volume())
           
        if self.get_best_ask() > self.volume_weighted_midprice() + 1:
            self.ask(self.get_best_ask() - 1 + balance_factor, self.max_sell_volume())

        
        return self.orders

class Trader:

    def __init__(self):
        self.products = ["TOMATOES", "EMERALDS"]
        self.limits = {"TOMATOES": 80, "EMERALDS": 80}
        self.position_thresholds = {"TOMATOES": 40, "EMERALDS": 40}
    
    def bid(self):
        return None # ignored for now
    
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))


        # Orders to be placed on exchange matching engine
        result = {}
        global_position = state.position

        #Emeralds
        #Emeralds has a fixed true price of 10000 and thus we only market make with a spread 
        emeralds = Emeralds("EMERALDS", 80, state)
        result[emeralds.name] = emeralds.get_orders()

       # Tomatoes
       # Tomatoes price fluctuates so we need an accurate predictor of true price
        tomatoes = Tomatoes("TOMATOES", 80, state)
        result[tomatoes.name] = tomatoes.get_orders()
    
        traderData = ""  # No state needed - we check position directly
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
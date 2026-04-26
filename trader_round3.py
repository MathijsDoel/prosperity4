from typing import List, Any
import string
import json
import numpy as np
import math

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

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
            state.timestamp, trader_data, self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths), self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades), state.position, self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for t in arr: compressed.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_obs = {}
        for p, o in observations.conversionObservations.items():
            conversion_obs[p] = [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
        return [observations.plainValueObservations, conversion_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for o in arr: compressed.append([o.symbol, o.price, o.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length: return value
        lo, hi = 0, max_length
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid] + "..." if mid < len(value) else value[:mid]
            if len(json.dumps(candidate)) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out

logger = Logger()

class TraderDataStore:
    def __init__(self, raw_trader_data: str, max_history: int = 300):
        self.max_history = max_history
        self.data = self._parse(raw_trader_data)
        self.data.setdefault("products", {})

    def _parse(self, raw_trader_data: str) -> dict:
        if not raw_trader_data: return {}
        try:
            parsed = json.loads(raw_trader_data)
            return parsed if isinstance(parsed, dict) else {}
        except Exception: return {}

    def _product_bucket(self, product_name: str) -> dict:
        return self.data.setdefault("products", {}).setdefault(product_name, {})

    def get_value(self, product_name: str, key: str, default=None):
        return self._product_bucket(product_name).get(key, default)

    def set_value(self, product_name: str, key: str, value: Any) -> None:
        self._product_bucket(product_name)[key] = value

    def add_value(self, product_name: str, key: str, val: float | None) -> list[float]:
        bucket = self._product_bucket(product_name)
        history = bucket.get(key, [])
        if not isinstance(history, list): history = []
        if val is not None: history.append(val)
        bucket[key] = history[-self.max_history:]
        return bucket[key]

    def to_json(self) -> str:
        try: return json.dumps(self.data, separators=(",", ":"))
        except Exception: return ""

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

# --- NEW: STANDALONE STRATEGY CLASS ---
class VelvetHydrogelPairsStrategy:
    """
    Encapsulates the Z-Score Statistical Arbitrage strategy for the 
    VELVETFRUIT_EXTRACT and HYDROGEL_PACK pair.
    """
    def __init__(self, state: TradingState, trader_data_store: TraderDataStore):
        # Instantiate the standalone Product classes
        self.velvet = Product("VELVETFRUIT_EXTRACT", 200, state, trader_data_store)
        self.hydrogel = Product("HYDROGEL_PACK", 200, state, trader_data_store)
        
        self.state = state
        self.store = trader_data_store
        
        # Strategy Parameters
        self.beta = 0.525  # Update this with your ADF best_beta

    def get_orders(self) -> dict[Symbol, list[Order]]:
        """Calculates the static spread and returns generated orders for both products."""
        result = {}
        
        # Ensure products are tradable on this tick
        if self.velvet.name not in self.state.order_depths or self.hydrogel.name not in self.state.order_depths:
            return result

        mid_v = self.velvet.midprice()
        mid_h = self.hydrogel.midprice()

        if mid_v is None or mid_h is None:
            return result

        # 1. Calculate Spread
        current_spread = mid_v - (self.beta * mid_h)
        
        # Strategy Thresholds
        spread_signal = 30  # Entry threshold
        exit_signal = 10     # Exit threshold (revert to 0 when spread is within this range)

        # Default to holding whatever our current position is
        target_v = self.velvet.initial_position
        target_h = self.hydrogel.initial_position

        # 2. Determine Target Positions based on the spread
        if current_spread > spread_signal:
            # Spread is too high -> Short Velvet, Long Hydrogel
            target_v = -self.velvet.position_limit
            target_h = self.hydrogel.position_limit
            
        elif current_spread < -spread_signal:
            # Spread is too low -> Long Velvet, Short Hydrogel
            target_v = self.velvet.position_limit
            target_h = -self.hydrogel.position_limit
            
        elif abs(current_spread) < exit_signal:
            # Spread has normalized -> Exit positions (Flatten)
            target_v = 0
            target_h = 0

        # 3. Execute Velvet Orders
        trade_v = target_v - self.velvet.initial_position
        if trade_v > 0:
            # Need to buy Velvet
            self.velvet.bid(self.velvet.get_best_ask(), trade_v)
        elif trade_v < 0:
            # Need to sell Velvet
            self.velvet.ask(self.velvet.get_best_bid(), abs(trade_v))

        # 4. Execute Hydrogel Orders
        trade_h = target_h - self.hydrogel.initial_position
        if trade_h > 0:
            # Need to buy Hydrogel
            self.hydrogel.bid(self.hydrogel.get_best_ask(), trade_h)
        elif trade_h < 0:
            # Need to sell Hydrogel
            self.hydrogel.ask(self.hydrogel.get_best_bid(), abs(trade_h))

        # 5. Append to results
        if self.velvet.orders:
            result[self.velvet.name] = self.velvet.orders
        if self.hydrogel.orders:
            result[self.hydrogel.name] = self.hydrogel.orders
            
        return result

class Option(Product):
    def __init__(self, name, limit, strike, state, trader_data_store):
        super().__init__(name, limit, state, trader_data_store)
        self.strike = strike
        self.underlying = Product("VELVETFRUIT_EXTRACT", 200, state, trader_data_store)
        
        # Base volatility (Consider updating this dynamically based on the ATM smile!)
        self.sigma = 0.012 

    @staticmethod
    def norm_cdf(x):
        """
        Calculates the Standard Normal Cumulative Distribution Function 
        using the Abramowitz and Stegun approximation.
        Works with both scalar values and numpy arrays.
        """
        p  = 0.2316419
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429

        sign = np.sign(x)
        x_abs = np.abs(x)

        pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x_abs**2)
        t = 1.0 / (1.0 + p * x_abs)
        poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
        cdf_pos = 1.0 - (pdf * poly)

        return np.where(sign >= 0, cdf_pos, 1.0 - cdf_pos)

    def Scholz_price(self):
        S = self.underlying.midprice()
        if S is None:
            return None
            
        K = self.strike
        sigma = self.sigma
        
        # FIXED: Dynamic, Annualized Time Till Expiry
        # Assumes 1,000,000 ticks per day, starting at 7 days, 252 trading days a year
        days_passed = self.state.timestamp / 1000000.0
        continuous_TTE_days = 7.0 - days_passed
        T = continuous_TTE_days / 252.0

        # Safeguard: If we somehow reach exactly expiration
        if T <= 0:
            return max(0.0, S - K)

        d1 = (np.log(S/K) + (sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        C = S * self.norm_cdf(d1) - K * self.norm_cdf(d2)
        return C

    def get_orders(self):
        """
        Calculates fair price and places market making limit orders 
        on both sides of the book.
        """
        fair_price = self.Scholz_price()
        
        if fair_price is None:
            return self.orders

        # --- MARKET MAKING PARAMETERS ---
        # "Edge" is how much profit you want to make per trade. 
        # A higher edge means safer trades but fewer fills.
        edge = 1.5 

        # --- QUOTE CALCULATION ---
        # Exchange requires integer prices. 
        # We use floor for bids (to buy cheaper) and ceil for asks (to sell higher)
        my_bid_price = math.floor(fair_price - edge)
        my_ask_price = math.ceil(fair_price + edge)

        # --- EXECUTION ---
        # Your base Product class handles the limit logic, so we can safely 
        # request our max available volume.
        if self.max_buy_volume > 0:
            self.bid(my_bid_price, self.max_buy_volume)
            
        if self.max_sell_volume > 0:
            self.ask(my_ask_price, self.max_sell_volume)

        return self.orders



class Trader:
    def bid(self):
        return None # ignored for now

    def run(self, state: TradingState):
        logger.print("traderData: " + state.traderData)

        result = {}
        trader_data_store = TraderDataStore(state.traderData)

        # --- THE CLEAN TRADER LOGIC ---
        # Instantiate the strategy and let it handle everything
        pairs_strategy = VelvetHydrogelPairsStrategy(state, trader_data_store)
   
        
        # Merge the generated orders into our final result dictionary
        # result.update(pairs_strategy.get_orders())
        result = pairs_strategy.get_orders()
        # logger.print("ORDERS:", pairs_strategy.get_orders())
        # ------------------------------
        option1 = Option("VEV_5000", 300, 5000, state, trader_data_store)
        result[option1.name] = option1.get_orders()

        traderData = trader_data_store.to_json()
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData
# from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Any
import string
import json
import math

from datamodel import Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Listing, Observation

# ── Config ────────────────────────────────────────────────────────────────────

# HYDROGEL_PACK — stable mean-reverter around 10000, spread ~16
HP_LIMIT      = 200
HP_GAMMA      = 0.10     # inventory skew: reservation shifts 0.1 per unit held
HP_TAKE_EDGE  = 15       # take mispriced orders when >15 from FV
# VELVETFRUIT_EXTRACT — drifty (+14/day hist.), spread ~6
# No take phase: EMA lags on trending days, causing wrong-way aggressive fills.
# Passive MM only using current mid as reference.
VEV_LIMIT     = 200
VEV_GAMMA     = 0.05

# VEV options — BS MM on liquid ATM strikes (5400+ has 2-tick spread → skip; 4000/4500 pure intrinsic)
ATM_STRIKES   = [5000, 5100, 5200, 5300]
OPT_LIMIT     = 50       # conservative per-strike position cap
OPT_SIGMA     = 0.016    # market implied vol for neutral BS pricing
OPT_HALF_SPD  = 1        # ±1 from BS FV → 2-tick spread; guarantees no zero-spread quoting
OPT_GAMMA     = 0.02     # small skew: moves quotes ~1 tick per 50 units inventory
OPT_TAKE_EDGE = 6        # take if ask < BS_FV - 6 or bid > BS_FV + 6

# TTE tracking
TTE_START     = 5.0
TICKS_PER_DAY = 1_000_000   # timestamps 0–999900, ~1M per day


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

# ── Black-Scholes helpers ─────────────────────────────────────────────────────

def _ncdf(x):
    if x < 0:
        return 1.0 - _ncdf(-x)
    k = 1.0 / (1.0 + 0.2316419 * x)
    p = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))))
    return 1.0 - math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi) * p

def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    st = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / st
    return S * _ncdf(d1) - K * _ncdf(d1 - st)


# ── Non-derivative MM (HP and VEV) ────────────────────────────────────────────

def mm_orders_hp(od, pos):
    """
    HYDROGEL_PACK MM:
      FV = current market mid. HP is stable so mid ≈ 10000 long-run.
      Phase 1: aggressive take when >HP_TAKE_EDGE from FV.
      Phase 2: passive overbid/undercut around inventory-skewed reservation.
    """
    def __init__(self, name, limit, state, trader_data_store):
        super().__init__(name, limit, state, trader_data_store)

    bb  = max(od.buy_orders)
    ba  = min(od.sell_orders)
    mid = (bb + ba) / 2.0

    fv   = mid
    mx_b = HP_LIMIT - pos
    mx_s = HP_LIMIT + pos
    orders = []

    # Phase 1: aggressive take
    for ap in sorted(od.sell_orders):
        if ap > fv - HP_TAKE_EDGE:
            break
        vol = min(abs(od.sell_orders[ap]), mx_b)
        if vol > 0:
            orders.append(Order("HYDROGEL_PACK", ap, vol))
            mx_b -= vol; pos += vol

    for bp in sorted(od.buy_orders, reverse=True):
        if bp < fv + HP_TAKE_EDGE:
            break
        vol = min(od.buy_orders[bp], mx_s)
        if vol > 0:
            orders.append(Order("HYDROGEL_PACK", bp, -vol))
            mx_s -= vol; pos -= vol

    # Phase 2: passive overbid/undercut
    reservation = fv - HP_GAMMA * pos
    bb_below = max((p for p in od.buy_orders  if p < reservation), default=None)
    ba_above = min((p for p in od.sell_orders if p > reservation), default=None)

    if bb_below is not None and mx_b > 0:
        orders.append(Order("HYDROGEL_PACK", bb_below + 1, mx_b))
    if ba_above is not None and mx_s > 0:
        orders.append(Order("HYDROGEL_PACK", ba_above - 1, -mx_s))

    return orders


def mm_orders_vev(od, pos, fv):
    """
    VELVETFRUIT_EXTRACT MM — passive only, no take phase.
    FV = current market mid (passed in). No lagging EMA to avoid wrong-way
    aggressive fills on trending days. Inventory skew via AS reservation.
    """
    if not od.buy_orders or not od.sell_orders:
        return []

    mx_b = VEV_LIMIT - pos
    mx_s = VEV_LIMIT + pos
    orders = []

    reservation = fv - VEV_GAMMA * pos
    bb_below = max((p for p in od.buy_orders  if p < reservation), default=None)
    ba_above = min((p for p in od.sell_orders if p > reservation), default=None)

    if bb_below is not None and mx_b > 0:
        orders.append(Order("VELVETFRUIT_EXTRACT", bb_below + 1, mx_b))
    if ba_above is not None and mx_s > 0:
        orders.append(Order("VELVETFRUIT_EXTRACT", ba_above - 1, -mx_s))

    return orders


# ── Option MM ────────────────────────────────────────────────────────────────

def option_mm_orders(name, od, pos, vev_mid, K, tte):
    """
    BS-based option MM using VEV mid-price as spot (correlation-based FV):
      - FV = BS(vev_mid, K, tte, OPT_SIGMA)
      - Reservation = FV - OPT_GAMMA * pos  (inventory skew keeps position balanced)
      - Phase 1: take if ask < FV - OPT_TAKE_EDGE or bid > FV + OPT_TAKE_EDGE
      - Phase 2: post bid at int(reservation) - OPT_HALF_SPD,
                       ask at int(reservation) + OPT_HALF_SPD
    """
    if not od.buy_orders or not od.sell_orders or vev_mid is None or tte <= 0:
        return []

    fv = bs_call(vev_mid, K, tte, OPT_SIGMA)
    if fv <= 0:
        return []

    bb = max(od.buy_orders)
    ba = min(od.sell_orders)

    mx_b = OPT_LIMIT - pos
    mx_s = OPT_LIMIT + pos
    orders = []

    # Phase 1: take obvious mispricings
    if ba <= fv - OPT_TAKE_EDGE and mx_b > 0:
        vol = min(abs(od.sell_orders[ba]), mx_b)
        orders.append(Order(name, ba, vol))
        mx_b -= vol
        pos  += vol

    if bb >= fv + OPT_TAKE_EDGE and mx_s > 0:
        vol = min(od.buy_orders[bb], mx_s)
        orders.append(Order(name, bb, -vol))
        mx_s -= vol
        pos  -= vol

    # Phase 2: fixed ±OPT_HALF_SPD around inventory-skewed reservation.
    # Small gamma keeps quotes near market even at large positions.
    # Guaranteed 2-tick spread prevents zero-spread adverse selection.
    reservation = fv - OPT_GAMMA * pos
    bid_p = int(reservation) - OPT_HALF_SPD
    ask_p = int(reservation) + OPT_HALF_SPD

    if bid_p < ba and mx_b > 0:
        orders.append(Order(name, bid_p, mx_b))
    if ask_p > bb and mx_s > 0:
        orders.append(Order(name, ask_p, -mx_s))

    return orders

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

        # ── Z-score mean reversion (conservative threshold) ────────────────────
        z_thr = {0: 2.0, 1: 1.5, 2: 1.0}.get(n_pairs, 1.5)
        z = self.Z_score(fair)
        if z > z_thr:
            self.ask(self.get_best_ask(), self.max_sell_volume)
        if z < -z_thr:
            self.bid(self.get_best_bid(), self.max_buy_volume)

        # ── Lag-1 reversal signal (threshold filters noise)
        prev_fair = self.trader_data_store.get_product_value(self.name, "prev_fair")
        if prev_fair is not None:
            delta = fair - prev_fair
            if delta > 1.0:
                self.ask(self.get_best_ask(), self.max_sell_volume)
            elif delta < -1.0:
                self.bid(self.get_best_bid(), self.max_buy_volume)
        self.trader_data_store.set_product_value(self.name, "prev_fair", fair)

        # ── FH passive making anchored to MM fair ─────────────────────────────
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

        # Load persisted state
        try:
            store = json.loads(state.traderData) if state.traderData else {}
            if not isinstance(store, dict):
                store = {}
        except Exception:
            store = {}

        # TTE tracking: detect day boundary when timestamp resets
        last_ts  = store.get("last_ts", 0)
        tte_base = store.get("tte_base", TTE_START)
        if state.timestamp < last_ts:             # day rolled over
            tte_base = max(0.0, tte_base - 1.0)
        tte = tte_base - state.timestamp / TICKS_PER_DAY
        store["last_ts"]  = state.timestamp
        store["tte_base"] = tte_base

        # ── HYDROGEL_PACK ─────────────────────────────────────────────────────
        # HP is stable mean-reverting: use slow EMA as FV + aggressive take phase.
        if "HYDROGEL_PACK" in state.order_depths:
            od  = state.order_depths["HYDROGEL_PACK"]
            pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = mm_orders_hp(od, pos)

        # ── VELVETFRUIT_EXTRACT ───────────────────────────────────────────────
        # VEV drifts: use current market mid as FV (no lagging EMA), passive MM only.
        vev_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            od = state.order_depths["VELVETFRUIT_EXTRACT"]
            if od.buy_orders and od.sell_orders:
                vev_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
            result["VELVETFRUIT_EXTRACT"] = mm_orders_vev(
                od,
                state.position.get("VELVETFRUIT_EXTRACT", 0),
                vev_mid,
            )

        # ── VEV Options: BS MM using current VEV price as spot ────────────────
        # VEV mid-price feeds directly into Black-Scholes to price each option.
        # This cross-asset correlation is the FV predictor for all derivatives.
        if vev_mid is not None and tte > 0.01:
            for K in ATM_STRIKES:
                name = f"VEV_{K}"
                if name not in state.order_depths:
                    continue
                od  = state.order_depths[name]
                pos = state.position.get(name, 0)
                result[name] = option_mm_orders(name, od, pos, vev_mid, K, tte)

        logger.flush(state, result, 0, json.dumps(store, separators=(",", ":")))
        return result, 0, json.dumps(store, separators=(",", ":"))

import json
import math

from datamodel import Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Listing, Observation

# ── Config ────────────────────────────────────────────────────────────────────

HP_LIMIT      = 200
HP_GAMMA      = 0.10
HP_TAKE_EDGE  = 19

VEV_LIMIT     = 200
VEV_GAMMA     = 0.05

ATM_STRIKES   = [5000, 5100, 5200, 5300]
OPT_LIMIT     = 300
OPT_SIGMA     = 0.016    # base at TTE=5 (day 0); rises by 0.001/day as TTE falls
OPT_HALF_SPD  = 1
OPT_GAMMA     = 0.02
OPT_TAKE_EDGE = 2

TTE_START     = 5.0
TICKS_PER_DAY = 1_000_000


# ── Logger ────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self):
        self.logs = ""
        self.max_log_length = 3750

    def flush(self, state, orders, conversions, trader_data):
        base = len(self.to_json([
            self.compress_state(state, ""), self.compress_orders(orders),
            conversions, "", "",
        ]))
        max_item = (self.max_log_length - base) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item),
            self.truncate(self.logs, max_item),
        ]))
        self.logs = ""

    def compress_state(self, state, td):
        return [
            state.timestamp, td,
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {s: [od.buy_orders, od.sell_orders] for s, od in state.order_depths.items()},
            self._trades(state.own_trades),
            self._trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def _trades(self, trades):
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for arr in trades.values() for t in arr
        ]

    def compress_observations(self, obs):
        co = {}
        for p, o in obs.conversionObservations.items():
            co[p] = [o.bidPrice, o.askPrice, o.transportFees,
                     o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
        return [obs.plainValueObservations, co]

    def compress_orders(self, orders):
        return [
            [o.symbol, o.price, o.quantity]
            for arr in orders.values() for o in arr
        ]

    def to_json(self, v):
        return json.dumps(v, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value, max_length):
        lo, hi, out = 0, min(len(value), max_length), ""
        while lo <= hi:
            mid = (lo + hi) // 2
            c = value[:mid] + ("..." if mid < len(value) else "")
            if len(json.dumps(c)) <= max_length:
                out = c
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ── Black-Scholes ─────────────────────────────────────────────────────────────

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


# ── Market making ─────────────────────────────────────────────────────────────

def mm_orders_hp(od, pos):
    if not od.buy_orders or not od.sell_orders:
        return []

    bb  = max(od.buy_orders)
    ba  = min(od.sell_orders)
    fv  = 10000.0          # HP is stable; fixed FV gives mean-reversion behaviour
    mx_b = HP_LIMIT - pos
    mx_s = HP_LIMIT + pos
    orders = []

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

    reservation = fv - HP_GAMMA * pos
    bb_below = max((p for p in od.buy_orders  if p < reservation), default=None)
    ba_above = min((p for p in od.sell_orders if p > reservation), default=None)

    if bb_below is not None and mx_b > 0:
        orders.append(Order("HYDROGEL_PACK", bb_below + 1, mx_b))
    if ba_above is not None and mx_s > 0:
        orders.append(Order("HYDROGEL_PACK", ba_above - 1, -mx_s))

    return orders


def mm_orders_vev(od, pos, fv):
    if not od.buy_orders or not od.sell_orders:
        return []

    mx_b = VEV_LIMIT - pos
    mx_s = VEV_LIMIT + pos
    reservation = fv - VEV_GAMMA * pos
    bb_below = max((p for p in od.buy_orders  if p < reservation), default=None)
    ba_above = min((p for p in od.sell_orders if p > reservation), default=None)

    orders = []
    if bb_below is not None and mx_b > 0:
        orders.append(Order("VELVETFRUIT_EXTRACT", bb_below + 1, mx_b))
    if ba_above is not None and mx_s > 0:
        orders.append(Order("VELVETFRUIT_EXTRACT", ba_above - 1, -mx_s))
    return orders


def option_mm_orders(name, od, pos, vev_mid, K, tte):
    if not od.buy_orders or not od.sell_orders or vev_mid is None or tte <= 0:
        return []

    sigma = OPT_SIGMA + (TTE_START - tte) * 0.001  # vol rises as TTE falls
    fv = bs_call(vev_mid, K, tte, sigma)
    if fv <= 0:
        return []

    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    mx_b = OPT_LIMIT - pos
    mx_s = OPT_LIMIT + pos
    orders = []

    if ba <= fv - OPT_TAKE_EDGE and mx_b > 0:
        vol = min(abs(od.sell_orders[ba]), mx_b)
        orders.append(Order(name, ba, vol))
        mx_b -= vol; pos += vol

    if bb >= fv + OPT_TAKE_EDGE and mx_s > 0:
        vol = min(od.buy_orders[bb], mx_s)
        orders.append(Order(name, bb, -vol))
        mx_s -= vol; pos -= vol

    reservation = fv - OPT_GAMMA * pos
    bid_p = int(reservation) - OPT_HALF_SPD
    ask_p = int(reservation) + OPT_HALF_SPD

    if bid_p < ba and mx_b > 0:
        orders.append(Order(name, bid_p, mx_b))
    if ask_p > bb and mx_s > 0:
        orders.append(Order(name, ask_p, -mx_s))

    return orders


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state):
        result = {}

        try:
            store = json.loads(state.traderData) if state.traderData else {}
            if not isinstance(store, dict):
                store = {}
        except Exception:
            store = {}

        last_ts  = store.get("last_ts", 0)
        tte_base = store.get("tte_base", TTE_START)
        if state.timestamp < last_ts:
            tte_base = max(0.0, tte_base - 1.0)
        tte = tte_base - state.timestamp / TICKS_PER_DAY
        store["last_ts"]  = state.timestamp
        store["tte_base"] = tte_base

        if "HYDROGEL_PACK" in state.order_depths:
            od  = state.order_depths["HYDROGEL_PACK"]
            pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = mm_orders_hp(od, pos)

        vev_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            od = state.order_depths["VELVETFRUIT_EXTRACT"]
            if od.buy_orders and od.sell_orders:
                vev_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
            result["VELVETFRUIT_EXTRACT"] = mm_orders_vev(
                od, state.position.get("VELVETFRUIT_EXTRACT", 0), vev_mid,
            )

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

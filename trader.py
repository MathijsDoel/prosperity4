import json
import math

from datamodel import Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Listing, Observation

# ── Config ────────────────────────────────────────────────────────────────────

HP_LIMIT          = 200
HP_GAMMA          = 0.10
HP_TAKE_EDGE      = 19

VEV_LIMIT         = 200
VEV_GAMMA         = 0.05

# Options — trade strikes closest to current VEV mid
ATM_STRIKES       = [5000, 5100, 5200, 5300, 5400]
OPT_LIMIT         = 300      # natural competition limit
OPT_HALF_SPD      = 1        # ±1 from FV = 2-tick spread
OPT_GAMMA         = 0.02     # AS inventory skew per unit held
OPT_TAKE_EDGE     = 2        # take when option lags a VEV move by >2 ticks

SIGMA_INIT        = 0.015    # starting guess for implied vol solver
SIGMA_EMA         = 0.5      # weight on new implied vol vs stored — fast enough to track in ~3 ticks

# R4: TTE_START = 4.0  |  R3: TTE_START = 5.0  — CHECK EVERY ROUND
TTE_START         = 4.0
TICKS_PER_DAY     = 1_000_000


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

def _npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0), (1.0 if S > K else 0.0), 0.0
    st = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / st
    price = S * _ncdf(d1) - K * _ncdf(d1 - st)
    delta = _ncdf(d1)
    vega  = S * math.sqrt(T) * _npdf(d1)
    return price, delta, vega

def implied_vol(market_price, S, K, T, sigma_init):
    """Newton-Raphson: find sigma s.t. BS(S,K,T,sigma) = market_price."""
    if T <= 0:
        return sigma_init
    intrinsic = max(S - K, 0.0)
    if market_price <= intrinsic + 0.5:
        return sigma_init  # price is essentially intrinsic — no time value to fit
    sigma = max(sigma_init, 0.001)
    for _ in range(25):
        price, _, vega = bs_call(S, K, T, sigma)
        if vega < 1e-6:
            break
        sigma -= (price - market_price) / vega
        sigma = max(0.001, min(sigma, 2.0))
    return sigma


# ── Market making — HP and VEV ────────────────────────────────────────────────

def mm_orders_hp(od, pos):
    if not od.buy_orders or not od.sell_orders:
        return []

    fv   = 10000.0
    bb   = max(od.buy_orders)
    ba   = min(od.sell_orders)
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


# ── Option MM ─────────────────────────────────────────────────────────────────

def option_mm_orders(name, od, pos, vev_mid, K, tte, sigma):
    if not od.buy_orders or not od.sell_orders or vev_mid is None or tte <= 0:
        return []

    fv, _, _ = bs_call(vev_mid, K, tte, sigma)
    if fv <= 0:
        return []

    bb   = max(od.buy_orders)
    ba   = min(od.sell_orders)
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

        # TTE tracking
        last_ts  = store.get("last_ts", 0)
        tte_base = store.get("tte_base", TTE_START)
        if state.timestamp < last_ts:
            tte_base = max(0.0, tte_base - 1.0)
        tte = tte_base - state.timestamp / TICKS_PER_DAY
        store["last_ts"]  = state.timestamp
        store["tte_base"] = tte_base

        # ── HYDROGEL_PACK ─────────────────────────────────────────────────────
        if "HYDROGEL_PACK" in state.order_depths:
            od  = state.order_depths["HYDROGEL_PACK"]
            pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = mm_orders_hp(od, pos)

        # ── VELVETFRUIT_EXTRACT ───────────────────────────────────────────────
        vev_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            od = state.order_depths["VELVETFRUIT_EXTRACT"]
            if od.buy_orders and od.sell_orders:
                vev_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
            result["VELVETFRUIT_EXTRACT"] = mm_orders_vev(
                od, state.position.get("VELVETFRUIT_EXTRACT", 0), vev_mid,
            )

        # ── VEV Options ───────────────────────────────────────────────────────
        if vev_mid is None or tte <= 0.01:
            logger.flush(state, result, 0, json.dumps(store, separators=(",", ":")))
            return result, 0, json.dumps(store, separators=(",", ":"))

        # Extract implied vol per strike — each option tracks its own market mid
        # FV starts at market mid, then updates with VEV moves via BS (cross-asset signal)
        sigmas = store.get("sigmas", {})

        for K in ATM_STRIKES:
            name = f"VEV_{K}"
            if name not in state.order_depths:
                continue
            od  = state.order_depths[name]
            pos = state.position.get(name, 0)
            sk  = str(K)

            if od.buy_orders and od.sell_orders:
                market_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
                prev_sigma = sigmas.get(sk, SIGMA_INIT)
                raw = implied_vol(market_mid, vev_mid, K, tte, prev_sigma)
                sigmas[sk] = (1 - SIGMA_EMA) * prev_sigma + SIGMA_EMA * raw
                sigmas[sk] = max(0.005, min(sigmas[sk], 0.5))

            sigma = sigmas.get(sk, SIGMA_INIT)
            result[name] = option_mm_orders(name, od, pos, vev_mid, K, tte, sigma)

        store["sigmas"] = sigmas

        logger.flush(state, result, 0, json.dumps(store, separators=(",", ":")))
        return result, 0, json.dumps(store, separators=(",", ":"))

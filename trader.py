from typing import Any
import json

from datamodel import Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Listing, Observation

# ── Config ────────────────────────────────────────────────────────────────────

HP_FAIR  = 10000.0   # used ONLY for obvious-misprice taking (wide edge)
HP_GAMMA = 0.10      # inventory skew: reservation = mid - GAMMA * position
HP_LIMIT = 200

VEV_GAMMA = 0.05
VEV_LIMIT = 200


# ── Logger ────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def flush(self, state: TradingState, orders: dict, conversions: int, trader_data: str) -> None:
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
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, obs):
        co = {}
        for p, o in obs.conversionObservations.items():
            co[p] = [o.bidPrice, o.askPrice, o.transportFees,
                     o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
        return [obs.plainValueObservations, co]

    def compress_orders(self, orders):
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, v: Any) -> str:
        return json.dumps(v, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi, out = 0, min(len(value), max_length), ""
        while lo <= hi:
            mid = (lo + hi) // 2
            c = value[:mid] + ("..." if mid < len(value) else "")
            if len(json.dumps(c)) <= max_length:
                out = c; lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


def mm_orders(name: str, od: OrderDepth, pos: int, limit: int,
              gamma: float, fair: float | None = None) -> list[Order]:
    """
    Pure market-maker: always posts a bid and an ask inside the current spread,
    skewed by inventory.

    reservation = mid - gamma * pos
      - positive pos → reservation < mid → we quote lower (prefer selling)
      - negative pos → reservation > mid → we quote higher (prefer buying)

    We then overbid the best existing bid below reservation, and undercut
    the best existing ask above reservation.  This guarantees both sides
    are always live regardless of where the price is trading.

    Optional `fair`: if set, also take orders that are clearly mispriced
    (ask far below fair or bid far above fair) with a wide edge to avoid
    chasing trends.
    """
    if not od.buy_orders or not od.sell_orders:
        return []

    bb  = max(od.buy_orders)
    ba  = min(od.sell_orders)
    mid = (bb + ba) / 2

    mx_b = limit - pos
    mx_s = limit + pos
    orders: list[Order] = []

    # Optional take: only for very obvious mispricings (e.g. HP_FAIR ± 15)
    if fair is not None:
        TAKE_EDGE = 15
        for ap in sorted(od.sell_orders):
            if ap <= fair - TAKE_EDGE and mx_b > 0:
                v = min(abs(od.sell_orders[ap]), mx_b)
                orders.append(Order(name, ap, v)); mx_b -= v
        for bp in sorted(od.buy_orders, reverse=True):
            if bp >= fair + TAKE_EDGE and mx_s > 0:
                v = min(od.buy_orders[bp], mx_s)
                orders.append(Order(name, bp, -v)); mx_s -= v

    # Passive quotes anchored to MARKET MID (not a fixed fair value)
    reservation = mid - gamma * pos

    bb_below = max((p for p in od.buy_orders  if p < reservation), default=None)
    ba_above = min((p for p in od.sell_orders if p > reservation), default=None)

    if bb_below is not None and mx_b > 0:
        orders.append(Order(name, bb_below + 1, mx_b))
    if ba_above is not None and mx_s > 0:
        orders.append(Order(name, ba_above - 1, -mx_s))

    return orders


class Trader:

    def run(self, state: TradingState):
        result: dict[Symbol, list[Order]] = {}

        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = mm_orders(
                "HYDROGEL_PACK",
                state.order_depths["HYDROGEL_PACK"],
                state.position.get("HYDROGEL_PACK", 0),
                HP_LIMIT, HP_GAMMA,
                fair=HP_FAIR,   # take orders only at HP_FAIR ± 15
            )

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            result["VELVETFRUIT_EXTRACT"] = mm_orders(
                "VELVETFRUIT_EXTRACT",
                state.order_depths["VELVETFRUIT_EXTRACT"],
                state.position.get("VELVETFRUIT_EXTRACT", 0),
                VEV_LIMIT, VEV_GAMMA,
                fair=None,  # no fixed fair for VEV; pure spread-capture
            )

        logger.flush(state, result, 0, "")
        return result, 0, ""

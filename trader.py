from typing import Any
import json

from datamodel import Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Listing, Observation

# ── Config ────────────────────────────────────────────────────────────────────
# With GAMMA=g and spread half-width=s, effective max inventory ≈ s/g.
# HP spread=16 (half=8), GAMMA=0.25 → max pos ≈ 32 units → low variance.
# VEV spread=5  (half=2.5), GAMMA=0.15 → max pos ≈ 17 units.

HP_GAMMA  = 0.15
HP_LIMIT  = 200

VEV_GAMMA = 0.15
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
            self._trades(state.own_trades), self._trades(state.market_trades),
            state.position, self.compress_observations(state.observations),
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


# ── Market-making helper ──────────────────────────────────────────────────────

def mm_orders(name: str, od: OrderDepth, pos: int, limit: int, gamma: float) -> list[Order]:
    """
    Pure passive market maker. No take phase — that was causing us to buy
    200 units into every HP downtrend.

    Posts:
      bid  at  best_bid_below_reservation  + 1   (overbid inside spread)
      ask  at  best_ask_above_reservation  - 1   (undercut inside spread)

    where reservation = mid - gamma * pos.

    As pos grows positive  → reservation falls → bid moves down, ask moves down
      → market naturally sells to us less and we sell more → inventory reverts.
    As pos grows negative  → reservation rises → same logic in reverse.

    With gamma=0.25 and HP spread half-width≈8: effective max pos ≈ 8/0.25 = 32.
    """
    if not od.buy_orders or not od.sell_orders:
        return []

    bb  = max(od.buy_orders)
    ba  = min(od.sell_orders)
    mid = (bb + ba) / 2
    mx_b = limit - pos
    mx_s = limit + pos

    reservation = mid - gamma * pos

    bb_below = max((p for p in od.buy_orders  if p < reservation), default=None)
    ba_above = min((p for p in od.sell_orders if p > reservation), default=None)

    orders: list[Order] = []
    if bb_below is not None and mx_b > 0:
        orders.append(Order(name, bb_below + 1, mx_b))
    if ba_above is not None and mx_s > 0:
        orders.append(Order(name, ba_above - 1, -mx_s))
    return orders


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        result: dict[Symbol, list[Order]] = {}

        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = mm_orders(
                "HYDROGEL_PACK",
                state.order_depths["HYDROGEL_PACK"],
                state.position.get("HYDROGEL_PACK", 0),
                HP_LIMIT, HP_GAMMA,
            )

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            result["VELVETFRUIT_EXTRACT"] = mm_orders(
                "VELVETFRUIT_EXTRACT",
                state.order_depths["VELVETFRUIT_EXTRACT"],
                state.position.get("VELVETFRUIT_EXTRACT", 0),
                VEV_LIMIT, VEV_GAMMA,
            )

        logger.flush(state, result, 0, "")
        return result, 0, ""

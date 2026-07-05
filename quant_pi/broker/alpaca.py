from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import (
    NewsRequest,
    OptionChainRequest,
    OptionLatestQuoteRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

ET = ZoneInfo("America/New_York")


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float
    day_pl: Optional[float]


@dataclass
class Position:
    symbol: str
    qty: float
    market_value: float
    unrealized_pl: float
    avg_entry_price: float
    side: str = "long"               # 'long' or 'short'
    cost_basis: float = 0.0          # qty * avg_entry_price (Alpaca-provided)
    change_today_pl: float = 0.0     # today's intraday $ P/L on this position
    change_today_pct: float = 0.0    # today's intraday % change of price
    current_price: float = 0.0
    lastday_price: float = 0.0


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    mid: float


class Broker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        self.trading = TradingClient(api_key, secret_key, paper=paper)
        self.data = StockHistoricalDataClient(api_key, secret_key)
        self.news_client = NewsClient(api_key, secret_key)
        self.option_data = OptionHistoricalDataClient(api_key, secret_key)
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper

    def account(self) -> Account:
        a = self.trading.get_account()
        equity = float(a.equity)
        last_equity = float(a.last_equity) if a.last_equity else equity
        return Account(
            equity=equity,
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            day_pl=equity - last_equity,
        )

    def positions(self) -> list[Position]:
        out = []
        for p in self.trading.get_all_positions():
            def _f(attr, default=0.0):
                v = getattr(p, attr, None)
                try:
                    return float(v) if v is not None else default
                except (TypeError, ValueError):
                    return default
            side = getattr(p, "side", None)
            side_str = (getattr(side, "value", None) or str(side or "long")).lower()
            out.append(Position(
                symbol=p.symbol,
                qty=_f("qty"),
                market_value=_f("market_value"),
                unrealized_pl=_f("unrealized_pl"),
                avg_entry_price=_f("avg_entry_price"),
                side=side_str,
                cost_basis=_f("cost_basis"),
                change_today_pl=_f("unrealized_intraday_pl"),
                change_today_pct=_f("change_today") * 100.0,  # Alpaca returns decimal (0.0123)
                current_price=_f("current_price"),
                lastday_price=_f("lastday_price"),
            ))
        return out

    def quote(self, symbol: str) -> Quote:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        result = self.data.get_stock_latest_quote(req)
        q = result[symbol]
        bid = float(q.bid_price) if q.bid_price else 0.0
        ask = float(q.ask_price) if q.ask_price else 0.0
        mid = (bid + ask) / 2 if bid and ask else max(bid, ask)
        return Quote(symbol=symbol, bid=bid, ask=ask, mid=mid)

    def recent_bars(self, symbol: str, days: int = 30) -> list[dict]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        # Free paper-data tier blocks the most recent 15 min of SIP data.
        # IEX feed has full history including current day with no delay.
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        result = self.data.get_stock_bars(req)
        bars = result.data.get(symbol, [])
        return [
            {"t": b.timestamp.isoformat(), "o": float(b.open), "h": float(b.high),
             "l": float(b.low), "c": float(b.close), "v": int(b.volume)}
            for b in bars
        ]

    def news(self, symbol: str, lookback_hours: int = 48, limit: int = 10) -> list[dict]:
        """Recent news headlines for a symbol from Alpaca's news API.
        Returns the most recent first. Summary is truncated to keep prompt cheap."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback_hours)
        req = NewsRequest(
            symbols=symbol.upper(),
            start=start,
            end=end,
            limit=min(max(limit, 1), 50),
            sort="desc",
            include_content=False,
            exclude_contentless=True,
        )
        result = self.news_client.get_news(req)
        items = result.data.get("news", []) if hasattr(result, "data") else getattr(result, "news", [])
        out = []
        for n in items:
            headline = getattr(n, "headline", "") or ""
            summary = (getattr(n, "summary", "") or "")[:280]
            ts = getattr(n, "created_at", None) or getattr(n, "updated_at", None)
            out.append({
                "ts": ts.isoformat() if ts else "",
                "headline": headline,
                "summary": summary,
                "source": getattr(n, "source", "") or "",
                "url": getattr(n, "url", "") or "",
            })
        return out

    def submit_market_order(self, symbol: str, side: str, qty: Optional[float] = None,
                            notional: Optional[float] = None) -> str:
        if (qty is None) == (notional is None):
            raise ValueError("Provide exactly one of qty or notional")
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            notional=notional,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading.submit_order(req)
        return str(order.id)

    def submit_limit_order(self, symbol: str, side: str, limit_price: float,
                           qty: Optional[float] = None, notional: Optional[float] = None,
                           extended_hours: bool = False) -> str:
        """Limit order. Set extended_hours=True for pre/post/24-5 sessions."""
        if (qty is None) == (notional is None):
            raise ValueError("Provide exactly one of qty or notional")
        # Alpaca's extended-hours limit orders require qty (not notional)
        if extended_hours and notional is not None:
            raise ValueError("Extended-hours orders must use qty, not notional")
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            notional=notional,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            extended_hours=extended_hours,
        )
        order = self.trading.submit_order(req)
        return str(order.id)

    def is_market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    def portfolio_history(self, period: str = "1D", timeframe: str = "5Min",
                           date_start: Optional[str] = None) -> list[dict]:
        """Alpaca's official portfolio history — same endpoint that powers
        their web dashboard chart. Continuous server-side, includes off-hours,
        no gaps from local dashboard downtime.

        period:     '1D', '1W', '1M', '3M', '1A', 'all'  (ignored if date_start)
        timeframe:  '1Min', '5Min', '15Min', '1H', '1D'
        date_start: ISO date 'YYYY-MM-DD' — overrides period to anchor at a
                    specific start (e.g. today's midnight for a true 1D view).

        Returns: [{'ts': ISO string, 'equity': float}, ...]
        """
        import requests
        base = "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets"
        params: dict = {"timeframe": timeframe, "extended_hours": "true"}
        if date_start:
            params["date_start"] = date_start
        else:
            params["period"] = period
        try:
            r = requests.get(
                f"{base}/v2/account/portfolio/history",
                params=params,
                headers={
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.secret_key,
                },
                timeout=8,
            )
            if r.status_code != 200:
                return []
            data = r.json()
        except Exception:
            return []
        equities = data.get("equity") or []
        stamps = data.get("timestamp") or []
        out = []
        for ts_unix, eq in zip(stamps, equities):
            if eq is None:
                continue
            try:
                ts_iso = datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat()
            except Exception:
                continue
            out.append({"ts": ts_iso, "equity": float(eq)})
        return out

    # Symbols Alpaca explicitly supports for 24-5 overnight trading.
    # Source: Alpaca's overnight-trading documentation. Conservative subset of
    # major ETFs + mega-caps. Update as Alpaca expands the list.
    OVERNIGHT_TRADABLE = frozenset({
        "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "EEM", "GLD", "SLV", "TLT",
        "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
        "NFLX", "AMD", "INTC", "BAC", "JPM", "WMT", "DIS", "V", "MA", "PYPL",
        "COIN", "MSTR", "PLTR", "SHOP", "SOFI", "CRWD", "ROKU", "UBER",
    })

    def is_extended_eligible(self, symbol: str) -> bool:
        """Whether Alpaca will accept an extended-hours order for this symbol."""
        return symbol.upper() in self.OVERNIGHT_TRADABLE

    def _get_asset_cached(self, symbol: str):
        """Per-instance asset cache to avoid hammering Alpaca with identical lookups."""
        if not hasattr(self, "_asset_cache"):
            self._asset_cache: dict = {}
        sym = symbol.upper()
        if sym in self._asset_cache:
            return self._asset_cache[sym]
        asset = self.trading.get_asset(sym)
        self._asset_cache[sym] = asset
        return asset

    def is_tradable(self, symbol: str) -> tuple[bool, str]:
        """Verify a symbol exists and is tradeable on Alpaca.
        Returns (ok, reason). Cached per Broker instance for the cycle's lifetime."""
        if not hasattr(self, "_tradable_cache"):
            self._tradable_cache: dict[str, tuple[bool, str]] = {}
        sym = symbol.upper()
        if sym in self._tradable_cache:
            return self._tradable_cache[sym]
        try:
            asset = self._get_asset_cached(sym)
        except Exception as e:
            result = (False, f"unknown symbol: {e}")
            self._tradable_cache[sym] = result
            return result
        if not getattr(asset, "tradable", False):
            result = (False, f"{sym} is not tradable on Alpaca")
        else:
            status = getattr(asset, "status", None)
            # Alpaca returns AssetStatus enum (e.g. AssetStatus.ACTIVE).
            status_value = getattr(status, "value", None)
            status_str = str(status_value if status_value is not None else status).lower()
            if status is not None and status_str != "active":
                result = (False, f"{sym} is not active (status={status_str})")
            else:
                result = (True, "ok")
        self._tradable_cache[sym] = result
        return result

    def is_shortable(self, symbol: str) -> tuple[bool, str]:
        """Whether Alpaca will accept a short-sale on this symbol — must be
        marked `shortable=True` AND `easy_to_borrow=True`. Hard-to-borrow
        names are rejected even if shorting is enabled on the account."""
        try:
            asset = self._get_asset_cached(symbol)
        except Exception as e:
            return (False, f"unknown symbol: {e}")
        if not getattr(asset, "shortable", False):
            return (False, f"{symbol} is not shortable on Alpaca")
        if not getattr(asset, "easy_to_borrow", False):
            return (False, f"{symbol} is hard-to-borrow — short rejected")
        return (True, "ok")

    def shorting_enabled(self) -> bool:
        """Whether the connected paper/live account allows shorts at all."""
        try:
            return bool(self.trading.get_account().shorting_enabled)
        except Exception:
            return False

    def open_orders(self) -> list:
        """Currently working orders (status=open: new, accepted, partially_filled, etc.)."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)
        return list(self.trading.get_orders(req))

    def cancel_order(self, order_id: str) -> None:
        self.trading.cancel_order_by_id(order_id)

    def get_order(self, order_id: str):
        """Fetch a single order's current state from Alpaca."""
        return self.trading.get_order_by_id(order_id)

    def cancel_stale_orders(self, max_age_minutes: int = 30) -> list[str]:
        """Cancel any open orders older than max_age_minutes. Returns list of canceled order IDs."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
        canceled: list[str] = []
        for o in self.open_orders():
            submitted_at = getattr(o, "submitted_at", None) or getattr(o, "created_at", None)
            if submitted_at is None:
                continue
            # Alpaca returns timezone-aware datetimes
            if submitted_at < cutoff:
                try:
                    self.trading.cancel_order_by_id(str(o.id))
                    canceled.append(str(o.id))
                except Exception:
                    pass
        return canceled

    @staticmethod
    def trade_window_status(windows: list) -> tuple:
        """Check current ET time against a list of 'HH:MM-HH:MM' windows.
        Returns (in_window: bool, status_msg: str).
        Empty windows = always in window."""
        if not windows:
            return (True, "always-on")
        now = datetime.now(ET)
        cur_min = now.hour * 60 + now.minute
        parsed = []
        for w in windows:
            try:
                start_str, end_str = w.split("-")
                sh, sm = map(int, start_str.split(":"))
                eh, em = map(int, end_str.split(":"))
                parsed.append((sh * 60 + sm, eh * 60 + em, w))
            except (ValueError, AttributeError):
                continue
        for start_min, end_min, w in parsed:
            if start_min <= cur_min < end_min:
                return (True, f"in window {w} ET")
        # Outside windows — find next start time today
        future = [(s, w) for s, _, w in parsed if s > cur_min]
        if future:
            s, w = min(future)
            hh, mm = divmod(s, 60)
            return (False, f"outside windows; next {hh:02d}:{mm:02d} ET")
        # No more windows today — first window tomorrow
        if parsed:
            first_start = min(s for s, _, _ in parsed)
            hh, mm = divmod(first_start, 60)
            return (False, f"outside windows; next {hh:02d}:{mm:02d} ET (tomorrow)")
        return (False, "outside windows")

    # ── OPTIONS ───────────────────────────────────────────────────
    # OCC-format option symbol: AAPL241220C00200000 = AAPL Dec 20 2024 $200 Call
    # Format: SYMBOL + YYMMDD + C/P + strike×1000 padded to 8 digits

    def options_enabled(self) -> bool:
        """Whether options trading is approved on the connected account."""
        try:
            acc = self.trading.get_account()
            level = getattr(acc, "options_trading_level", 0) or 0
            return int(level) >= 1
        except Exception:
            return False

    def option_chain(self, underlying: str, max_dte: int = 60,
                     option_type: Optional[str] = None,
                     limit: int = 30) -> list[dict]:
        """List active option contracts for an underlying.
        - max_dte: only return contracts expiring within N days
        - option_type: 'call', 'put', or None for both
        - limit: cap the count (sorted by expiration, then |strike − spot|)"""
        from alpaca.trading.enums import ContractType, AssetStatus
        end = (datetime.now(timezone.utc) + timedelta(days=max_dte)).date()
        ctype = None
        if option_type:
            ctype = ContractType.CALL if option_type.lower() == "call" else ContractType.PUT
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying.upper()],
            status=AssetStatus.ACTIVE,
            expiration_date_lte=end,
            type=ctype,
            limit=200,
        )
        contracts = self.trading.get_option_contracts(req).option_contracts or []
        try:
            spot = self.quote(underlying).mid
        except Exception:
            spot = 0.0
        out = []
        for c in contracts:
            strike = float(c.strike_price)
            out.append({
                "symbol": c.symbol,
                "underlying": c.underlying_symbol,
                "type": c.type.value if hasattr(c.type, "value") else str(c.type).lower(),
                "strike": strike,
                "expiration": c.expiration_date.isoformat() if c.expiration_date else "",
                "moneyness_pct": ((strike - spot) / spot * 100) if spot else 0.0,
                "tradable": bool(getattr(c, "tradable", True)),
            })
        # Sort by expiration ascending, then by closeness to spot
        out.sort(key=lambda x: (x["expiration"], abs(x["moneyness_pct"])))
        return out[:limit]

    def option_quote(self, option_symbol: str) -> dict:
        """Latest bid/ask + greek-friendly mid for a single option contract."""
        req = OptionLatestQuoteRequest(symbol_or_symbols=option_symbol)
        result = self.option_data.get_option_latest_quote(req)
        q = result.get(option_symbol)
        if q is None:
            return {"error": f"no quote for {option_symbol}"}
        bid = float(q.bid_price) if q.bid_price else 0.0
        ask = float(q.ask_price) if q.ask_price else 0.0
        mid = (bid + ask) / 2 if bid and ask else max(bid, ask)
        return {
            "symbol": option_symbol,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "bid_size": int(q.bid_size or 0),
            "ask_size": int(q.ask_size or 0),
        }

    def submit_option_order(self, option_symbol: str, side: str,
                             qty: int, limit_price: Optional[float] = None) -> str:
        """Submit a single-leg option order. qty = number of CONTRACTS
        (each = 100 shares of underlying). limit_price required for safety —
        we don't allow market orders on options because spreads can be wide."""
        if qty <= 0:
            raise ValueError("qty must be > 0")
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        if limit_price is None:
            raise ValueError("Options require limit_price (no market orders for safety)")
        req = LimitOrderRequest(
            symbol=option_symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
        )
        order = self.trading.submit_order(req)
        return str(order.id)

    def current_session(self) -> str:
        """Detect Alpaca session from wall clock in ET. Returns one of:
          'regular'  — Mon-Fri 9:30am-4:00pm ET. Market orders OK.
          'extended' — pre/post-market & 24-5 overnight. Limit orders only.
          'closed'   — weekends + Fri 8pm to Sun 8pm ET window. No orders.
        """
        now = datetime.now(ET)
        weekday = now.weekday()  # 0 Mon, 6 Sun
        mins = now.hour * 60 + now.minute

        # Saturday: closed all day
        if weekday == 5:
            return "closed"
        # Sunday before 8 PM ET (24-5 session opens at 8 PM ET Sunday)
        if weekday == 6 and mins < 20 * 60:
            return "closed"
        # Friday after 8 PM ET (24-5 session closes Fri 8 PM ET)
        if weekday == 4 and mins >= 20 * 60:
            return "closed"

        # Mon-Fri regular hours: 9:30 AM (570) to 4:00 PM (960) ET
        if 0 <= weekday <= 4 and 570 <= mins < 960:
            return "regular"

        # Otherwise: extended (Sun-Fri overnight, Mon-Fri pre/post)
        return "extended"


def main() -> None:
    import sys
    from pathlib import Path
    from quant_pi.config import load_config, require_keys

    cfg = load_config(Path(__file__).resolve().parents[2] / "config.toml")
    require_keys(cfg)
    b = Broker(cfg.alpaca_api_key, cfg.alpaca_secret_key, paper=cfg.alpaca_paper)
    acc = b.account()
    print(f"Equity:       ${acc.equity:,.2f}")
    print(f"Cash:         ${acc.cash:,.2f}")
    print(f"Buying power: ${acc.buying_power:,.2f}")
    print(f"Day P/L:      ${acc.day_pl:,.2f}" if acc.day_pl is not None else "Day P/L: n/a")
    print(f"Market open:  {b.is_market_open()}")
    pos = b.positions()
    print(f"Positions:    {len(pos)}")
    for p in pos:
        print(f"  {p.symbol:6s} qty={p.qty:>8.2f} mv=${p.market_value:>10,.2f} upl=${p.unrealized_pl:>+8,.2f}")


if __name__ == "__main__":
    main()

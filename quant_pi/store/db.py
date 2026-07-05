import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracked_symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    reason TEXT NOT NULL,
    formula TEXT,
    metric_value REAL,
    threshold REAL,
    expires_at TEXT,
    status TEXT NOT NULL DEFAULT 'watching',
    trade_type TEXT NOT NULL DEFAULT 'long'  -- 'long', 'short', 'call', 'put'
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    summary TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    raw_json TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    notional REAL,
    status TEXT NOT NULL,
    order_id TEXT,
    reject_reason TEXT,
    filled_avg_price REAL
);
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    day_pl REAL
);
CREATE TABLE IF NOT EXISTS flags (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decision_formulas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER NOT NULL,
    ts TEXT NOT NULL,
    formula TEXT NOT NULL,
    FOREIGN KEY (decision_id) REFERENCES decisions(id)
);
CREATE INDEX IF NOT EXISTS idx_decision_formulas_formula ON decision_formulas(formula);
CREATE INDEX IF NOT EXISTS idx_decision_formulas_decision ON decision_formulas(decision_id);
"""

# Canonical list of formulas tracked across the agent + dashboard.
# Ordered longest-first so multi-word names match before their substrings
# (e.g. "Value at Risk" before "VaR" — though VaR is also in the list as an alias
# we deduplicate after matching).
FORMULAS = [
    "Black-Scholes",
    "Value at Risk",
    "Monte Carlo",
    "Yield Curve",
    "CAPM",
    "Sharpe",
    "ARIMA",
    "Beta",
    "Duration",
    "VaR",
]


def extract_formulas(text: str) -> list[str]:
    """Scan text for known formula names. Returns deduplicated list, preserving
    the canonical FORMULAS order. Treats VaR as an alias of Value at Risk."""
    if not text:
        return []
    lower = text.lower()
    found = [f for f in FORMULAS if f.lower() in lower]
    if "VaR" in found and "Value at Risk" in found:
        found.remove("VaR")
    return found


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as c:
        c.executescript(SCHEMA)
        # Idempotent migrations
        cols = [r["name"] for r in c.execute("PRAGMA table_info(tracked_symbols)").fetchall()]
        if "trade_type" not in cols:
            c.execute("ALTER TABLE tracked_symbols ADD COLUMN trade_type TEXT NOT NULL DEFAULT 'long'")
        cols = [r["name"] for r in c.execute("PRAGMA table_info(trades)").fetchall()]
        if "filled_avg_price" not in cols:
            c.execute("ALTER TABLE trades ADD COLUMN filled_avg_price REAL")
        cur = c.execute("SELECT value FROM flags WHERE key = 'PAUSED'")
        if cur.fetchone() is None:
            c.execute("INSERT INTO flags (key, value) VALUES ('PAUSED', '0')")
    backfill_decision_formulas(db_path)


def backfill_decision_formulas(db_path: Path) -> None:
    """One-shot scan of historical decisions whose formula rows haven't been
    populated yet. Cheap (only runs on rows with no children) and idempotent."""
    with connect(db_path) as c:
        rows = c.execute(
            "SELECT id, ts, summary FROM decisions d "
            "WHERE NOT EXISTS (SELECT 1 FROM decision_formulas f WHERE f.decision_id = d.id)"
        ).fetchall()
        for r in rows:
            for f in extract_formulas(r["summary"] or ""):
                c.execute(
                    "INSERT INTO decision_formulas (decision_id, ts, formula) VALUES (?, ?, ?)",
                    (r["id"], r["ts"], f),
                )


def is_paused(db_path: Path) -> bool:
    with connect(db_path) as c:
        row = c.execute("SELECT value FROM flags WHERE key = 'PAUSED'").fetchone()
        return row is not None and row["value"] == "1"


def set_paused(db_path: Path, paused: bool) -> None:
    with connect(db_path) as c:
        c.execute("UPDATE flags SET value = ? WHERE key = 'PAUSED'", ("1" if paused else "0",))


def log_decision(db_path: Path, summary: str, reasoning: str, raw_json: Optional[str] = None) -> None:
    ts = now_iso()
    with connect(db_path) as c:
        cur = c.execute(
            "INSERT INTO decisions (ts, summary, reasoning, raw_json) VALUES (?, ?, ?, ?)",
            (ts, summary, reasoning, raw_json),
        )
        decision_id = cur.lastrowid
        for f in extract_formulas(summary or ""):
            c.execute(
                "INSERT INTO decision_formulas (decision_id, ts, formula) VALUES (?, ?, ?)",
                (decision_id, ts, f),
            )


def formula_frequency(db_path: Path, limit_decisions: Optional[int] = None) -> list[tuple[str, int]]:
    """Count formula citations across recent decisions.
    Returns [(formula, count), ...] sorted by count descending.
    If limit_decisions is set, restrict to the most recent N decisions."""
    with connect(db_path) as c:
        if limit_decisions:
            rows = c.execute(
                "SELECT formula, COUNT(*) AS n FROM decision_formulas "
                "WHERE decision_id IN ("
                "  SELECT id FROM decisions ORDER BY id DESC LIMIT ?"
                ") GROUP BY formula ORDER BY n DESC",
                (limit_decisions,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT formula, COUNT(*) AS n FROM decision_formulas "
                "GROUP BY formula ORDER BY n DESC"
            ).fetchall()
    return [(r["formula"], int(r["n"])) for r in rows]


def log_trade(db_path: Path, symbol: str, side: str, qty: float, notional: Optional[float],
              status: str, order_id: Optional[str] = None, reject_reason: Optional[str] = None) -> None:
    with connect(db_path) as c:
        c.execute(
            "INSERT INTO trades (ts, symbol, side, qty, notional, status, order_id, reject_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now_iso(), symbol, side, qty, notional, status, order_id, reject_reason),
        )


def log_equity(db_path: Path, equity: float, cash: float, day_pl: Optional[float]) -> None:
    with connect(db_path) as c:
        c.execute(
            "INSERT INTO equity_snapshots (ts, equity, cash, day_pl) VALUES (?, ?, ?, ?)",
            (now_iso(), equity, cash, day_pl),
        )


TERMINAL_STATUSES = {"filled", "canceled", "cancelled", "expired", "rejected", "replaced"}


def open_trade_order_ids(db_path: Path) -> list[str]:
    """Order IDs of trades not yet in a terminal status. These need polling."""
    with connect(db_path) as c:
        rows = c.execute(
            "SELECT order_id, status FROM trades WHERE order_id IS NOT NULL"
        ).fetchall()
    return [r["order_id"] for r in rows if (r["status"] or "").lower() not in TERMINAL_STATUSES]


def update_trade_status(db_path: Path, order_id: str, status: str,
                        reject_reason: Optional[str] = None,
                        filled_qty: Optional[float] = None,
                        filled_notional: Optional[float] = None,
                        filled_avg_price: Optional[float] = None) -> None:
    with connect(db_path) as c:
        sets = ["status = ?"]
        params: list = [status]
        if reject_reason is not None:
            sets.append("reject_reason = ?")
            params.append(reject_reason)
        if filled_qty is not None and filled_qty > 0:
            sets.append("qty = ?")
            params.append(filled_qty)
        if filled_notional is not None and filled_notional > 0:
            sets.append("notional = ?")
            params.append(filled_notional)
        if filled_avg_price is not None and filled_avg_price > 0:
            sets.append("filled_avg_price = ?")
            params.append(filled_avg_price)
        params.append(order_id)
        c.execute(
            f"UPDATE trades SET {', '.join(sets)} WHERE order_id = ?",
            params,
        )


def matching_buy_for_sell(db_path: Path, symbol: str, sell_id: int) -> Optional[sqlite3.Row]:
    """For computing realized P&L on a SELL: find the most recent FILLED BUY
    of the same symbol that landed BEFORE this sell. Returns None if no
    matching long entry exists."""
    with connect(db_path) as c:
        return c.execute(
            "SELECT id, ts, qty, filled_avg_price, notional FROM trades "
            "WHERE symbol = ? AND side = 'buy' AND status = 'filled' "
            "  AND id < ? AND filled_avg_price IS NOT NULL "
            "ORDER BY id DESC LIMIT 1",
            (symbol, sell_id),
        ).fetchone()


def trades_today(db_path: Path) -> int:
    today = datetime.now(timezone.utc).date().isoformat()
    with connect(db_path) as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE status = 'submitted' AND ts >= ?",
            (today,),
        ).fetchone()
        return int(row["n"]) if row else 0


def latest_decision(db_path: Path) -> Optional[sqlite3.Row]:
    with connect(db_path) as c:
        return c.execute(
            "SELECT ts, summary FROM decisions ORDER BY id DESC LIMIT 1"
        ).fetchone()


def recent_decisions(db_path: Path, limit: int = 20) -> list[sqlite3.Row]:
    with connect(db_path) as c:
        return list(c.execute(
            "SELECT id, ts, summary, reasoning FROM decisions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall())


def trades_around(db_path: Path, ts_iso: str, window_seconds: int = 180) -> list[sqlite3.Row]:
    """Fetch trades within +/- window_seconds of the given ISO timestamp.
    Excludes pure rate-limit rejections — they're transient and don't
    represent real agent decisions worth showing in History."""
    from datetime import datetime, timedelta
    try:
        center = datetime.fromisoformat(ts_iso)
    except ValueError:
        return []
    lo = (center - timedelta(seconds=window_seconds)).isoformat()
    hi = (center + timedelta(seconds=window_seconds)).isoformat()
    with connect(db_path) as c:
        return list(c.execute(
            "SELECT id, ts, symbol, side, qty, notional, status, "
            "       reject_reason, filled_avg_price "
            "FROM trades WHERE ts BETWEEN ? AND ? "
            "  AND NOT (status = 'rejected' AND reject_reason LIKE '%42910000%') "
            "  AND NOT (status = 'rejected' AND reject_reason LIKE '%rate limit%') "
            "ORDER BY id ASC",
            (lo, hi),
        ).fetchall())


def recent_rejections(db_path: Path, limit: int = 8) -> list[sqlite3.Row]:
    """Recent rejected trades — feed back into the agent's prompt so it stops
    repeating real mistakes. Filters out transient broker errors (rate limits,
    network blips) that aren't agent-decision issues."""
    with connect(db_path) as c:
        return list(c.execute(
            "SELECT ts, symbol, side, reject_reason FROM trades "
            "WHERE status = 'rejected' AND reject_reason IS NOT NULL "
            "  AND reject_reason NOT LIKE '%42910000%' "
            "  AND reject_reason NOT LIKE '%rate limit%' "
            "  AND reject_reason NOT LIKE '%broker error%' "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall())


def add_tracked_symbol(db_path: Path, symbol: str, reason: str,
                        formula: Optional[str] = None,
                        metric_value: Optional[float] = None,
                        threshold: Optional[float] = None,
                        expires_at: Optional[str] = None,
                        trade_type: str = "long") -> None:
    """Agent's 'watch this for later' note. If the symbol is already tracked,
    update its reason/metric instead of duplicating.
    trade_type: 'long', 'short', 'call', or 'put'."""
    tt = (trade_type or "long").lower()
    if tt not in ("long", "short", "call", "put"):
        tt = "long"
    with connect(db_path) as c:
        existing = c.execute(
            "SELECT id FROM tracked_symbols WHERE symbol = ? AND status = 'watching'",
            (symbol.upper(),),
        ).fetchone()
        if existing:
            c.execute(
                "UPDATE tracked_symbols SET ts = ?, reason = ?, formula = ?, "
                "metric_value = ?, threshold = ?, expires_at = ?, trade_type = ? WHERE id = ?",
                (now_iso(), reason, formula, metric_value, threshold, expires_at, tt, existing["id"]),
            )
        else:
            c.execute(
                "INSERT INTO tracked_symbols (ts, symbol, reason, formula, metric_value, threshold, expires_at, trade_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (now_iso(), symbol.upper(), reason, formula, metric_value, threshold, expires_at, tt),
            )


def remove_tracked_symbol(db_path: Path, symbol: str, status: str = "removed") -> None:
    with connect(db_path) as c:
        c.execute(
            "UPDATE tracked_symbols SET status = ? WHERE symbol = ? AND status = 'watching'",
            (status, symbol.upper()),
        )


def tracked_symbols(db_path: Path, limit: int = 20) -> list[sqlite3.Row]:
    with connect(db_path) as c:
        return list(c.execute(
            "SELECT ts, symbol, reason, formula, metric_value, threshold, expires_at, trade_type "
            "FROM tracked_symbols WHERE status = 'watching' "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall())


def equity_history(db_path: Path, limit: int = 100) -> list[sqlite3.Row]:
    """Most recent N equity snapshots, oldest first (for plotting)."""
    with connect(db_path) as c:
        rows = list(c.execute(
            "SELECT ts, equity FROM equity_snapshots ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall())
    return list(reversed(rows))


def equity_near(db_path: Path, ts_iso: str) -> Optional[sqlite3.Row]:
    """Most recent equity snapshot at or before the given timestamp."""
    with connect(db_path) as c:
        return c.execute(
            "SELECT ts, equity, cash, day_pl FROM equity_snapshots "
            "WHERE ts <= ? ORDER BY id DESC LIMIT 1",
            (ts_iso,),
        ).fetchone()

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    vault_path: Path
    vault_subfolder: Optional[str]
    brain_root_note: Optional[str]
    brain_max_depth: int
    watchlist: list[str]
    cycle_minutes: int
    max_position_pct: float
    max_daily_trades: int
    max_order_notional: float
    max_trades_per_cycle: int
    auto_cancel_minutes: int
    trade_windows: list
    offhours_cycle_hours: int
    sim_capital: Optional[float]  # if set, bot trades as if this is total equity
    provider: str  # "anthropic" or "ollama"
    model: str
    ollama_host: str
    ollama_model: str
    max_tokens: int
    display_width: int
    display_height: int
    display_poll_seconds: float
    display_always_on_top: bool
    display_sleep_after_minutes: float
    db_path: Path
    anthropic_api_key: str
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_paper: bool
    finnhub_api_key: str
    marketaux_api_key: str
    newsapi_api_key: str


def load_config(config_path: Path | str = "config.toml") -> Config:
    load_dotenv(override=True)
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    project_root = Path(config_path).resolve().parent

    return Config(
        vault_path=Path(raw["brain"]["vault_path"]),
        vault_subfolder=raw["brain"].get("subfolder"),
        brain_root_note=raw["brain"].get("root_note"),
        brain_max_depth=int(raw["brain"].get("max_depth", 3)),
        watchlist=raw["trading"]["watchlist"],
        cycle_minutes=raw["trading"]["cycle_minutes"],
        max_position_pct=raw["trading"]["max_position_pct"],
        max_daily_trades=raw["trading"]["max_daily_trades"],
        max_order_notional=raw["trading"]["max_order_notional"],
        max_trades_per_cycle=int(raw["trading"].get("max_trades_per_cycle", 1)),
        auto_cancel_minutes=int(raw["trading"].get("auto_cancel_minutes", 30)),
        trade_windows=list(raw["trading"].get("trade_windows", [])),
        offhours_cycle_hours=int(raw["trading"].get("offhours_cycle_hours", 6)),
        sim_capital=float(raw["trading"]["sim_capital"]) if raw["trading"].get("sim_capital") else None,
        provider=raw["agent"].get("provider", "anthropic"),
        model=raw["agent"]["model"],
        ollama_host=raw["agent"].get("ollama_host", "http://localhost:11434"),
        ollama_model=raw["agent"].get("ollama_model", "qwen2.5:7b"),
        max_tokens=raw["agent"]["max_tokens"],
        display_width=raw["display"]["width"],
        display_height=raw["display"]["height"],
        display_poll_seconds=raw["display"]["poll_seconds"],
        display_always_on_top=raw["display"]["always_on_top"],
        display_sleep_after_minutes=float(raw["display"].get("sleep_after_minutes", 2.5)),
        db_path=project_root / raw["store"]["db_path"],
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        alpaca_api_key=os.environ.get("ALPACA_API_KEY", ""),
        alpaca_secret_key=os.environ.get("ALPACA_SECRET_KEY", ""),
        alpaca_paper=os.environ.get("ALPACA_PAPER", "true").lower() == "true",
        finnhub_api_key=os.environ.get("FINNHUB_API_KEY", ""),
        marketaux_api_key=os.environ.get("MARKETAUX_API_KEY", ""),
        newsapi_api_key=os.environ.get("NEWSAPI_API_KEY", ""),
    )


def require_keys(cfg: Config) -> None:
    required = {
        "ALPACA_API_KEY": cfg.alpaca_api_key,
        "ALPACA_SECRET_KEY": cfg.alpaca_secret_key,
    }
    if cfg.provider == "anthropic":
        required["ANTHROPIC_API_KEY"] = cfg.anthropic_api_key
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}. Copy .env.example to .env and fill in.", file=sys.stderr)
        sys.exit(1)

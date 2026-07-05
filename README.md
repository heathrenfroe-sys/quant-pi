# quant-pi paperweight

800×480 dashboard showing a paper-trading portfolio managed by an LLM agent that
reads a curated Obsidian brain folder. Currently runs on Windows as a Tkinter
window matching the Pi 5 + Hosyond 5" DSI screen. Same code will run on the Pi
unchanged — DSI screen acts as primary display.

Two LLM modes:
- **Anthropic** — Claude Sonnet 4.6 via API. Best quality. Needs API key + credits.
- **Ollama** — local model on your machine (Qwen 2.5 7B, Llama 3.2 3B). Free, slower, lower quality. Same code path.

Toggle via `[agent].provider = "anthropic"` or `"ollama"` in `config.toml`.

## Setup

```cmd
cd quant-pi
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env
copy config.example.toml config.toml
notepad .env                        :: fill in keys
notepad config.toml                 :: vault path, watchlist, provider, caps
```

You always need:
- Alpaca **paper** API key + secret — https://app.alpaca.markets/paper/dashboard/overview

You only need one of these (per `[agent].provider`):
- **Anthropic** — API key from https://console.anthropic.com/ + ~$5 in credits
- **Ollama** — install locally (see below), no API key

### Ollama (Windows, optional)

If you set `[agent].provider = "ollama"`:

1. Install Ollama for Windows: https://ollama.com/download/windows
2. Open a new cmd window, pull the model:
   ```cmd
   ollama pull qwen2.5:7b
   ```
   ~4.5 GB download. Coffee-break time.
3. Verify it's running:
   ```cmd
   ollama list
   curl http://localhost:11434/api/tags
   ```
4. Run `py -m quant_pi.main` as usual. Agent will route through localhost.

To switch back to Anthropic, edit `config.toml` → `provider = "anthropic"` and re-run.

## Run

Smoke tests, in order:

```cmd
py -m quant_pi.brain.loader                  :: confirms vault read
py -m quant_pi.broker.alpaca                 :: confirms Alpaca + prints account
py -m quant_pi.agent.trader --dry-run        :: one cycle, place_order mocked
py -m quant_pi.main                          :: full run (800x480 window)
```

## Architecture

```
quant_pi/
  agent/trader.py        # Tool-use loop. Dispatches to Anthropic SDK or OpenAI-compat (Ollama).
  broker/alpaca.py       # alpaca-py wrapper
  brain/loader.py        # walks brain folder, concatenates *.md
  display/dashboard.py   # Tkinter 800x480 — equity, positions, agent action, touch buttons
  store/db.py            # SQLite: decisions, trades, equity, flags
  main.py                # APScheduler cycles + Tkinter mainloop + button callbacks
```

Cadence is dual-mode:
- **Regular market hours** (Mon-Fri 9:30am–4:00pm ET) — every
  `trading.cycle_minutes` (default **30 min**), clock-aligned.
- **Off-hours** (weekends, overnight, pre/post-market) — every
  `trading.offhours_cycle_hours` (default **6 hours**), since most names
  aren't liquid then.

Each cycle: agent inspects account/positions/quotes, applies Brain formulas,
may submit orders. Display polls independently every 2s so it stays live
between cycles.

## Dashboard buttons

| Button | What it does |
|---|---|
| PAUSE / RESUME | Flips the SQLite `PAUSED` flag — paused cycles are skipped |
| RUN NOW | Triggers a cycle immediately on a background thread |
| HISTORY | Opens a window listing the last 20 decisions |
| INFO | Shows current provider, model, watchlist, caps |

## Safety rails

The `place_order` tool validates against hard caps **before** submitting to Alpaca:

- `max_position_pct` — single position cap as % of equity
- `max_order_notional` — single-order notional cap
- `max_daily_trades` — daily order count cap
- Watchlist allowlist — agent can only trade symbols you list
- Market-closed orders are auto-rejected
- `PAUSED` flag in SQLite — set to skip all cycles

## Cost notes

**Anthropic provider:** the system prompt embeds your full brain corpus and is
marked for prompt caching, so cycles after the first should be cheap. If
`brain/loader.py` reports >50K tokens, narrow ingestion via `[brain].subfolder`
or `[brain].root_note`.

**Ollama provider:** $0/cycle. Just CPU/RAM time on your machine.

## Pi deployment

- Same code, same config, same brain folder — `bash pi/install_pi.sh` on the Pi, `deploy.bat` to push updates from the PC
- Hosyond 5" DSI screen acts as primary display — Tkinter window renders to it directly, no driver swap
- Install Ollama on the Pi: `curl -fsSL https://ollama.com/install.sh | sh`
- `ollama pull qwen2.5:7b`
- Run `main.py` as a `systemd` service on boot
- Agent / broker / brain / store unchanged

import threading
import traceback
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from quant_pi.agent.trader import run_cycle
from quant_pi.brain.loader import load_corpus
from quant_pi.broker.alpaca import Broker
from quant_pi.config import load_config, require_keys
from quant_pi.display.dashboard import Dashboard
from quant_pi.store import db


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.toml")
    require_keys(cfg)
    db.init_db(cfg.db_path)

    broker = Broker(cfg.alpaca_api_key, cfg.alpaca_secret_key, paper=cfg.alpaca_paper)
    brain = load_corpus(cfg.vault_path, cfg.vault_subfolder, cfg.brain_root_note, cfg.brain_max_depth)
    print(f"[boot] brain: {brain.file_count} files, ~{brain.approx_tokens:,} tokens")

    # Initial equity snapshot so the display shows something immediately
    try:
        acc = broker.account()
        scale = cfg.sim_capital / 100_000.0 if cfg.sim_capital else 1.0
        db.log_equity(cfg.db_path, acc.equity * scale, acc.cash * scale,
                      acc.day_pl * scale if acc.day_pl else acc.day_pl)
    except Exception as e:
        print(f"[boot] account fetch failed: {e}")

    # One-shot backfill: fetch fill prices for any historical filled trades
    # that landed before filled_avg_price capture was wired. Runs in the
    # background so it doesn't slow boot — HISTORY P/L populates as it works.
    def _bg_backfill() -> None:
        try:
            from quant_pi.agent.trader import backfill_fill_prices
            n = backfill_fill_prices(cfg, broker)
            if n:
                print(f"[boot] backfilled fill prices for {n} historical trades")
        except Exception as e:
            print(f"[boot] backfill skipped: {e}")
    threading.Thread(target=_bg_backfill, daemon=True).start()


    # Box the lock so STOP can swap in a fresh one and zombie cycles can't
    # report as "still running" once the user has explicitly STOPPED.
    cycle_lock_box = [threading.Lock()]

    def cycle_job() -> None:
        if db.is_paused(cfg.db_path):
            print("[cycle] skipped — STOP flag is set")
            return
        my_lock = cycle_lock_box[0]
        if not my_lock.acquire(blocking=False):
            print("[cycle] skipped — previous cycle still running")
            return
        try:
            print("[cycle] starting")
            result = run_cycle(cfg, broker, brain, dry_run=False)
            print(f"[cycle] done — orders={result.orders_submitted} summary={result.summary!r}")
            try:
                acc = broker.account()
                scale = cfg.sim_capital / 100_000.0 if cfg.sim_capital else 1.0
                db.log_equity(cfg.db_path, acc.equity * scale, acc.cash * scale,
                              acc.day_pl * scale if acc.day_pl else acc.day_pl)
            except Exception as e:
                print(f"[cycle] equity snapshot failed: {e}")
        except Exception:
            print("[cycle] ERROR:")
            traceback.print_exc()
        finally:
            try:
                my_lock.release()
            except RuntimeError:
                # Lock was swapped out by STOP — zombie cycle, harmless
                pass

    scheduler = BackgroundScheduler()
    # ── Two-schedule cadence ─────────────────────────────────────────────
    # Regular market hours (Mon-Fri 9:30am-4:00pm ET): every cycle_minutes,
    # clock-aligned. Active reasoning when liquidity is real.
    interval = cfg.cycle_minutes
    if interval >= 60:
        # Cron minute fields can't step past 59, so hour-scale cadences step
        # whole hours instead, from the first full hour after the 9:30 open
        # to before the 16:00 close: 120 -> 10,12,14 ET, per config.toml.
        step_h = max(1, round(interval / 60))
        market_hour = ",".join(str(h) for h in range(10, 16, step_h))
        market_minute = "0"
    elif 60 % interval == 0:
        market_hour = "9-16"
        market_minute = f"*/{interval}"
    else:
        # Fallback: 30 if user picked something non-divisible
        market_hour = "9-16"
        market_minute = "*/30"
        print(f"[boot] WARNING: cycle_minutes={interval} does not divide 60; "
              "falling back to every 30 minutes")
    scheduler.add_job(
        cycle_job, "cron",
        day_of_week="mon-fri",
        hour=market_hour,
        minute=market_minute,
        timezone="America/New_York",
        id="market_hours_cycle",
    )

    # Off-hours: weekends + nights + pre-market + after-close. Long intervals
    # since most of the universe is illiquid and the agent shouldn't churn
    # through cash on tape that won't fill cleanly anyway.
    offhours_step = max(1, cfg.offhours_cycle_hours)
    grid = list(range(0, 24, offhours_step))
    # Nights + pre/post-market, every day: grid hours outside the 9-16 session.
    night_hours = [str(h) for h in grid if not (9 <= h <= 16)]
    scheduler.add_job(
        cycle_job, "cron",
        hour=",".join(night_hours),
        minute="0",
        timezone="America/New_York",
        id="offhours_cycle",
    )
    # Grid hours inside 9-16 are still off-hours on weekends; without this job
    # a 12h cadence collapses to a single midnight fire all week long.
    weekend_hours = [str(h) for h in grid if 9 <= h <= 16]
    if weekend_hours:
        scheduler.add_job(
            cycle_job, "cron",
            day_of_week="sat,sun",
            hour=",".join(weekend_hours),
            minute="0",
            timezone="America/New_York",
            id="offhours_weekend_cycle",
        )

    print(f"[boot] scheduler: market-hours mon-fri hours {market_hour} ET minute {market_minute}"
          f"  |  off-hours at {night_hours} ET daily"
          + (f" + {weekend_hours} ET weekends" if weekend_hours else ""))
    scheduler.start()

    # Fire one cycle immediately in a background thread so the display can come up.
    threading.Thread(target=cycle_job, daemon=True).start()

    def toggle_paused() -> None:
        was_stopped = db.is_paused(cfg.db_path)
        db.set_paused(cfg.db_path, not was_stopped)
        if was_stopped:
            # Transitioning STOP → START: swap in a fresh lock so any zombie
            # cycle from before doesn't make us look "still running" anymore.
            cycle_lock_box[0] = threading.Lock()
            print("[control] START — fresh cycle lock issued")
        else:
            print("[control] STOP — new cycles blocked")

    def run_cycle_now() -> None:
        threading.Thread(target=cycle_job, daemon=True).start()

    dashboard = Dashboard(
        cfg, broker,
        paused_getter=lambda: db.is_paused(cfg.db_path),
        toggle_paused=toggle_paused,
        run_cycle_now=run_cycle_now,
        is_cycling=lambda: cycle_lock_box[0].locked(),
    )
    try:
        dashboard.run()
    finally:
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()

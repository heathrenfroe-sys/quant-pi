# FINbot — Pi 5 deployment

Drop-in setup turning a fresh Pi 5 into an autonomous Claude-driven trader
that boots straight to the touchscreen.

## Hardware

- Pi 5 8GB
- 64GB SD card (Pi OS Lite 64-bit, headless via Pi Imager)
- Hosyond 5" 800×480 IPS DSI touchscreen (plugged into Pi's DSI ribbon)
- 27W GaN USB-C PSU
- Active cooler (required for 24/7 — Pi 5 throttles under sustained load)

## Phase 1 — Flash the SD card (do this on your PC)

1. Install **Raspberry Pi Imager** from raspberrypi.com.
2. Insert SD card via the included USB reader.
3. In Imager:
   - **Device:** Raspberry Pi 5
   - **OS:** Raspberry Pi OS Lite (64-bit) — under "Raspberry Pi OS (other)"
   - **Storage:** the SD card
4. Click ⚙ settings cog and set:
   - Hostname: `finbot`
   - Enable SSH (use password auth)
   - Username: `finbot`, password: pick one
   - Wifi SSID + password (your home network)
   - Locale + timezone
5. Save → Write. Eject the SD card when done.

## Phase 2 — First boot

1. Insert SD card into Pi 5.
2. Plug DSI ribbon into the screen (or HDMI to a monitor for setup).
3. Connect ethernet OR rely on the wifi config from Imager.
4. Plug in the 27W PSU. Pi boots automatically.
5. From your PC or phone, find the Pi:
   ```
   ssh finbot@finbot.local
   ```
   (If `.local` doesn't resolve, find the IP from your router's admin page.)

## Phase 3 — Install FINbot

```bash
# On the Pi (over SSH)
git clone https://github.com/<your-username>/quant-pi.git ~/quant-pi
cd ~/quant-pi

# Create .env with your secrets
cat > .env <<'EOF'
ANTHROPIC_API_KEY=sk-ant-...
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true
EOF

# Run the one-shot installer
bash pi/install_pi.sh

# Reboot
sudo reboot
```

After reboot, the DSI screen boots straight to the FINbot dashboard. No
keyboard, no monitor, no mouse needed. The Pi is now a tablet.

## Files in this folder

| File | Purpose |
|---|---|
| `install_pi.sh` | One-shot setup: deps, venv, systemd service, autologin, kiosk |
| `finbot.service` | systemd unit that owns the dashboard process |
| `xinitrc` | Launches the Tk dashboard fullscreen with no window manager |
| `bash_profile_snippet.sh` | Hook that triggers `startx` on tty1 autologin |
| `README.md` | This file |

## Operational reference

- **Logs:**       `journalctl -u finbot -f`
- **Restart:**    `sudo systemctl restart finbot`
- **Stop:**       `sudo systemctl stop finbot`
- **Disable autostart:**  `sudo systemctl disable finbot`
- **Update code:**  `cd ~/quant-pi && git pull && sudo systemctl restart finbot`
- **Pause trading:** press the 🛑 STOP button on the dashboard, or
  `sqlite3 ~/quant-pi/quant_pi.db "UPDATE flags SET value='1' WHERE key='PAUSED'"`

## What runs where

```
PI 5 (your desk):
  • Tk dashboard on 5" DSI screen
  • SQLite log of trades + decisions
  • 15-min cycle scheduler
  • Brain folder (Markdown formulas)
  • Sends prompts to Anthropic, places orders on Alpaca

ANTHROPIC (cloud):
  • Claude Sonnet 4.6 reasoning (paid by API credits)

ALPACA (cloud):
  • Paper trading account + market data + Benzinga news

YAHOO + GOOGLE NEWS (cloud, free):
  • Multi-source news aggregation
```

The Pi never needs your PC after setup. Internet + wall power are the only
ongoing requirements.

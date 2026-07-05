#!/usr/bin/env bash
# install_pi.sh — one-shot setup for Pi 5 running Pi OS Desktop (64-bit).
#
# Run from the repo root:  bash pi/install_pi.sh
#
# After this finishes, reboot once and the dashboard auto-starts on the DSI
# screen.

set -e

USER="${USER:-$(whoami)}"
REPO="/home/$USER/quant-pi"

echo "==> Updating apt + installing system packages"
sudo apt-get update
sudo apt-get install -y \
    python3-venv python3-full python3-tk python3-dev \
    xserver-xorg xinit x11-xserver-utils \
    unclutter git \
    fonts-noto-color-emoji

echo "==> Creating Python venv at $REPO/.venv"
cd "$REPO"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "==> Installing systemd service (substituting user '$USER')"
sed -e "s|/home/pi|/home/$USER|g" -e "s|^User=pi$|User=$USER|" -e "s|^Group=pi$|Group=$USER|" \
    pi/finbot.service | sudo tee /etc/systemd/system/finbot.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable finbot.service
echo "    (will start automatically after reboot)"

echo "==> Installing kiosk xinitrc"
cp pi/xinitrc /home/$USER/.xinitrc
chmod +x /home/$USER/.xinitrc

echo "==> Wiring tty1-autologin → startx in ~/.bash_profile"
if ! grep -q "exec startx" /home/$USER/.bash_profile 2>/dev/null; then
    cat pi/bash_profile_snippet.sh >> /home/$USER/.bash_profile
    echo "    appended startx hook"
else
    echo "    already present, skipping"
fi

echo "==> Disabling unused services to free RAM"
for svc in bluetooth cups triggerhappy avahi-daemon ModemManager; do
    sudo systemctl disable --now "$svc" 2>/dev/null || true
done

echo
echo "============================================================"
echo "  All set. Reboot now:   sudo reboot"
echo
echo "  After reboot:"
echo "    - DSI screen shows FINbot dashboard automatically"
echo "    - SSH still works for remote tweaks"
echo "    - Logs:    journalctl -u finbot -f"
echo "    - Update:  cd ~/quant-pi && bash pi/install_pi.sh"
echo "============================================================"

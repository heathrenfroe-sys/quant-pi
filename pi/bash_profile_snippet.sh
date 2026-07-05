# Append this to ~/.bash_profile so the dashboard auto-starts on
# tty1 login. Combined with raspi-config "Console Autologin", this means
# powering the Pi on goes:  Boot â†’ autologin â†’ startx â†’ dashboard.

if [[ -z "$DISPLAY" && "$XDG_VTNR" == "1" ]]; then
    exec startx
fi

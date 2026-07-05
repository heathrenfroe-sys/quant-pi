import tkinter as tk
from datetime import datetime, timedelta
from typing import Callable, Optional

from quant_pi.broker.alpaca import Broker
from quant_pi.config import Config
from quant_pi.store import db

# Color palette — Oklahoma State University brand
# Orange: PANTONE 021, #FE5C00
# Cool Gray 2 (light): #DDDDDD
# Cool Gray 10 (dark): #757575
BG = "#000000"          # black background
PANEL = "#0a0a0a"        # near-black panels for slight contrast
FG = "#FFFFFF"           # white primary text
DIM = "#757575"          # secondary text, labels

# ── FINbot ocean palette — light blue everything ────────────────
ACCENT = "#1FB8FF"       # ocean blue — every accent, status, highlight
EDGE = "#0a2840"         # deep navy panel borders
TRIM = "#1FB8FF"         # blue trim strips
RED = ACCENT             # alias kept so existing code paths work
BTN_BG = "#1a1a1a"       # button rest state
BTN_BG_HOVER = "#1FB8FF" # button hover/active
BTN_FG = "#FFFFFF"       # button text white
BTN_FG_HOVER = "#000000" # text on blue hover = black for contrast
YELLOW = "#FFD60A"       # market session indicators (sun / moon icons)
LIGHT_GREEN = "#5BE38B"  # agent fish — distinguishes "agent acting" from UI accent
PINK = "#FF6FB3"         # brain / octopus — neuro / cephalopod accents
CRIMSON = "#E63946"      # mission / clock — true red, not the legacy RED alias
UP_GREEN = "#4ADE80"     # gains — bright/light green for P/L pops
DOWN_RED = "#EF4444"     # losses — down indicators, negative P/L
DOLPHIN_BLUE = "#0EA5E0" # dolphin emoji — slightly deeper than ACCENT
SHARK_BLUE = "#0470A0"   # shark emoji — darker still, the apex predator blue
DARK_GREEN = "#15803D"   # cash — deep forest green, clearly darker than UP_GREEN
GOLD = "#E6B800"         # buying power — gold for "leverage available"
PEACH = "#F8D470"        # market session — soft yellow with mild warmth
LIGHT_GRAY = "#DDDDDD"   # OSU Cool Gray 2 — available for accents if needed

# Edge padding (the "micro buffer")
PAD = 12

# Fonts — Verdana family.
FONT_TITLE = "Verdana"
FONT_BODY = "Verdana"
FONT_MONO = "Consolas"          # tabular data (positions table)
FONT_EMOJI = "Segoe UI Emoji"   # forces Windows color-emoji rendering where possible


def title_font(size: int) -> tuple:
    """Title font with bold weight (since Verdana Pro Black isn't installed)."""
    return (FONT_TITLE, size, "bold")


def emoji_font(size: int, bold: bool = False) -> tuple:
    """Use Segoe UI Emoji explicitly so Windows renders the colored glyph instead
    of the monochrome fallback that comes from Verdana."""
    if bold:
        return (FONT_EMOJI, size, "bold")
    return (FONT_EMOJI, size)


def mono_emoji_font(size: int, bold: bool = False) -> tuple:
    """Segoe UI Symbol renders emojis in MONOCHROME, which means the `fg`
    color actually applies. Use this when you want a tinted emoji (e.g.
    a light-green fish or a pink octopus) instead of the OS default colors."""
    fam = "Segoe UI Symbol"
    return (fam, size, "bold") if bold else (fam, size)


def _bind_scroll(canvas, *widgets) -> None:
    """Bind scroll to a canvas: Windows MouseWheel, Linux Button-4/5,
    Pi touchscreen drag."""
    def _wheel(e):
        canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        return "break"
    def _up(e):
        canvas.yview_scroll(-3, "units")
        return "break"
    def _down(e):
        canvas.yview_scroll(3, "units")
        return "break"
    _drag: dict = {"y": 0}
    def _press(e):
        _drag["y"] = e.y_root
    def _motion(e):
        dy = _drag["y"] - e.y_root
        if abs(dy) > 6:
            canvas.yview_scroll(int(dy / 12), "units")
            _drag["y"] = e.y_root
    for w in (canvas, *widgets):
        w.bind("<MouseWheel>", _wheel)
        w.bind("<Button-4>",   _up)
        w.bind("<Button-5>",   _down)
        w.bind("<ButtonPress-1>", _press)
        w.bind("<B1-Motion>",     _motion)


class Dashboard:
    """800x480 dashboard sized for Hosyond 5\" DSI touchscreen.

    Sections (top to bottom):
      [Header strip]   equity (big), day P/L, market status, time
      [Body]           positions table (left) + agent action panel (right)
      [Button row]     Pause / Run Now / History / Provider
    """

    def __init__(
        self,
        cfg: Config,
        broker: Broker,
        paused_getter: Callable[[], bool],
        toggle_paused: Callable[[], None],
        run_cycle_now: Callable[[], None],
        is_cycling: Callable[[], bool] = lambda: False,
    ) -> None:
        self.cfg = cfg
        self.broker = broker
        self.paused_getter = paused_getter
        self.toggle_paused = toggle_paused
        self.run_cycle_now = run_cycle_now
        self.is_cycling = is_cycling
        self._spin_idx = 0
        self._sleep_idx = 0
        self._last_interaction = datetime.now()
        self._sleeping = False


        self._sleep_idx = 0

        # FINbot ocean theme — always-on, no palette flipping.
        self._fish_mode = True

        W = cfg.display_width
        H = cfg.display_height

        self.root = tk.Tk()
        self.root.title("quant-pi")
        self.root.geometry(f"{W}x{H}")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        if cfg.display_always_on_top:
            self.root.attributes("-topmost", True)

        # Track any user interaction for sleep timeout
        for ev in ("<Button>", "<Key>", "<Motion>"):
            self.root.bind_all(ev, self._on_interaction)

        # ── Orange trim strips (top + bottom edge of the screen) ──
        trim_h = 4
        tk.Frame(self.root, bg=TRIM).place(x=0, y=0, width=W, height=trim_h)
        tk.Frame(self.root, bg=TRIM).place(x=0, y=H - trim_h, width=W, height=trim_h)
        # Vertical orange rails on the left and right edges
        tk.Frame(self.root, bg=TRIM).place(x=0, y=0, width=trim_h, height=H)
        tk.Frame(self.root, bg=TRIM).place(x=W - trim_h, y=0, width=trim_h, height=H)

        # Layout regions (with PAD micro-buffer, accounting for trim)
        header_h = 138
        button_h = 56
        inner_pad = PAD - trim_h  # so content sits PAD from screen edge
        body_y = trim_h + inner_pad + header_h + inner_pad
        body_h = H - body_y - button_h - inner_pad - trim_h - inner_pad
        button_y = H - trim_h - inner_pad - button_h

        # ── Header panel ──────────────────────────────────────────
        header = tk.Frame(self.root, bg=PANEL, highlightthickness=1, highlightbackground=EDGE)
        header.place(x=trim_h + inner_pad, y=trim_h + inner_pad,
                     width=W - 2 * (trim_h + inner_pad), height=header_h)

        # Title strip — sharks bookend the FINbot title.
        bookend = "🦈"
        tagline = "   ·   Algorithmic Agentic Trader  ·  Just keep trading, just keep trading"
        title_strip = tk.Frame(header, bg=PANEL)
        title_strip.place(x=14, y=4, width=W - 2 * (trim_h + inner_pad) - 28, height=20)
        tk.Label(title_strip, text=bookend, bg=PANEL, fg=SHARK_BLUE,
                 font=mono_emoji_font(13, bold=True)).pack(side="left", padx=(0, 6))
        tk.Label(title_strip, text="FINbot 3K", bg=PANEL, fg=ACCENT,
                 font=title_font(12)).pack(side="left")
        tk.Label(title_strip, text=bookend, bg=PANEL, fg=SHARK_BLUE,
                 font=mono_emoji_font(13, bold=True)).pack(side="left", padx=(6, 0))
        tk.Label(title_strip, text=tagline, bg=PANEL, fg=DIM,
                 font=(FONT_BODY, 10)).pack(side="left")
        tk.Label(title_strip, text="🦈", bg=PANEL, fg=SHARK_BLUE,
                 font=mono_emoji_font(11, bold=True)).pack(side="left", padx=(4, 0))
        # Spinner shows when a cycle is actively running; hidden when idle
        self.spinner_var = tk.StringVar(value="")
        self.spinner_label = tk.Label(title_strip, textvariable=self.spinner_var,
                                       bg=PANEL, fg=ACCENT,
                                       font=emoji_font(13, bold=True))
        self.spinner_label.pack(side="left", padx=(8, 0))

        # Equity: split into colored arrow + white amount so the arrow can
        # turn green/red without recoloring the equity number.
        equity_row = tk.Frame(header, bg=PANEL)
        equity_row.place(x=14, y=30)
        self.equity_icon_var = tk.StringVar(value="▲")
        self.equity_icon_label = tk.Label(equity_row, textvariable=self.equity_icon_var,
                                          bg=PANEL, fg=UP_GREEN,
                                          font=mono_emoji_font(22, bold=True))
        self.equity_icon_label.pack(side="left", padx=(0, 6))
        self.equity_var = tk.StringVar(value="$ —")
        tk.Label(equity_row, textvariable=self.equity_var, bg=PANEL, fg=FG,
                 font=title_font(28), anchor="w").pack(side="left")

        # Cash (dark green) + Buying Power (gold) breakdown
        breakdown = tk.Frame(header, bg=PANEL)
        breakdown.place(x=16, y=78)
        self.cash_var = tk.StringVar(value="💵 Cash $—")
        tk.Label(breakdown, textvariable=self.cash_var, bg=PANEL, fg=DARK_GREEN,
                 font=(FONT_BODY, 11, "bold"), anchor="w").pack(side="left")
        tk.Label(breakdown, text="  ·  ", bg=PANEL, fg=DIM,
                 font=(FONT_BODY, 11)).pack(side="left")
        self.bp_var = tk.StringVar(value="🐴 Buying Power $—")
        tk.Label(breakdown, textvariable=self.bp_var, bg=PANEL, fg=GOLD,
                 font=(FONT_BODY, 11, "bold"), anchor="w").pack(side="left")

        # Day P/L — Canvas-rendered with a stroke effect (orange fill, white outline
        # for gains, black outline for losses). All tk Labels can't draw text strokes;
        # Canvas can fake one by drawing the text 8 times in the outline color around
        # the center, then once in fill on top.
        self._pl_canvas = tk.Canvas(header, bg=PANEL, bd=0, highlightthickness=0,
                                     width=400, height=24)
        self._pl_canvas.place(x=16, y=108)
        self._pl_text = "—"
        self._pl_outline = PANEL  # neutral until first refresh

        self.market_var = tk.StringVar(value="● Market Closed")
        self.market_label = tk.Label(header, textvariable=self.market_var, bg=PANEL, fg=DIM,
                                      font=(FONT_BODY, 12, "bold"), anchor="e")
        self.market_label.place(relx=1.0, x=-14, y=30, anchor="ne")

        self.time_var = tk.StringVar(value="")
        tk.Label(header, textvariable=self.time_var, bg=PANEL, fg=DIM,
                 font=(FONT_MONO, 14), anchor="e").place(relx=1.0, x=-14, y=54, anchor="ne")

        # PAUSED indicator (orange, only visible when paused)
        self.paused_var = tk.StringVar(value="")
        tk.Label(header, textvariable=self.paused_var, bg=PANEL, fg=ACCENT,
                 font=title_font(11), anchor="e").place(relx=1.0, x=-14, y=80, anchor="ne")

        # Provider:model — bottom-right, baseline-aligned with Day P/L on the left
        self.provider_var = tk.StringVar(value="")
        tk.Label(header, textvariable=self.provider_var, bg=PANEL, fg=DIM,
                 font=(FONT_BODY, 12), anchor="ne").place(relx=1.0, x=-14, y=106, anchor="ne")

        # ── Body: positions (left) + action (right) ───────────────
        body_x = trim_h + inner_pad
        body_w = W - 2 * (trim_h + inner_pad)
        positions_w = int(body_w * 0.55)
        action_w = body_w - positions_w - PAD

        # ── EQUITY CHART panel (replaces old POSITIONS panel on home page) ──
        positions = tk.Frame(self.root, bg=PANEL, highlightthickness=1, highlightbackground=EDGE)
        positions.place(x=body_x, y=body_y, width=positions_w, height=body_h)

        tk.Label(positions, text="EQUITY", bg=PANEL, fg=FG,
                 font=title_font(11), anchor="w").place(x=12, y=6)

        self._chart_range = "DAILY"

        # Tab buttons: DAILY / TOTAL
        tab_frame = tk.Frame(positions, bg=PANEL)
        tab_frame.place(x=positions_w - 140, y=6)
        self._chart_tab_btns = {}
        for i, tab_name in enumerate(("DAILY", "TOTAL")):
            btn = tk.Label(tab_frame, text=tab_name, bg=PANEL, fg=DIM,
                           font=(FONT_BODY, 9, "bold"), cursor="hand2",
                           padx=6, pady=1)
            btn.pack(side="left", padx=2)
            btn.bind("<Button-1>", lambda e, t=tab_name: self._switch_chart_tab(t))
            self._chart_tab_btns[tab_name] = btn
        self._chart_tab_btns["DAILY"].configure(fg=ACCENT)

        # Live equity curve canvas — full height, no range buttons
        self._equity_canvas = tk.Canvas(positions, bg=PANEL, bd=0, highlightthickness=0)
        self._equity_canvas.place(x=12, y=26, width=positions_w - 24, height=body_h - 62)
        self._chart_state: Optional[dict] = None
        self._equity_canvas.bind("<Motion>", self._on_chart_hover)
        self._chart_mouse_event = None
        def _on_chart_leave(e):
            self._chart_mouse_event = None
            self._equity_canvas.delete("crosshair")
        self._equity_canvas.bind("<Leave>", _on_chart_leave)
        # Caption below the chart — split into colored P/L (green/red) +
        # neutral context. Both rendered on a Frame placed at the same y.
        self._equity_caption_var = tk.StringVar(value="")
        self._equity_context_var = tk.StringVar(value="")
        cap_frame = tk.Frame(positions, bg=PANEL)
        cap_frame.place(x=12, y=body_h - 32)
        self._equity_caption_label = tk.Label(
            cap_frame, textvariable=self._equity_caption_var, bg=PANEL,
            fg=UP_GREEN, font=(FONT_BODY, 10, "bold"), anchor="w")
        self._equity_caption_label.pack(side="left")
        tk.Label(cap_frame, textvariable=self._equity_context_var, bg=PANEL,
                 fg=FG, font=(FONT_BODY, 10), anchor="w").pack(side="left", padx=(6, 0))

        action = tk.Frame(self.root, bg=PANEL, highlightthickness=1,
                          highlightbackground=EDGE, highlightcolor=EDGE)
        action.place(x=body_x + positions_w + PAD, y=body_y, width=action_w, height=body_h)

        action_title = tk.Frame(action, bg=PANEL)
        action_title.place(x=12, y=8)
        tk.Label(action_title, text="🐟", bg=PANEL, fg=LIGHT_GREEN,
                 font=mono_emoji_font(11, bold=True)).pack(side="left", padx=(0, 5))
        tk.Label(action_title, text="LAST AGENT ACTION", bg=PANEL, fg=ACCENT,
                 font=title_font(11)).pack(side="left")
        tk.Label(action_title, text="🐟", bg=PANEL, fg=LIGHT_GREEN,
                 font=mono_emoji_font(11, bold=True)).pack(side="left", padx=(5, 0))

        self.action_ts_var = tk.StringVar(value="—")
        tk.Label(action, textvariable=self.action_ts_var, bg=PANEL, fg=ACCENT,
                 font=(FONT_BODY, 11, "bold"), anchor="w").place(x=12, y=30)

        # Scrollable read-only Text widget so long agent rationales never get visually cut off.
        self.action_text = tk.Text(action, bg=PANEL, fg=FG, bd=0,
                                    highlightthickness=0, takefocus=0,
                                    font=(FONT_BODY, 11), wrap="word", padx=0, pady=0,
                                    cursor="arrow", insertontime=0)
        self.action_text.place(x=12, y=52, width=action_w - 24, height=body_h - 62)
        self.action_text.insert("end", "(no decisions yet)")
        self.action_text.configure(state="disabled")

        # Platform-aware scroll for the action panel (Windows + Linux + Pi touch)
        _bind_scroll(self.action_text)

        # ── Button row ────────────────────────────────────────────
        btn_row_w = W - 2 * (trim_h + inner_pad)
        # 5 buttons with 4 gaps of 6px between them
        btn_w = (btn_row_w - 24) // 5
        btn_x0 = trim_h + inner_pad

        def make_btn(emoji: str, label: str, emoji_color: str, command, x: int,
                     label_var: Optional[tk.StringVar] = None) -> tk.Frame:
            """Frame-based button so the emoji can be tinted separately while
            the text label stays white. Click + hover handled on the frame
            and both child labels."""
            frame = tk.Frame(self.root, bg=BTN_BG, cursor="hand2")
            frame.place(x=x, y=button_y, width=btn_w, height=button_h)
            inner = tk.Frame(frame, bg=BTN_BG)
            inner.place(relx=0.5, rely=0.5, anchor="center")
            # Segoe UI Symbol (mono) lacks newer emojis like 🎣 and renders
            # them as fallback geometry. When the caller passes BTN_FG (white)
            # — i.e. "let the emoji be itself" — use Segoe UI Emoji so the
            # full-color glyph renders correctly.
            ef = (emoji_font(11) if emoji_color == BTN_FG
                  else mono_emoji_font(11, bold=True))
            emoji_lbl = tk.Label(inner, text=emoji, bg=BTN_BG, fg=emoji_color,
                                  font=ef)
            emoji_lbl.pack(side="left", padx=(0, 6))
            if label_var is not None:
                text_lbl = tk.Label(inner, textvariable=label_var, bg=BTN_BG,
                                     fg=BTN_FG, font=title_font(11))
            else:
                text_lbl = tk.Label(inner, text=label, bg=BTN_BG, fg=BTN_FG,
                                     font=title_font(11))
            text_lbl.pack(side="left")

            def _click(_e=None):
                command()
            def _enter(_e=None):
                for w in (frame, inner, emoji_lbl, text_lbl):
                    w.configure(bg=BTN_BG_HOVER)
                text_lbl.configure(fg=BTN_FG_HOVER)
                emoji_lbl.configure(fg=BTN_FG_HOVER)
            def _leave(_e=None):
                for w in (frame, inner, emoji_lbl, text_lbl):
                    w.configure(bg=BTN_BG)
                text_lbl.configure(fg=BTN_FG)
                emoji_lbl.configure(fg=emoji_color)
            for w in (frame, inner, emoji_lbl, text_lbl):
                w.bind("<Button-1>", _click)
                w.bind("<Enter>", _enter)
                w.bind("<Leave>", _leave)
            return frame

        self.pause_btn_var = tk.StringVar(value="STOP")
        make_btn("🛑", "STOP", DOWN_RED, self._on_pause, btn_x0,
                 label_var=self.pause_btn_var)
        make_btn("🎣", "RUN NOW", BTN_FG, self._on_run_now,
                 btn_x0 + (btn_w + 6) * 1)
        make_btn("⚓", "HOLDINGS", DIM, self._on_holdings,
                 btn_x0 + (btn_w + 6) * 2)
        make_btn("🐬", "HISTORY", LIGHT_GRAY, self._on_history,
                 btn_x0 + (btn_w + 6) * 3)
        self.refresh_btn_var = tk.StringVar(value="REFRESH")
        make_btn("🔄", "REFRESH", ACCENT, self._on_refresh,
                 btn_x0 + (btn_w + 6) * 4,
                 label_var=self.refresh_btn_var)

        self._cached_market_open: Optional[bool] = None
        self._market_check_counter = 0
        self._sync_counter = 0  # for periodic order-status sync

    # ── Button handlers ───────────────────────────────────────────

    def _on_pause(self) -> None:
        self.toggle_paused()

    def _on_run_now(self) -> None:
        self.run_cycle_now()
        self._set_action_text("Cycle requested — agent thinking…")
        self.action_ts_var.set(datetime.now().strftime("%H:%M:%S"))

    def _set_action_text(self, text: str) -> None:
        self.action_text.configure(state="normal")
        self.action_text.delete("1.0", "end")
        self.action_text.insert("end", text)
        self.action_text.configure(state="disabled")
        self.action_text.yview_moveto(0)

    def _make_popup(self, title: str) -> tk.Toplevel:
        """Create a popup that matches the main dashboard's size + theme,
        including the OSU orange trim strips on all four edges."""
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry(f"{self.cfg.display_width}x{self.cfg.display_height}")
        win.configure(bg=BG)
        win.resizable(False, False)
        if self.cfg.display_always_on_top:
            win.attributes("-topmost", True)

        # Orange trim strips on all four edges — matches the main dashboard
        trim_h = 4
        W = self.cfg.display_width
        H = self.cfg.display_height
        tk.Frame(win, bg=TRIM).place(x=0, y=0, width=W, height=trim_h)
        tk.Frame(win, bg=TRIM).place(x=0, y=H - trim_h, width=W, height=trim_h)
        tk.Frame(win, bg=TRIM).place(x=0, y=0, width=trim_h, height=H)
        tk.Frame(win, bg=TRIM).place(x=W - trim_h, y=0, width=trim_h, height=H)
        return win

    def _add_close_button(self, win: tk.Toplevel) -> None:
        tk.Button(win, text="CLOSE", command=win.destroy,
                  bg=BTN_BG, fg=BTN_FG, activebackground=BTN_BG_HOVER, activeforeground=BTN_FG_HOVER,
                  bd=0, relief="flat", font=title_font(13), cursor="hand2").place(
            x=PAD, rely=1.0, y=-PAD, anchor="sw",
            width=self.cfg.display_width - PAD * 2, height=46,
        )

    def _on_holdings(self) -> None:
        """Holdings popup — scrollable: WATCHING strip + positions table +
        analysis-methods bar chart at the bottom."""
        win = self._make_popup("Holdings")
        W = self.cfg.display_width
        H = self.cfg.display_height
        section_w = W - PAD * 2

        # ── HEADER (fixed) ─────────────────────────────────────────
        pos_header_h = 56
        pos_y = PAD
        pos_header = tk.Frame(win, bg=PANEL, highlightthickness=1,
                               highlightbackground=EDGE, highlightcolor=EDGE)
        pos_header.place(x=PAD, y=pos_y, width=section_w, height=pos_header_h)
        pos_title = tk.Frame(pos_header, bg=PANEL)
        pos_title.place(x=14, y=14)
        tk.Label(pos_title, text="⚓", bg=PANEL, fg=DIM,
                 font=mono_emoji_font(15, bold=True)).pack(side="left", padx=(0, 8))
        tk.Label(pos_title, text="HOLDINGS", bg=PANEL, fg=ACCENT,
                 font=title_font(15)).pack(side="left")

        # Fetch positions + account
        try:
            positions_list = self.broker.positions()

            def _gl_pct(p):
                cost = p.cost_basis or (p.qty * p.avg_entry_price if p.avg_entry_price else 0.0)
                return (p.unrealized_pl / cost * 100) if cost else 0.0

            positions_list.sort(key=_gl_pct, reverse=True)
        except Exception:
            positions_list = []
        try:
            acc = self.broker.account()
            total_equity = self._sim_equity(acc.equity) if acc.equity else 0.0
        except Exception:
            total_equity = 0.0

        sim_scale = self.cfg.sim_capital / 100_000.0 if self.cfg.sim_capital else 1.0
        if positions_list:
            total_mv = sum(p.market_value for p in positions_list) * sim_scale
            total_pl = sum(p.unrealized_pl for p in positions_list) * sim_scale
            total_day = sum(p.change_today_pl for p in positions_list) * sim_scale
            pl_sign = "+" if total_pl >= 0 else ""
            day_sign = "+" if total_day >= 0 else ""
            tk.Label(pos_header,
                     text=f"{len(positions_list)} open  ·  ${total_mv:,.0f}  ·  "
                          f"day {day_sign}${total_day:,.2f}  ·  total {pl_sign}${total_pl:,.2f}",
                     bg=PANEL, fg=DIM, font=(FONT_BODY, 11), anchor="e").place(
                relx=1.0, x=-14, y=18, anchor="ne"
            )
        else:
            tk.Label(pos_header, text="0 open", bg=PANEL, fg=DIM,
                     font=(FONT_BODY, 11), anchor="e").place(
                relx=1.0, x=-14, y=18, anchor="ne"
            )

        # ── SCROLLABLE BODY ────────────────────────────────────────
        body_y = pos_y + pos_header_h + 4
        bottom_buffer = 46 + 12 + 50
        body_h = H - body_y - bottom_buffer

        body_frame = tk.Frame(win, bg=PANEL, highlightthickness=1,
                              highlightbackground=EDGE, highlightcolor=EDGE)
        body_frame.place(x=PAD, y=body_y, width=section_w, height=body_h)

        scrollbar = tk.Scrollbar(body_frame, orient="vertical",
                                 bg=EDGE, troughcolor=PANEL,
                                 activebackground=ACCENT, width=14,
                                 highlightthickness=0, bd=0)
        scrollbar.pack(side="right", fill="y")

        canvas = tk.Canvas(body_frame, bg=PANEL, bd=0, highlightthickness=0,
                           takefocus=0, yscrollcommand=scrollbar.set)
        inner = tk.Frame(canvas, bg=PANEL)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw", width=section_w - 18)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=canvas.yview)

        _bind_scroll(canvas, body_frame, win)

        # ── Section: 👀 WATCHING ───────────────────────────────────
        try:
            watching = db.tracked_symbols(self.cfg.db_path, limit=20)
        except Exception:
            watching = []
        if watching:
            wl_section = tk.Frame(inner, bg=PANEL)
            wl_section.pack(fill="x", padx=8, pady=(8, 0))
            wl_title = tk.Frame(wl_section, bg=PANEL)
            wl_title.pack(anchor="w", padx=6, pady=(4, 0))
            tk.Label(wl_title, text="🦀", bg=PANEL, fg=DOWN_RED,
                     font=mono_emoji_font(11, bold=True)).pack(side="left", padx=(0, 6))
            tk.Label(wl_title, text="WATCHING", bg=PANEL, fg=ACCENT,
                     font=title_font(11)).pack(side="left")

            # Tabular layout: TYPE | TICKER, color-coded by trade type
            TYPE_COLORS = {
                "long":  UP_GREEN,
                "short": DOWN_RED,
                "call":  UP_GREEN,
                "put":   DOWN_RED,
            }
            TYPE_LABELS = {
                "long":  "LONG ",
                "short": "SHORT",
                "call":  "CALL ",
                "put":   "PUT  ",
            }
            wl_grid = tk.Frame(wl_section, bg=PANEL)
            wl_grid.pack(anchor="w", padx=6, pady=(2, 6))
            # Render as up-to-2 columns × N rows for compact horizontal use
            per_col = 7
            for i, w in enumerate(watching[:14]):
                tt = (w["trade_type"] or "long").lower()
                col = i // per_col
                row = i % per_col
                cell = tk.Frame(wl_grid, bg=PANEL)
                cell.grid(row=row, column=col, sticky="w", padx=(0, 14))
                tk.Label(cell, text=TYPE_LABELS.get(tt, "LONG "),
                         bg=PANEL, fg=TYPE_COLORS.get(tt, UP_GREEN),
                         font=(FONT_MONO, 10, "bold")).pack(side="left")
                tk.Label(cell, text=f" {w['symbol']}",
                         bg=PANEL, fg=FG, font=(FONT_MONO, 11)).pack(side="left")

        # ── Section: POSITIONS TABLE ───────────────────────────────
        pos_section = tk.Frame(inner, bg=PANEL)
        pos_section.pack(fill="x", padx=8, pady=(8, 0))

        header_text = (
            f"{'TYPE':<6}{'TICKER':<9}{'QTY':>6} {'ENTRY':>8} {'PRICE':>8} "
            f"{'MKT VAL':>10} {'DAY $':>9} {'DAY %':>8} "
            f"{'G/L $':>10} {'G/L %':>8} {'%PORT':>7}"
        )
        tk.Label(pos_section, text=header_text,
                 bg=PANEL, fg=DIM, font=(FONT_MONO, 10), anchor="w").pack(anchor="w", padx=6)

        # OCC option-symbol detector: 6+ underlying chars + YYMMDD + C/P + 8 strike digits
        import re as _re
        _OCC_RE = _re.compile(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$")

        def _classify(p):
            """Returns (type_label, color, display_ticker, display_qty)."""
            sym = p.symbol or ""
            if _OCC_RE.match(sym):
                # Option contract — extract underlying + C/P + readable expiry
                m = _re.match(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$", sym)
                if m:
                    under, yymmdd, cp, strike8 = m.groups()
                    strike = int(strike8) / 1000
                    short = f"{under} {yymmdd[2:4]}/{yymmdd[4:6]} {strike:g}{cp}"
                    if cp == "C":
                        return ("CALL ", UP_GREEN, short[:14], p.qty)
                    else:
                        return ("PUT  ", DOWN_RED, short[:14], p.qty)
            # Equity
            if p.side == "short":
                return ("SHORT", DOWN_RED, p.symbol[:9], -abs(p.qty))
            return ("LONG ", UP_GREEN, p.symbol[:9], p.qty)

        # One Text widget sized exactly to row count → no inner scroll, the
        # outer canvas handles all scrolling.
        n_rows = max(1, len(positions_list))
        pos_text = tk.Text(pos_section, bg=PANEL, fg=FG, bd=0, highlightthickness=0,
                           takefocus=0, height=n_rows,
                           font=(FONT_MONO, 10), wrap="none", padx=6, pady=2,
                           cursor="arrow", insertontime=0)
        pos_text.pack(fill="x", padx=0, pady=(2, 0))
        # Per-type tags so the TYPE column color-codes at a glance
        pos_text.tag_configure("type_long",  foreground=UP_GREEN)
        pos_text.tag_configure("type_short", foreground=DOWN_RED)
        pos_text.tag_configure("type_call",  foreground=UP_GREEN)
        pos_text.tag_configure("type_put",   foreground=DOWN_RED)
        pos_text.tag_configure("short_row",  foreground=DOWN_RED)
        if not positions_list:
            pos_text.insert("end", "(no open positions)")
        else:
            for p in positions_list:
                type_label, type_color, ticker, signed_qty = _classify(p)
                entry = p.avg_entry_price
                price = p.current_price or ((p.market_value / p.qty) if p.qty else 0.0)
                gl = p.unrealized_pl * sim_scale
                mv = p.market_value * sim_scale
                day_pl_pos = p.change_today_pl * sim_scale
                cost = p.cost_basis or (p.qty * entry if entry else 0.0)
                gl_pct = (p.unrealized_pl / cost * 100) if cost else 0.0
                pct_port = (mv / total_equity * 100) if total_equity else 0.0
                type_tag = "type_" + type_label.strip().lower()
                pos_text.insert("end", f"{type_label:<6}", type_tag)
                rest = (
                    f"{ticker:<9}{signed_qty:>6.0f} {entry:>8.2f} {price:>8.2f} "
                    f"{mv:>10,.0f} {day_pl_pos:>+9,.2f} "
                    f"{p.change_today_pct:>+7.2f}% "
                    f"{gl:>+10,.2f} {gl_pct:>+7.2f}% {pct_port:>6.1f}%\n"
                )
                # Body of short rows still tinted red for full-row legibility
                if type_label.strip() == "SHORT":
                    pos_text.insert("end", rest, "short_row")
                else:
                    pos_text.insert("end", rest)
        pos_text.configure(state="disabled")
        _bind_scroll(canvas, pos_text)

        # ── Section: 📊 ANALYSIS METHODS — formula frequency bars ───
        try:
            freq = db.formula_frequency(self.cfg.db_path)
        except Exception:
            freq = []

        analytics = tk.Frame(inner, bg=PANEL)
        analytics.pack(fill="x", padx=8, pady=(14, 8))

        head = tk.Frame(analytics, bg=PANEL)
        head.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(head, text="🦞", bg=PANEL, fg=DOWN_RED,
                 font=mono_emoji_font(12, bold=True)).pack(side="left", padx=(0, 6))
        tk.Label(head, text="ANALYSIS METHODS", bg=PANEL, fg=ACCENT,
                 font=title_font(12)).pack(side="left")
        total_cites = sum(c for _, c in freq)
        tk.Label(head, text=f"{total_cites} citations  ·  {len(freq)} methods",
                 bg=PANEL, fg=DIM, font=(FONT_BODY, 10), anchor="e").pack(side="right")

        if not freq:
            tk.Label(analytics, text="(no decisions logged yet)", bg=PANEL, fg=DIM,
                     font=(FONT_BODY, 10)).pack(padx=10, pady=(0, 10))
        else:
            max_n = max(c for _, c in freq) or 1
            bar_max_w = section_w - 8 - 100 - 60 - 50  # left pad + label col + count col + right pad
            for name, count in freq:
                row = tk.Frame(analytics, bg=PANEL)
                row.pack(fill="x", padx=10, pady=2)
                tk.Label(row, text=name, bg=PANEL, fg=FG,
                         font=(FONT_BODY, 10, "bold"), width=14, anchor="w"
                         ).pack(side="left")
                bar_canvas = tk.Canvas(row, bg=PANEL, bd=0, highlightthickness=0,
                                        height=14, width=bar_max_w)
                bar_canvas.pack(side="left", padx=(0, 8))
                bar_w = max(2, int(bar_max_w * count / max_n))
                bar_canvas.create_rectangle(0, 2, bar_w, 12, fill=ACCENT, outline="")
                tk.Label(row, text=f"{count:>3}", bg=PANEL, fg=ACCENT,
                         font=(FONT_MONO, 11, "bold"), width=4, anchor="e"
                         ).pack(side="left")
            tk.Label(analytics, text="across all logged decisions  ·  scroll up for positions",
                     bg=PANEL, fg=DIM, font=(FONT_BODY, 9), anchor="w"
                     ).pack(anchor="w", padx=10, pady=(6, 10))

        # 20px tail spacer so the analytics card never kisses the panel border
        tk.Frame(inner, bg=PANEL, height=20).pack(fill="x")

        self._add_close_button(win)

    def _on_history(self) -> None:
        """History popup — full-screen scrollable cards of recent decisions."""
        win = self._make_popup("History")
        W = self.cfg.display_width
        H = self.cfg.display_height
        section_w = W - PAD * 2

        # ── HEADER ────────────────────────────────────────────────
        dec_header_h = 56
        dec_y = PAD
        dec_header = tk.Frame(win, bg=PANEL, highlightthickness=1,
                               highlightbackground=EDGE, highlightcolor=EDGE)
        dec_header.place(x=PAD, y=dec_y, width=section_w, height=dec_header_h)
        # Dolphin in slightly-deeper blue, title text in standard accent.
        dec_title = tk.Frame(dec_header, bg=PANEL)
        dec_title.place(x=14, y=14)
        tk.Label(dec_title, text="🐬", bg=PANEL, fg=LIGHT_GRAY,
                 font=mono_emoji_font(15, bold=True)).pack(side="left", padx=(0, 8))
        tk.Label(dec_title, text="AGENT DECISIONS", bg=PANEL, fg=ACCENT,
                 font=title_font(15)).pack(side="left")
        rows = db.recent_decisions(self.cfg.db_path, limit=20)
        tk.Label(dec_header, text=f"{len(rows)} entries", bg=PANEL, fg=DIM,
                 font=(FONT_BODY, 11), anchor="e").place(relx=1.0, x=-14, y=18, anchor="ne")

        # ── FULL-HEIGHT SCROLLABLE CARDS ──────────────────────────
        list_y = dec_y + dec_header_h + 4
        # Match holdings: ~30px breathing room above the close button so the
        # bottom border is clearly visible, no clipping
        bottom_buffer = 46 + 12 + 50
        list_h = H - list_y - bottom_buffer

        canvas_frame = tk.Frame(win, bg=PANEL, highlightthickness=1,
                                 highlightbackground=EDGE, highlightcolor=EDGE)
        canvas_frame.place(x=PAD, y=list_y, width=section_w, height=list_h)

        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical",
                                 bg=EDGE, troughcolor=PANEL,
                                 activebackground=ACCENT, width=14,
                                 highlightthickness=0, bd=0)
        scrollbar.pack(side="right", fill="y")

        canvas = tk.Canvas(canvas_frame, bg=PANEL, bd=0, highlightthickness=0,
                           takefocus=0, yscrollcommand=scrollbar.set)
        inner = tk.Frame(canvas, bg=PANEL)

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw", width=section_w - 18)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=canvas.yview)

        _bind_scroll(canvas, canvas_frame, win)

        if not rows:
            empty = tk.Frame(inner, bg=PANEL, highlightthickness=1, highlightbackground=EDGE)
            empty.pack(fill="x", pady=4)
            tk.Label(empty, text="(no decisions yet)", bg=PANEL, fg=DIM,
                     font=(FONT_BODY, 11), anchor="w").pack(fill="x", padx=14, pady=18)
        else:
            wrap_w = self.cfg.display_width - PAD * 2 - 36

            # Formula names that might appear in summaries — tag them as chips.
            # Uses the canonical list shared with the analytics counter in db.py.
            FORMULAS = db.FORMULAS
            # Action keywords
            ACTIONS = {
                "buy": ("BUY", ACCENT),
                "bought": ("BUY", ACCENT),
                "sell": ("SELL", RED),
                "sold": ("SELL", RED),
                "hold": ("HOLD", DIM),
                "held": ("HOLD", DIM),
                "no trade": ("HOLD", DIM),
                "no trades": ("HOLD", DIM),
            }

            for r in rows:
                ts = r["ts"]
                try:
                    dt = datetime.fromisoformat(ts).astimezone()
                    ts_str = dt.strftime("%m-%d  %H:%M")
                except Exception:
                    ts_str = ts[:16]

                summary = r["summary"] or ""
                lower = summary.lower()

                # Detect action verdict
                verdict_label, verdict_color = "DECISION", DIM
                for kw, (label, color) in ACTIONS.items():
                    if kw in lower:
                        verdict_label, verdict_color = label, color
                        break

                # Detect formulas mentioned (shared canonical extractor)
                found_formulas = db.extract_formulas(summary)

                card = tk.Frame(inner, bg=PANEL, highlightthickness=1, highlightbackground=EDGE)
                card.pack(fill="x", pady=4)

                # ── Top strip: 🐟 timestamp + verdict 🐟  (fish bookends) ──
                top = tk.Frame(card, bg=PANEL)
                top.pack(fill="x", padx=14, pady=(10, 4))
                tk.Label(top, text="🐟", bg=PANEL, fg=LIGHT_GREEN,
                         font=mono_emoji_font(12, bold=True)).pack(side="left", padx=(0, 6))
                tk.Label(top, text=ts_str, bg=PANEL, fg=ACCENT,
                         font=title_font(12), anchor="w").pack(side="left")
                # Right side: verdict + matching 🐟
                tk.Label(top, text="🐟", bg=PANEL, fg=LIGHT_GREEN,
                         font=mono_emoji_font(12, bold=True)).pack(side="right", padx=(6, 0))
                tk.Label(top, text=verdict_label, bg=PANEL, fg=verdict_color,
                         font=title_font(10), anchor="e").pack(side="right")

                # ── Separator line ──
                tk.Frame(card, bg=ACCENT, height=1).pack(fill="x", padx=14, pady=(0, 8))

                # ── Section: EQUITY SNAPSHOT ──
                eq = db.equity_near(self.cfg.db_path, ts)
                if eq:
                    tk.Label(card, text="EQUITY AT TIME", bg=PANEL, fg=DIM,
                             font=title_font(9), anchor="w").pack(fill="x", padx=14, pady=(0, 2))
                    eq_row = tk.Frame(card, bg=PANEL)
                    eq_row.pack(fill="x", padx=14, pady=(0, 8))
                    pl = eq["day_pl"] or 0.0
                    pl_sign = "+" if pl >= 0 else ""
                    pl_color = UP_GREEN if pl >= 0 else DOWN_RED
                    tk.Label(eq_row, text=f"${eq['equity']:,.2f}", bg=PANEL, fg=FG,
                             font=title_font(12)).pack(side="left")
                    tk.Label(eq_row, text=f"  cash ${eq['cash']:,.2f}", bg=PANEL, fg=DIM,
                             font=(FONT_BODY, 10)).pack(side="left", padx=(8, 0))
                    tk.Label(eq_row, text=f"  Day {pl_sign}${pl:,.2f}", bg=PANEL, fg=pl_color,
                             font=(FONT_BODY, 10, "bold")).pack(side="left", padx=(8, 0))

                # ── Section: TRADES THIS CYCLE ──
                trades = db.trades_around(self.cfg.db_path, ts, window_seconds=120)
                tk.Label(card, text=f"TRADES ({len(trades)})", bg=PANEL, fg=DIM,
                         font=title_font(9), anchor="w").pack(fill="x", padx=14, pady=(0, 2))
                if trades:
                    # Column header
                    th = tk.Frame(card, bg=PANEL)
                    th.pack(fill="x", padx=14, pady=(0, 2))
                    tk.Label(th,
                             text=f"{'SIDE':<5} {'SYM':<6} {'SIZE':>10} {'@PRICE':>10} {'P/L':>12} {'STATUS':>10}",
                             bg=PANEL, fg=DIM, font=(FONT_MONO, 9), anchor="w").pack(side="left")
                    for t in trades:
                        raw_side = (t["side"] or "").lower()
                        sym = (t["symbol"] or "")[:6]
                        qty = t["qty"] or 0.0
                        notional = t["notional"]
                        if qty and qty > 0:
                            size_str = f"{qty:.0f} sh"
                        elif notional:
                            size_str = f"${notional:,.0f}"
                        else:
                            size_str = "—"
                        fill_px = t["filled_avg_price"] if "filled_avg_price" in t.keys() else None
                        price_str = f"${fill_px:,.2f}" if fill_px else "—"

                        # Distinguish 4 cases:
                        # - sell + matching prior buy = SELL  (closing a long)
                        # - sell + no matching buy   = SHORT (opening a short)
                        # - buy  + no prior sell      = BUY   (opening a long)
                        # - buy  + matching prior short = COVER (closing a short)
                        try:
                            buy_match = db.matching_buy_for_sell(
                                self.cfg.db_path, t["symbol"], t["id"])
                        except Exception:
                            buy_match = None
                        if raw_side == "sell":
                            if buy_match:
                                side_str, side_label = "SELL", "SELL"
                                side_color = DOWN_RED
                            else:
                                side_str, side_label = "SELL", "SHORT"
                                side_color = DOWN_RED
                        else:  # buy
                            side_str, side_label = "BUY", "BUY"
                            side_color = UP_GREEN

                        # Per-trade P/L: only meaningful for closing trades.
                        # Opening shorts have no realized P/L until covered.
                        pl_str, pl_color = "—", DIM
                        if (raw_side == "sell" and buy_match and fill_px and qty
                                and (t["status"] or "").lower() == "filled"
                                and buy_match["filled_avg_price"]):
                            buy_px = buy_match["filled_avg_price"]
                            pl_dollar = (fill_px - buy_px) * qty
                            sign = "+" if pl_dollar >= 0 else ""
                            pl_str = f"{sign}${pl_dollar:,.2f}"
                            pl_color = UP_GREEN if pl_dollar >= 0 else DOWN_RED

                        status = (t["status"] or "").lower()
                        status_color = (
                            UP_GREEN if status == "filled" else
                            ACCENT if status == "submitted" else
                            DOWN_RED if status == "rejected" else
                            DIM
                        )
                        row = tk.Frame(card, bg=PANEL)
                        row.pack(fill="x", padx=14, pady=(0, 1))
                        tk.Label(row, text=f"{side_label:<5}", bg=PANEL, fg=side_color,
                                 font=(FONT_MONO, 10, "bold")).pack(side="left")
                        tk.Label(row, text=f" {sym:<6}", bg=PANEL, fg=FG,
                                 font=(FONT_MONO, 10)).pack(side="left")
                        tk.Label(row, text=f"{size_str:>10}", bg=PANEL, fg=FG,
                                 font=(FONT_MONO, 10)).pack(side="left")
                        tk.Label(row, text=f"{price_str:>10}", bg=PANEL, fg=FG,
                                 font=(FONT_MONO, 10)).pack(side="left")
                        tk.Label(row, text=f"{pl_str:>12}", bg=PANEL, fg=pl_color,
                                 font=(FONT_MONO, 10, "bold")).pack(side="left")
                        tk.Label(row, text=f"{status:>10}", bg=PANEL, fg=status_color,
                                 font=(FONT_MONO, 10, "bold")).pack(side="left")
                        if t["reject_reason"]:
                            tk.Label(card, text=f"   ↳ {t['reject_reason']}", bg=PANEL, fg=RED,
                                     font=(FONT_BODY, 9, "italic"), anchor="w",
                                     wraplength=wrap_w).pack(fill="x", padx=14, pady=(0, 2))
                    # gap after trade list
                    tk.Frame(card, bg=PANEL, height=6).pack(fill="x")
                else:
                    tk.Label(card, text="(no trades placed)", bg=PANEL, fg=DIM,
                             font=(FONT_BODY, 10, "italic"), anchor="w").pack(
                        fill="x", padx=14, pady=(0, 8)
                    )

                # ── Section: SUMMARY ──
                # Clean up markdown + chain-of-thought preamble before render
                clean = summary or "(no summary)"
                if clean and clean != "(no summary)":
                    import re as _re
                    # Strip **bold** / *italic* / __underline__ markers
                    clean = _re.sub(r"\*\*(.+?)\*\*", r"\1", clean)
                    clean = _re.sub(r"\*(.+?)\*", r"\1", clean)
                    clean = _re.sub(r"__(.+?)__", r"\1", clean)
                    # Drop "Let me", "I'll now", "Let's now" preambles + the
                    # decision-process leak phrases. Keep the actual outcome.
                    junk = (r"^(Let me |I'll |I will |Let's |Now |Alright,? |"
                            r"Looking at |Based on |My final |Final )"
                            r"[^.!?]*[.!?]\s*")
                    while _re.match(junk, clean, flags=_re.IGNORECASE):
                        clean = _re.sub(junk, "", clean, count=1, flags=_re.IGNORECASE)
                    # Collapse extra blank lines
                    clean = _re.sub(r"\n{3,}", "\n\n", clean).strip()
                tk.Label(card, text="SUMMARY", bg=PANEL, fg=DIM,
                         font=title_font(9), anchor="w").pack(fill="x", padx=14, pady=(0, 2))
                tk.Label(card, text=clean, bg=PANEL, fg=FG,
                         font=(FONT_BODY, 11), anchor="w", justify="left",
                         wraplength=wrap_w).pack(fill="x", padx=14, pady=(0, 8))

                # ── Section: FORMULAS USED ──
                tk.Label(card, text="FORMULAS REFERENCED", bg=PANEL, fg=DIM,
                         font=title_font(9), anchor="w").pack(fill="x", padx=14, pady=(0, 2))
                if found_formulas:
                    chips = tk.Frame(card, bg=PANEL)
                    chips.pack(fill="x", padx=14, pady=(0, 10))
                    for f in found_formulas:
                        chip = tk.Label(chips, text=f"  {f}  ", bg=BTN_BG, fg=ACCENT,
                                        font=title_font(10), padx=2, pady=2)
                        chip.pack(side="left", padx=(0, 6), pady=2)
                else:
                    tk.Label(card, text="(none cited)", bg=PANEL, fg=DIM,
                             font=(FONT_BODY, 10, "italic"), anchor="w").pack(
                        fill="x", padx=14, pady=(0, 10)
                    )
            # Trailing spacer so the last card's bottom doesn't touch the panel border
            tk.Frame(inner, bg=PANEL, height=20).pack(fill="x")

        self._add_close_button(win)

    def _on_refresh(self) -> None:
        """Force-refresh equity numbers and chart from Alpaca."""
        import threading
        self._chart_cache = {}
        self._equity_in_flight = False
        self.refresh_btn_var.set("REFRESHING…")
        def _fetch():
            try:
                acc = self.broker.account()
                self.root.after(0, lambda: self._apply_equity(acc))
            except Exception:
                pass
            finally:
                self.root.after(0, lambda: self._finish_refresh())
        threading.Thread(target=_fetch, daemon=True).start()
        self.root.after(200, self._draw_home_equity_chart)

    def _finish_refresh(self) -> None:
        self.refresh_btn_var.set("DONE ✓")
        self.root.after(1500, lambda: self.refresh_btn_var.set("REFRESH"))

    def _on_about(self) -> None:
        win = self._make_popup("About FINbot")

        # ── Header — bold manifesto-style brand strip ──
        header_h = 140
        header = tk.Frame(win, bg=PANEL, highlightthickness=1, highlightbackground=EDGE)
        header.place(x=PAD, y=PAD, width=self.cfg.display_width - PAD * 2, height=header_h)

        # Title row — sharks bookend FINbot 3K, matching the home-screen header
        title_row = tk.Frame(header, bg=PANEL)
        title_row.place(x=20, y=20)
        tk.Label(title_row, text="🦈", bg=PANEL, fg=SHARK_BLUE,
                 font=mono_emoji_font(20, bold=True)).pack(side="left", padx=(0, 10))
        tk.Label(title_row, text="FINbot 3K", bg=PANEL, fg=ACCENT,
                 font=title_font(26)).pack(side="left")
        tk.Label(title_row, text="🦈", bg=PANEL, fg=SHARK_BLUE,
                 font=mono_emoji_font(20, bold=True)).pack(side="left", padx=(10, 0))

        # Tagline row — same line that lives on the home screen
        tk.Label(header, text="Algorithmic Agentic Trader  ·  Just keep trading, just keep trading",
                 bg=PANEL, fg=ACCENT, font=(FONT_BODY, 11, "italic"),
                 anchor="w").place(x=20, y=66)
        tk.Label(header, text="A self-contained quantitative trading agent.",
                 bg=PANEL, fg=FG, font=(FONT_BODY, 12, "italic"), anchor="w").place(x=20, y=86)
        tk.Label(header, text="Read the brain · weigh the data · take a position · learn.",
                 bg=PANEL, fg=DIM, font=(FONT_BODY, 10), anchor="w").place(x=20, y=108)

        # Right-side meta: provider on top, model below — all dim, no orange accents
        prov = self.cfg.provider.title()
        model_label = self.cfg.model if self.cfg.provider == "anthropic" else self.cfg.ollama_model
        tk.Label(header, text=prov.upper(), bg=PANEL, fg=DIM,
                 font=title_font(11), anchor="e").place(relx=1.0, x=-20, y=12, anchor="ne")
        tk.Label(header, text=model_label, bg=PANEL, fg=DIM,
                 font=(FONT_BODY, 11), anchor="e").place(relx=1.0, x=-20, y=32, anchor="ne")
        try:
            sess = self.broker.current_session()
        except Exception:
            sess = "unknown"
        sess_label = {"regular": "MARKET OPEN", "extended": "EXTENDED 24/5",
                      "closed": "MARKET CLOSED"}.get(sess, sess.upper())
        tk.Label(header, text=sess_label, bg=PANEL, fg=DIM,
                 font=title_font(10), anchor="e").place(relx=1.0, x=-20, y=58, anchor="ne")

        # ── Body — scrollable ─────────────────────────────────────
        body_y = PAD + header_h + PAD
        body_h = self.cfg.display_height - body_y - 46 - PAD * 2
        body_w = self.cfg.display_width - PAD * 2

        canvas_frame = tk.Frame(win, bg=PANEL, highlightthickness=1,
                                 highlightbackground=EDGE, highlightcolor=EDGE)
        canvas_frame.place(x=PAD, y=body_y, width=body_w, height=body_h)

        canvas = tk.Canvas(canvas_frame, bg=PANEL, bd=0, highlightthickness=0,
                           takefocus=0)
        inner = tk.Frame(canvas, bg=PANEL)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw", width=body_w - 4)
        canvas.pack(fill="both", expand=True)

        _bind_scroll(canvas, canvas_frame, win)

        # Inner text region width — Canvas window minus card padding (28) and a safety margin
        wrap = body_w - 4 - 36

        def section_card(title: str, emoji: str = "", emoji_color: str = ACCENT,
                         text_color: str = ACCENT) -> tk.Frame:
            """Section card with optional leading emoji rendered in its own color
            (monochrome font so fg actually applies)."""
            card = tk.Frame(inner, bg=PANEL, highlightthickness=1, highlightbackground=EDGE)
            card.pack(fill="x", pady=5)
            row = tk.Frame(card, bg=PANEL)
            row.pack(fill="x", padx=18, pady=(12, 4))
            if emoji:
                tk.Label(row, text=emoji, bg=PANEL, fg=emoji_color,
                         font=mono_emoji_font(15, bold=True)).pack(side="left", padx=(0, 8))
            tk.Label(row, text=title, bg=PANEL, fg=text_color,
                     font=title_font(15), anchor="w").pack(side="left")
            # subtle separator under title — matches the title text color
            tk.Frame(card, bg=text_color, height=1).pack(fill="x", padx=18, pady=(0, 8))
            return card

        def body_text(card: tk.Frame, text: str) -> None:
            tk.Label(card, text=text, bg=PANEL, fg=FG, font=(FONT_BODY, 12),
                     anchor="w", justify="left", wraplength=wrap).pack(
                fill="x", padx=18, pady=(0, 14)
            )

        def numbered_step(card: tk.Frame, n: int, text: str) -> None:
            row = tk.Frame(card, bg=PANEL)
            row.pack(fill="x", padx=18, pady=(0, 8))
            tk.Label(row, text=f"{n}.", bg=PANEL, fg=ACCENT,
                     font=title_font(12), anchor="nw", width=3).pack(side="left", anchor="n")
            tk.Label(row, text=text, bg=PANEL, fg=FG, font=(FONT_BODY, 12),
                     anchor="w", justify="left", wraplength=wrap - 36).pack(side="left", fill="x", expand=True)

        # ── MISSION ──────────────────────────────────────────────
        c = section_card("Mission", emoji="🎯", emoji_color=CRIMSON)
        body_text(c,
            "An autonomous algorithmic trader that hunts for asymmetric "
            "opportunities in three directions — long, short, and options — "
            "using formal quantitative models on live market data, all within "
            "hard-coded risk caps. It exists to find hidden gems the herd "
            "misses: pre-catalyst setups, mispriced volatility, broken charts "
            "the analysts haven't downgraded yet. Default-long is intellectual "
            "laziness. The market has both directions every day."
        )

        # ── HUNTING HIDDEN GEMS ──────────────────────────────────
        c = section_card("Hunting Hidden Gems", emoji="💎",
                         emoji_color=DOLPHIN_BLUE)
        body_text(c,
            "Every cycle opens with a NEGATIVE-DIRECTION SWEEP — before any "
            "long candidate is considered, the agent scans for shorts and "
            "options first. This is intentional. The asymmetric edge often "
            "lives where most retail traders refuse to look."
        )
        gem_patterns = [
            ("📉 Stealth Short",
             "Small/mid-cap with a recent analyst downgrade NOT YET reflected in the price action. Short before the herd catches up."),
            ("🎰 Pre-Earnings Mispricing",
             "IV percentile < 50 with a binary catalyst in 7–30 days. Cheap premium + defined risk = buy the option, not the stock."),
            ("📈 Sector Breakdown",
             "Sector leader breaking key support after rotation. Short the head-fake bounce; the real move comes 2–3 sessions later."),
            ("👤 Insider Cluster",
             "Three or more form-4 insider sales within 30 days at all-time highs. The people who run the company are voting with their wallets."),
            ("⚖ Vol Mispricing",
             "Realized 30d vol higher than implied vol on an upcoming event = market is asleep. Buy the option."),
            ("🚫 Hard-to-Borrow Bears",
             "A short thesis on a name Alpaca won't let you short directly? Buy a put. It's the only legal short."),
        ]
        for label, desc in gem_patterns:
            row = tk.Frame(c, bg=PANEL)
            row.pack(fill="x", padx=18, pady=(0, 6))
            tk.Label(row, text=label, bg=PANEL, fg=DOLPHIN_BLUE,
                     font=title_font(10), anchor="nw", width=22).pack(side="left", anchor="n")
            tk.Label(row, text=desc, bg=PANEL, fg=FG, font=(FONT_BODY, 11),
                     anchor="w", justify="left", wraplength=wrap - 180).pack(
                side="left", fill="x", expand=True)
        tk.Frame(c, bg=PANEL, height=8).pack(fill="x")

        # ── HOW IT THINKS ─────────────────────────────────────────
        c = section_card("How It Thinks · One Cycle", emoji="🐙",
                         emoji_color=PINK)
        numbered_step(c, 1,
            "📖  Loads the brain folder (Markdown formulas) into Claude's system prompt as authoritative context, plus the agent's running watchlist + recent rejections so it learns across cycles.")
        numbered_step(c, 2,
            "🔍  Surveys the world: get_account, get_positions, get_quote, get_recent_bars on multiple candidates. Pulls live numbers, never assumes.")
        numbered_step(c, 3,
            "🦈  NEGATIVE-DIRECTION SWEEP first: scans existing holdings + watchlist for fresh bad-news catalysts (downgrades, missed guidance, lawsuits, fraud, exec departures). Hidden gems often hide in your own book.")
        numbered_step(c, 4,
            "📰  get_news pulls from 6 aggregated feeds (Alpaca/Benzinga + Yahoo + Google + optional Finnhub/MarketAux/NewsAPI). Requires 3+ headlines from 2+ distinct sources. Cross-source confirmation = signal.")
        numbered_step(c, 5,
            "🧮  Applies the formulas via dedicated tools: get_beta (OLS vs SPY), get_var (historical 95% VaR), get_black_scholes (Δ/Γ/Θ/ν + IV), get_monte_carlo (probability paths). Real numbers, not vibes.")
        numbered_step(c, 6,
            "🎯  Picks the single highest-conviction direction this cycle: LONG (stock or call), SHORT (stock or put), or NO-TRADE. All three are equal-priority — there is no default.")
        numbered_step(c, 7,
            "📝  Writes a structured rationale citing the catalyst by source, formula values, and risk metrics. Calls place_order or place_option_order — text descriptions are not trades.")
        numbered_step(c, 8,
            "✅  Server-side validator enforces caps + shortability + options-eligibility BEFORE the order touches Alpaca. Rejections feed back as 'do not repeat' memory.")
        numbered_step(c, 9,
            "💾  finish() summary + full reasoning + every order logged to local SQLite. The HISTORY tab is the audit trail; ANALYSIS METHODS tracks formula citation frequency over time.")
        numbered_step(c, 10,
            "🧹  Next cycle starts with auto-cancel of stale unfilled orders + a fresh news sweep. Rinse, repeat, every 15 minutes during RTH.")

        # ── THE BRAIN · FORMULAS USED ─────────────────────────────
        c = section_card("The Brain · Formulas Used", emoji="🧠",
                         emoji_color=PINK)
        body_text(c,
            "Quantitative models the agent reaches for via dedicated tools. "
            "Each gates a specific decision — long entry, short entry, option "
            "pricing, position sizing, risk caps. Every trade rationale must "
            "cite one or more by name with real numerical values."
        )
        formulas = [
            ("CAPM", "Expected return = Rf + β(Rm − Rf). LONG eligible at α ≥ 1.5×(Rm−Rf); SHORT eligible at α ≤ Rf − 1.0×(Rm−Rf). Computed via get_beta."),
            ("Beta (β)", "OLS regression of daily log returns vs SPY (60-day window). Tool: get_beta. Returned with R² so you can judge fit quality."),
            ("Sharpe Ratio", "(R − Rf) / σ trailing 20 days. LONG entry needs ≥ 1.0; SHORT entry needs ≤ −0.5. Primary directional filter."),
            ("Black-Scholes", "Tool: get_black_scholes returns Δ, Γ, Θ, ν, theoretical price, implied vol from market premium via bisection. OPTIONS gate: Δ ∈ [0.30, 0.55], IV percentile < 50."),
            ("Monte Carlo", "Tool: get_monte_carlo runs 5,000 GBM paths calibrated on 60-day drift + vol. Returns terminal-price percentiles + probability of hitting target %. Required before any high-conviction trade."),
            ("ARIMA", "Short-horizon time-series momentum forecast. Triggers entries when next 5-bar magnitude ≥ 1× trailing ATR (long) or ≤ −1× ATR (short)."),
            ("Value at Risk", "Tool: get_var. Empirical 95% 1-day quantile loss of daily returns × position notional. Cited in every RISK section."),
            ("Yield Curve", "2y/10y term structure for macro risk-on/off bias. Influences sector rotation decisions and overall portfolio posture."),
            ("Duration", "Rate sensitivity. Used to understand cascade effects of Fed surprises into financials, REITs, utilities."),
        ]
        for name, desc in formulas:
            row = tk.Frame(c, bg=PANEL)
            row.pack(fill="x", padx=18, pady=(0, 6))
            tk.Label(row, text=name, bg=PANEL, fg=ACCENT,
                     font=title_font(11), anchor="nw", width=18).pack(side="left", anchor="n")
            tk.Label(row, text=desc, bg=PANEL, fg=FG, font=(FONT_BODY, 11),
                     anchor="w", justify="left", wraplength=wrap - 150).pack(
                side="left", fill="x", expand=True
            )
        # Trailing pad so the section breathes
        tk.Frame(c, bg=PANEL, height=8).pack(fill="x")

        # ── WHEN IT TRADES ────────────────────────────────────────
        c = section_card("When It Trades · Cadence + Auto-Resume", emoji="⏰",
                         emoji_color=CRIMSON)
        body_text(c,
            "Cadence is split: every {m} minutes during regular market hours "
            "(Mon-Fri 9:30am–4:00pm ET) when liquidity is real, and every {h} "
            "hours outside regular hours since most names aren't tradeable "
            "overnight anyway. Whether an order actually executes also depends "
            "on three independent gates — any one of them blocks trading; all "
            "three must be open."
            .format(m=self.cfg.cycle_minutes, h=self.cfg.offhours_cycle_hours)
        )
        # Three-state explanation
        gates = tk.Frame(c, bg=PANEL)
        gates.pack(fill="x", padx=18, pady=(0, 14))

        def gate_row(name: str, status: str, resume: str, color: str = FG) -> None:
            row = tk.Frame(gates, bg=PANEL)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=name, bg=PANEL, fg=ACCENT,
                     font=title_font(11), anchor="w", width=14).pack(side="left")
            tk.Label(row, text=status, bg=PANEL, fg=color,
                     font=(FONT_BODY, 11, "bold"), anchor="w", width=14).pack(side="left")
            tk.Label(row, text=resume, bg=PANEL, fg=DIM,
                     font=(FONT_BODY, 11), anchor="w").pack(side="left", fill="x", expand=True)

        # Manual pause
        paused = self.paused_getter()
        gate_row(
            "Manual pause",
            "PAUSED" if paused else "active",
            "tap RESUME to clear" if paused else "no auto-resume — user-controlled",
            color=RED if paused else ACCENT,
        )
        # Market session
        try:
            sess = self.broker.current_session()
        except Exception:
            sess = "unknown"
        sess_status = sess.upper()
        if sess == "regular":
            sess_resume = "trades freely; closes 4:00 PM ET"
        elif sess == "extended":
            sess_resume = "limit orders only; eligible symbols only"
        else:
            sess_resume = "auto-resumes Sunday 8 PM ET / Monday 9:30 AM ET"
        gate_row("Market session", sess_status, sess_resume,
                 color=ACCENT if sess in ("regular", "extended") else RED)
        # Trade window
        if self.cfg.trade_windows:
            in_win, win_msg = self.broker.trade_window_status(self.cfg.trade_windows)
            gate_row(
                "Trade window",
                "IN WINDOW" if in_win else "OUTSIDE",
                f"windows: {', '.join(self.cfg.trade_windows)} ET" if in_win
                    else f"auto-resumes at {win_msg.split(';')[-1].strip()}",
                color=ACCENT if in_win else RED,
            )
        else:
            gate_row("Trade window", "ALWAYS-ON", "no windows configured (trade anytime)",
                     color=ACCENT)

        # ── RATIONALE STANDARD ────────────────────────────────────
        c = section_card("Trade Rationale Standard", emoji="🪼",
                         emoji_color=PINK)
        body_text(c,
            "Every order carries weight. The agent is required to cite a specific "
            "Brain formula by name AND include at least two numerical values "
            "produced by that formula using current data. Vague reasoning is rejected "
            "by design."
        )
        # Three example tiers
        ex = tk.Frame(c, bg=PANEL)
        ex.pack(fill="x", padx=18, pady=(0, 14))
        for label, text, color in [
            ("REJECTED", '"CAPM looks favorable for SPY"', RED),
            ("OK",       '"CAPM β=1.0 on SPY suggests fair valuation"', DIM),
            ("REQUIRED", '"CAPM: β=1.0, Rf=4.5%, Rm=10% → expected return 10%. '
                          'Sharpe(30d) 1.15 confirms favorable risk-adjusted setup."', ACCENT),
        ]:
            row = tk.Frame(ex, bg=PANEL)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, bg=PANEL, fg=color,
                     font=title_font(10), anchor="w", width=12).pack(side="left", anchor="n")
            tk.Label(row, text=text, bg=PANEL, fg=FG,
                     font=(FONT_BODY, 11, "italic"), anchor="w", justify="left",
                     wraplength=wrap - 100).pack(side="left", fill="x", expand=True)

        # ── UNIVERSE ──────────────────────────────────────────────
        c = section_card("Universe", emoji="⭐", emoji_color=YELLOW)
        if self.cfg.watchlist:
            body_text(c,
                f"Restricted to {len(self.cfg.watchlist)} symbols set in config. "
                "The agent may only buy or sell names on this list."
            )
            chips = tk.Frame(c, bg=PANEL)
            chips.pack(fill="x", padx=18, pady=(0, 14))
            for s in self.cfg.watchlist:
                chip = tk.Label(chips, text=f"  {s}  ", bg=BTN_BG, fg=ACCENT,
                                font=title_font(10), padx=2, pady=2)
                chip.pack(side="left", padx=(0, 6), pady=2)
        else:
            body_text(c,
                "OPEN UNIVERSE — no watchlist restriction. The agent may trade any "
                "liquid US stock or ETF that Alpaca lists as tradable. It hunts "
                "across sectors and market caps for setups that match the Brain's "
                "formulas. Pre-flight validation against Alpaca's asset registry "
                "blocks hallucinated tickers before they touch the broker."
            )

        # ── DEPLOYMENT TARGET ─────────────────────────────────────
        c = section_card("Deployment · Sole-Purpose Pi", emoji="🐋")
        body_text(c,
            "FINbot's deployment target is a sole-purpose Raspberry Pi 5 8GB "
            "running 24/7 on a 5\" DSI touchscreen — total hardware cost ~$140 "
            "one-time. The Pi handles dashboard rendering, scheduling, news "
            "aggregation, order submission, and SQLite logging. Reasoning is "
            "rented from Anthropic's Claude API at ~$5–10/month — vastly "
            "cheaper than the GPU you'd need to run a comparable model "
            "locally, and far more reliable for the strict tool-calling "
            "discipline this agent demands. The Pi never needs your computer "
            "after setup; it boots to the dashboard, reconnects to wifi, and "
            "trades autonomously until you unplug it. Local model fallback "
            "(Ollama + Mistral-Nemo or Qwen) remains available via a single "
            "config switch."
        )

        # ── CURRENT CONFIG ────────────────────────────────────────
        c = section_card("Current Configuration", emoji="⚙", emoji_color=DIM)
        col_w = (wrap - 16) // 2
        cfg_grid = tk.Frame(c, bg=PANEL)
        cfg_grid.pack(fill="x", padx=18, pady=(0, 14))

        left_lines = [
            ("Provider", prov),
            ("Model", model_label),
            ("Market hours cycle", f"every {self.cfg.cycle_minutes} min"),
            ("Off-hours cycle", f"every {self.cfg.offhours_cycle_hours} hours"),
            ("Auto-cancel", f"unfilled > {self.cfg.auto_cancel_minutes} min"),
        ]
        right_lines = [
            ("Sim capital", f"${self.cfg.sim_capital:,.0f}" if self.cfg.sim_capital else "off"),
            ("Position cap", f"{self.cfg.max_position_pct:.0%} of equity"),
            ("Order notional cap", f"${self.cfg.max_order_notional:,.0f}"),
            ("Daily trade cap", str(self.cfg.max_daily_trades)),
            ("Trade windows", f"{len(self.cfg.trade_windows)} configured"
                                if self.cfg.trade_windows else "always-on"),
        ]

        row_h = 32

        def cfg_col(parent: tk.Frame, x: int, lines: list[tuple[str, str]]) -> None:
            col = tk.Frame(parent, bg=PANEL)
            col.place(x=x, y=0, width=col_w, height=4 + len(lines) * row_h)
            for i, (k, v) in enumerate(lines):
                row = tk.Frame(col, bg=PANEL)
                row.place(x=0, y=i * row_h, width=col_w)
                tk.Label(row, text=k, bg=PANEL, fg=DIM,
                         font=(FONT_BODY, 12), anchor="w").pack(side="left")
                tk.Label(row, text=v, bg=PANEL, fg=FG,
                         font=(FONT_BODY, 12, "bold"), anchor="e").pack(side="right")

        cfg_grid.configure(height=4 + max(len(left_lines), len(right_lines)) * row_h)
        cfg_col(cfg_grid, 0, left_lines)
        cfg_col(cfg_grid, col_w + 16, right_lines)

        # ── BUILD ─────────────────────────────────────────────────
        c = section_card("The Hardware Future", emoji="🐳")
        body_text(c,
            "Tuned on a Windows PC, designed to graduate to a Raspberry Pi 5 8GB "
            "with a 5-inch Hosyond DSI touchscreen. Once on the Pi, no PC required, "
            "no recurring cost, no internet dependency beyond Alpaca itself. "
            "Push code from your desktop, the Pi pulls it via Git, restarts the "
            "service, and continues trading. Forever."
        )

        # Trailing spacer so the last section card doesn't clip the panel border
        tk.Frame(inner, bg=PANEL, height=20).pack(fill="x")

        self._add_close_button(win)

    # ── Refresh loops ─────────────────────────────────────────────

    def _tick_clock(self) -> None:
        """Lightweight 1s clock tick — runs independently of the heavier _refresh."""
        self.time_var.set(datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    # Active cycle spinner: rotating sea creatures (FINbot ocean rotation)
    _SPINNER_FRAMES = ("🐠", "🐟", "🐡", "🦈", "🐙", "🦀", "🐚", "🪼")
    _FISH_SPINNER_FRAMES = _SPINNER_FRAMES  # alias — same frames everywhere
    # Sleep cycle: drifting z z z, centered horizontally so the label
    # (anchor="center") puts them dead-center on the screen.
    _SLEEP_FRAMES = (
        "       z       ",
        "      z z      ",
        "     z z z     ",
        "    z z z z    ",
        "   z z z z z   ",
        "    z z z z    ",
        "     z z z     ",
        "      z z      ",
        "       z       ",
        "               ",
    )

    def _tick_spinner(self) -> None:
        """100ms tick — animate spinner only when the agent is genuinely working
        AND not paused. Hides when paused or idle."""
        # If paused, show no spinner regardless of background cycle state
        try:
            if self.paused_getter():
                self.spinner_var.set("")
                self._spin_idx = 0
                self._spin_started_at = None
                self.root.after(500, self._tick_spinner)
                return
        except Exception:
            pass
        try:
            cycling = self.is_cycling()
        except Exception:
            cycling = False
        if cycling:
            # Track when this cycle started so we can flag hangs
            if not getattr(self, "_spin_started_at", None):
                self._spin_started_at = datetime.now()
            frames = (self._FISH_SPINNER_FRAMES if getattr(self, "_fish_mode", False)
                      else self._SPINNER_FRAMES)
            elapsed = (datetime.now() - self._spin_started_at).total_seconds()
            # Local Nemo runs slower than cloud Sonnet — give it a longer
            # leash before flagging a hang
            hang_threshold = 720 if getattr(self, "_fish_mode", False) else 360
            if elapsed > hang_threshold:
                self.spinner_var.set("⚠ " + frames[self._spin_idx])
                self.spinner_label.configure(fg=RED)
            else:
                self.spinner_var.set(frames[self._spin_idx])
                self.spinner_label.configure(fg=ACCENT)
            self._spin_idx = (self._spin_idx + 1) % len(frames)
            self.root.after(100, self._tick_spinner)
        else:
            # Cycle complete — hide the spinner entirely
            self.spinner_var.set("")
            self._spin_idx = 0
            self._spin_started_at = None
            self.root.after(500, self._tick_spinner)

    def _tick_equity(self) -> None:
        """3-second equity tick. Snapshots logged every 30s for a lively chart."""
        import threading
        if getattr(self, "_equity_in_flight", False):
            self.root.after(1000, self._tick_equity)
            return
        self._equity_in_flight = True

        def _fetch():
            try:
                acc = self.broker.account()
                self.root.after(0, lambda: self._apply_equity(acc))
                # Equity snapshot every 5 ticks (~5s) so the chart line grows
                # in near-real-time without flooding the DB.
                self._equity_log_counter = getattr(self, "_equity_log_counter", 0) + 1
                if self._equity_log_counter >= 5:
                    self._equity_log_counter = 0
                    try:
                        log_eq = self._sim_equity(acc.equity)
                        scale = self.cfg.sim_capital / 100_000.0 if self.cfg.sim_capital else 1.0
                        log_cash = acc.cash * scale
                        log_pl = acc.day_pl * scale if acc.day_pl else acc.day_pl
                        db.log_equity(self.cfg.db_path, log_eq, log_cash, log_pl)
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                self._equity_in_flight = False
        threading.Thread(target=_fetch, daemon=True).start()
        self.root.after(1000, self._tick_equity)

    def _draw_stroked_pl(self, text: str, color: str) -> None:
        """Render Day P/L on a Canvas in green (up) or red (down)."""
        c = self._pl_canvas
        c.delete("all")
        font = (FONT_BODY, 14, "bold")
        c.create_text(2, 12, text=text, fill=color, font=font, anchor="w")

    def _sim_equity(self, real_equity: float) -> float:
        """If sim_capital is set, ratio-scale real Alpaca equity to sim scale.
        Baseline is always $100K (Alpaca paper start). +13% on $100K → +13% on $3K."""
        if not self.cfg.sim_capital:
            return real_equity
        return self.cfg.sim_capital * (real_equity / 100_000.0)

    def _apply_equity(self, acc) -> None:
        real_eq = acc.equity
        display_eq = self._sim_equity(real_eq)
        self._live_equity = display_eq
        self._raw_day_pl = acc.day_pl
        try:
            STARTING = self.cfg.sim_capital or 100_000.00
            up = display_eq >= STARTING
            self.equity_icon_var.set("▲" if up else "▼")
            self.equity_icon_label.configure(fg=UP_GREEN if up else DOWN_RED)
            self.equity_var.set(f"${display_eq:,.2f}")
            if self.cfg.sim_capital:
                scale = self.cfg.sim_capital / 100_000.0
                display_cash = acc.cash * scale
                display_bp = acc.buying_power * scale
            else:
                display_cash = acc.cash
                display_bp = acc.buying_power
            self.cash_var.set(f"💵 Cash ${display_cash:,.2f}")
            self.bp_var.set(f"💰 Buying Power ${display_bp:,.2f}")

            # Day P/L: ratio-scale to sim capital
            day_pl = acc.day_pl
            if self.cfg.sim_capital:
                scale = self.cfg.sim_capital / 100_000.0
                day_pl = day_pl * scale if day_pl else day_pl
            base = display_eq - day_pl if day_pl is not None else None

            if day_pl is not None:
                sign = "+" if day_pl >= 0 else ""
                pct = (day_pl / base * 100) if base else 0.0
                day_icon = "🔼" if day_pl >= 0 else "🔽"
                pl_text = f"{day_icon} Day {sign}${day_pl:,.2f}  ({sign}{pct:.2f}%)"
                pl_color = UP_GREEN if day_pl >= 0 else DOWN_RED
                self._draw_stroked_pl(pl_text, pl_color)

            try:
                session = self.broker.current_session()
            except Exception:
                session = "unknown"
            if session == "regular":
                status, color = "☀ Market Open", PEACH
            elif session == "extended":
                status, color = "🌙 Extended (24/5)", PEACH
            else:
                status, color = "🌙 Market Closed", PEACH
            # Trade-window overlay: if windows configured and we're outside one,
            # show that instead — it's the actual blocker for orders.
            if self.cfg.trade_windows and session != "closed":
                in_win, win_msg = self.broker.trade_window_status(self.cfg.trade_windows)
                if not in_win:
                    next_part = win_msg.split(";")[-1].strip()  # "next 09:35 ET"
                    status = f"● Window: {next_part}"
                    color = DIM
            self.market_var.set(status)
            self.market_label.configure(fg=color)
        except Exception:
            pass  # network blip — keep last numbers visible

    def _refresh(self) -> None:
        """Slower 2-second tick — positions table, last action panel,
        provider/paused indicators. Order-status sync runs in a background
        thread so the UI never freezes during multi-call Alpaca polling."""
        import threading
        # 5s poll × 6 ticks ≈ 30s between order-status syncs
        self._sync_counter += 1
        if self._sync_counter >= 6:
            self._sync_counter = 0

            def _bg_sync():
                try:
                    from quant_pi.agent.trader import sync_order_statuses
                    sync_order_statuses(self.cfg, self.broker)
                except Exception:
                    pass
            threading.Thread(target=_bg_sync, daemon=True).start()

        # Redraw the home-page equity chart (cheap — local SQLite query, no API)
        try:
            self._draw_home_equity_chart()
        except Exception:
            pass

        # Local-only updates (no HTTP) — safe to run on main thread
        try:
            paused = self.paused_getter()
            self.paused_var.set("🛑 Stopped" if paused else "")
            self.pause_btn_var.set("START" if paused else "STOP")

            provider_label = (
                f"{self.cfg.provider.title()}: {self.cfg.model}"
                if self.cfg.provider == "anthropic"
                else f"{self.cfg.provider.title()}: {self.cfg.ollama_model}"
            )
            self.provider_var.set(provider_label)

            row = db.latest_decision(self.cfg.db_path)
            if row:
                ts = row["ts"]
                try:
                    dt = datetime.fromisoformat(ts).astimezone()
                    self.action_ts_var.set(dt.strftime("%m-%d %H:%M"))
                except Exception:
                    self.action_ts_var.set(ts[:16])
                current = self.action_text.get("1.0", "end-1c")
                if current != row["summary"]:
                    self._set_action_text(row["summary"])
        except Exception as e:
            self._set_action_text(f"display error: {e}")

        self.root.after(int(self.cfg.display_poll_seconds * 1000), self._refresh)

    def _apply_positions(self, positions) -> None:
        # Positions table now lives in the HISTORY popup. The home page shows
        # the equity chart instead (drawn separately by _draw_home_equity_chart).
        pass

    def _on_chart_hover(self, event) -> None:
        """Crosshair snaps to nearest data point. No interpolation."""
        self._chart_mouse_event = event
        c = self._equity_canvas
        c.delete("crosshair")
        st = self._chart_state
        if not st or st["n"] < 2:
            return
        x = event.x
        if x < st["x0"] or x > st["x1"]:
            return

        # Find nearest data point by pixel distance
        px_xs = st.get("px_xs")
        if not px_xs:
            return
        best_i = 0
        best_dist = abs(x - px_xs[0])
        for i in range(1, len(px_xs)):
            d = abs(x - px_xs[i])
            if d < best_dist:
                best_dist = d
                best_i = i

        # Snap to nearest point
        px = px_xs[best_i]
        v = st["values"][best_i]

        # At last point, use live equity
        if best_i >= st["n"] - 1:
            live = getattr(self, "_live_equity", None)
            if live is not None:
                v = live

        # Format timestamp
        t_str = ""
        try:
            t = datetime.fromisoformat(st["timestamps"][best_i])
            if t.tzinfo:
                from zoneinfo import ZoneInfo
                t = t.astimezone(ZoneInfo("America/Chicago"))
            t_str = t.strftime("%H:%M") if t.date() == datetime.now().date() else t.strftime("%m/%d")
        except Exception:
            pass

        py = st["y0"] + (1 - (v - st["y_lo"]) / st["rng"]) * st["h"]

        c.create_line(px, st["y0"], px, st["y1"],
                      fill=DIM, dash=(2, 3), width=1, tags="crosshair")
        c.create_oval(px - 4, py - 4, px + 4, py + 4,
                      fill=ACCENT, outline="#000000", width=1, tags="crosshair")
        # Fixed label pinned to top-right of chart (below title area)
        text = f"{t_str}   ${v:,.2f}"
        c.create_text(st["x1"], st["y0"] + 2, anchor="ne", fill=ACCENT,
                      font=(FONT_MONO, 9, "bold"),
                      text=text, tags="crosshair")

    def _market_open_dt(self) -> datetime:
        """Today's market open as an aware local datetime: 9:30 AM ET."""
        from zoneinfo import ZoneInfo
        ct = ZoneInfo("America/Chicago")
        now_ct = datetime.now(ct)
        open_ct = now_ct.replace(hour=8, minute=30, second=0, microsecond=0)
        return open_ct.astimezone()

    def _market_open_equity(self) -> Optional[float]:
        """Equity at today's market open — the first snapshot taken at or
        after 9:30 AM ET. Returns None if no snapshot has been recorded yet
        today (e.g. fresh start or pre-open)."""
        try:
            cutoff = self._market_open_dt().isoformat()
            with db.connect(self.cfg.db_path) as conn:
                row = conn.execute(
                    "SELECT equity FROM equity_snapshots WHERE ts >= ? "
                    "ORDER BY id ASC LIMIT 1",
                    (cutoff,),
                ).fetchone()
            return float(row["equity"]) if row else None
        except Exception:
            return None

    @staticmethod
    def _snap_to_close(rows: list, tz) -> list:
        """Snap daily bar timestamps to 3pm CT (market close) so the TOTAL
        chart shows meaningful times instead of Alpaca's midnight-UTC artifacts."""
        from datetime import datetime as _dt
        out = []
        for row in rows:
            try:
                t = _dt.fromisoformat(row["ts"])
                if t.tzinfo:
                    t = t.astimezone(tz)
                t = t.replace(hour=15, minute=0, second=0, microsecond=0)
                out.append({"ts": t.isoformat(), "equity": row["equity"]})
            except Exception:
                out.append(row)
        return out

    @staticmethod
    def _interpolate_weekends(rows: list, tz) -> list:
        """Insert Saturday and Sunday points between Friday and Monday so the
        TOTAL chart shows a gradual slope over weekends instead of a jump."""
        from datetime import datetime as _dt, timedelta as _td
        out = []
        for i, row in enumerate(rows):
            out.append(row)
            if i >= len(rows) - 1:
                continue
            try:
                t1 = _dt.fromisoformat(row["ts"])
                t2 = _dt.fromisoformat(rows[i + 1]["ts"])
                if t1.tzinfo:
                    t1 = t1.astimezone(tz)
                if t2.tzinfo:
                    t2 = t2.astimezone(tz)
            except Exception:
                continue
            gap_days = (t2.date() - t1.date()).days
            if gap_days <= 1:
                continue
            eq1 = float(row["equity"])
            eq2 = float(rows[i + 1]["equity"])
            for d in range(1, gap_days):
                frac = d / gap_days
                mid_t = t1 + _td(days=d)
                mid_eq = eq1 + (eq2 - eq1) * frac
                out.append({"ts": mid_t.isoformat(), "equity": mid_eq})
        return out

    def _patch_live_equity(self, range_key: str, rows: list) -> list:
        """Replace the last data point's equity with the live value so the
        chart tip always matches the header number."""
        live = getattr(self, "_live_equity", None)
        if live is not None and rows:
            rows = list(rows)
            rows[-1] = dict(rows[-1], equity=live)
        return rows

    def _switch_chart_tab(self, tab_name: str) -> None:
        self._chart_range = tab_name
        self._chart_cache = {}
        for name, btn in self._chart_tab_btns.items():
            btn.configure(fg=ACCENT if name == tab_name else DIM)
        self._draw_home_equity_chart()

    def _equity_rows_for_range(self, range_key: str) -> list:
        """Equity curve from Alpaca portfolio history API.
        DAILY = midnight-to-midnight 5-min bars. TOTAL = all-time 1-day bars.
        Falls back to local DB if API fails. Never returns empty — appends
        last known equity if needed so the chart always shows something."""
        import time as _time
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo

        cache_key = range_key
        now_ts = _time.time()
        cached = getattr(self, "_chart_cache", {})
        if cached.get("key") == cache_key and now_ts - cached.get("ts", 0) < 10:
            return self._patch_live_equity(range_key, cached.get("rows", []))

        ET = ZoneInfo("America/Chicago")
        rows = []

        try:
            if range_key == "DAILY":
                today = _dt.now(ET).strftime("%Y-%m-%d")
                rows = self.broker.portfolio_history(
                    timeframe="5Min", date_start=today)
            else:
                rows = self.broker.portfolio_history(
                    period="all", timeframe="1D")
        except Exception:
            pass

        if rows and self.cfg.sim_capital:
            scale = self.cfg.sim_capital / 100_000.0
            rows = [dict(r, equity=float(r["equity"]) * scale) for r in rows]

        if not rows:
            try:
                with db.connect(self.cfg.db_path) as conn:
                    if range_key == "DAILY":
                        now_et = _dt.now(ET)
                        cutoff = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
                        raw = conn.execute(
                            "SELECT ts, equity FROM equity_snapshots WHERE ts >= ? ORDER BY id ASC",
                            (cutoff.isoformat(),)
                        ).fetchall()
                    else:
                        raw = conn.execute(
                            "SELECT ts, equity FROM equity_snapshots ORDER BY id ASC"
                        ).fetchall()
                rows = [{"ts": r["ts"], "equity": r["equity"]} for r in raw]
            except Exception:
                pass

        if not rows:
            try:
                with db.connect(self.cfg.db_path) as conn:
                    last = conn.execute(
                        "SELECT ts, equity FROM equity_snapshots ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                if last:
                    eq = float(last["equity"])
                    ts = last["ts"]
                    rows = [{"ts": ts, "equity": eq}, {"ts": ts, "equity": eq}]
            except Exception:
                pass

        # TOTAL: snap daily bars to 3pm CT (market close) and interpolate weekends
        if range_key == "TOTAL" and len(rows) >= 2:
            rows = self._snap_to_close(rows, ET)
            rows = self._interpolate_weekends(rows, ET)

        # DAILY: filter to CT midnight-to-midnight and prepend midnight point
        if range_key == "DAILY" and rows:
            now_et = _dt.now(ET)
            ct_midnight = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
            filtered = []
            for row in rows:
                try:
                    t = _dt.fromisoformat(row["ts"])
                    if t.tzinfo:
                        t = t.astimezone(ET)
                    if t >= ct_midnight:
                        filtered.append(row)
                except Exception:
                    filtered.append(row)
            rows = filtered if filtered else rows
            now_et = _dt.now(ET)
            first_eq = float(rows[0]["equity"])
            try:
                first_t = _dt.fromisoformat(rows[0]["ts"])
                if first_t.tzinfo:
                    first_t = first_t.astimezone(ET)
                if first_t.hour > 0 or first_t.minute > 0:
                    midnight = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
                    rows.insert(0, {"ts": midnight.isoformat(), "equity": first_eq})
            except Exception:
                pass

        if rows:
            now_et = _dt.now(ET)
            last_eq = float(rows[-1]["equity"])
            live_eq = getattr(self, "_live_equity", last_eq)
            rows.append({"ts": now_et.isoformat(), "equity": live_eq})

        self._chart_cache = {"key": cache_key, "ts": now_ts, "rows": rows}
        return self._patch_live_equity(range_key, rows)

    def _draw_home_equity_chart(self) -> None:
        """Render the equity chart using pure tkinter Canvas primitives.
        No matplotlib, no Pillow — safe on Pi 5 / Python 3.13 ARM64.
        Falls back to a 'no data yet' message if rows < 2."""
        try:
            rows = self._equity_rows_for_range(self._chart_range)
            c = self._equity_canvas
            w = max(c.winfo_width(), 200)
            h = max(c.winfo_height(), 100)

            if len(rows) < 2:
                c.delete("all")
                c.create_text(w / 2, h / 2,
                              text="(equity history will populate as cycles run)",
                              fill=DIM, font=(FONT_BODY, 10, "italic"))
                self._equity_caption_var.set("")
                self._equity_context_var.set("")
                return

            self._render_chart_to_canvas(rows, w, h)

            # Caption — context-aware per tab
            first_eq = float(rows[0]["equity"])
            last_eq = float(rows[-1]["equity"])
            if self._chart_range == "DAILY":
                scale = self.cfg.sim_capital / 100_000.0 if self.cfg.sim_capital else 1.0
                day_pl = getattr(self, "_raw_day_pl", None)
                if day_pl is not None:
                    change = day_pl * scale
                    base = last_eq - change
                    pct = (change / base * 100) if base else 0.0
                else:
                    change = last_eq - first_eq
                    pct = (change / first_eq * 100) if first_eq else 0.0
                sign = "+" if change >= 0 else ""
                self._equity_caption_var.set(f"{sign}${change:,.2f} ({sign}{pct:.2f}%)")
                self._equity_context_var.set("today")
            else:
                seed = self.cfg.sim_capital if self.cfg.sim_capital else first_eq
                change = last_eq - seed
                sign = "+" if change >= 0 else ""
                pct = (change / seed * 100) if seed else 0.0
                self._equity_caption_var.set(f"{sign}${change:,.2f} ({sign}{pct:.2f}%)")
                self._equity_context_var.set(f"since start ${seed:,.0f}")
            self._equity_caption_label.configure(
                fg=UP_GREEN if change >= 0 else DOWN_RED
            )
        except Exception:
            pass

    def _render_chart_to_canvas(self, rows: list, w: int, h: int) -> None:
        """Pure-tkinter equity chart — no matplotlib, no Pillow, no native libs.
        Draws directly on self._equity_canvas with create_line / create_polygon
        so nothing can segfault on Pi 5 / Python 3.13."""

        # ── parse timestamps & values ──────────────────────────────
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/Chicago")
        xs_raw, ys = [], []
        for r in rows:
            try:
                t = datetime.fromisoformat(r["ts"])
                if t.tzinfo is not None:
                    t = t.astimezone(_ET).replace(tzinfo=None)
                xs_raw.append(t)
                ys.append(float(r["equity"]))
            except Exception:
                continue
        if len(xs_raw) < 2:
            return

        n = len(xs_raw)
        y_min = min(ys)
        y_max = max(ys)
        y_span = y_max - y_min or 1.0

        # ── layout margins ─────────────────────────────────────────
        LEFT   = 72   # room for y-axis labels ($X,XXX)
        RIGHT  = 8
        TOP    = 18   # room for session boundary labels above chart
        BOTTOM = 22   # room for x-axis labels

        x0 = LEFT
        x1 = w - RIGHT
        y0 = TOP
        y1 = h - BOTTOM
        pw = max(1, x1 - x0)   # plot width  in pixels
        ph = max(1, y1 - y0)   # plot height in pixels

        # Add vertical buffer so the line never touches the edges
        y_pad = y_span * 0.15
        y_lo  = y_min - y_pad
        y_hi  = y_max + y_pad
        y_rng = y_hi - y_lo

        _daily = getattr(self, "_chart_range", None) == "DAILY"
        if _daily:
            day_start = xs_raw[0].replace(hour=0, minute=0, second=0, microsecond=0)
            day_seconds = 24 * 3600.0
            def px_x(i: int) -> float:
                elapsed = (xs_raw[i] - day_start).total_seconds()
                return x0 + (elapsed / day_seconds) * pw
        else:
            def px_x(i: int) -> float:
                return x0 + i * pw / (n - 1)

        def px_y(v: float) -> float:
            return y0 + (1.0 - (v - y_lo) / y_rng) * ph

        c = self._equity_canvas
        c.delete("all")

        # ── background ─────────────────────────────────────────────
        c.create_rectangle(0, 0, w, h, fill=PANEL, outline="")

        # ── DAILY session boundary lines (vertical dashes) ─────────
        if _daily:
            boundaries = [
                (2.0,  DIM,    "EXT"),
                (8.5,  YELLOW, "OPEN"),
                (15.0, YELLOW, "CLOSE"),
                (19.0, DIM,    "EXT"),
            ]
            for hr, color, label in boundaries:
                bx = x0 + (hr / 24.0) * pw
                c.create_line(bx, y0, bx, y1, fill=color, width=1, dash=(4, 4))
                c.create_text(bx, y0 - 1, anchor="s", text=label,
                              fill=color, font=(FONT_MONO, 7))

        # ── subtle fill under the line ─────────────────────────────
        pts = []
        for i in range(n):
            pts.extend([px_x(i), px_y(ys[i])])
        pts.extend([px_x(n - 1), y1, px_x(0), y1])
        c.create_polygon(pts, fill="#001a2e", outline="")

        # ── horizontal grid lines + y-axis labels ──────────────────
        # 4 evenly-spaced ticks in the value range
        n_ticks = 5
        for ti in range(1, n_ticks):
            frac = ti / n_ticks
            v = y_lo + frac * y_rng
            py = y0 + (1.0 - frac) * ph
            grid_x1 = px_x(n - 1) if _daily else x1
            c.create_line(x0, py, grid_x1, py, fill="#0a2840", width=1, dash=(3, 4))
            if abs(v) >= 1_000_000:
                label = f"${v/1_000_000:.1f}M"
            elif abs(v) >= 10_000 and y_rng >= 5_000:
                label = f"${v/1_000:.0f}K"
            elif abs(v) >= 1_000:
                label = f"${v:,.0f}"
            else:
                label = f"${v:.0f}"
            c.create_text(x0 - 4, py, anchor="e", text=label,
                          fill=DIM, font=(FONT_MONO, 8))

        # ── x-axis labels ──────────────────────────────────────────
        if _daily:
            for hour in (0, 4, 8, 12, 16, 20, 24):
                px = x0 + (hour / 24.0) * pw
                lbl = f"{hour:02d}:00" if hour < 24 else "00:00"
                anc = "nw" if hour == 0 else ("ne" if hour == 24 else "n")
                c.create_text(px, y1 + 2, anchor=anc, text=lbl,
                              fill=DIM, font=(FONT_MONO, 8))
        else:
            t_start = xs_raw[0]
            t_end   = xs_raw[-1]
            t_span  = (t_end - t_start).total_seconds()
            n_x = max(2, min(6, int(pw // 90)))
            for ti in range(n_x):
                frac = ti / max(1, n_x - 1)
                idx  = int(round(frac * (n - 1)))
                px   = px_x(idx)
                t    = xs_raw[idx]
                if t_span < 86400 * 2:
                    lbl = t.strftime("%H:%M")
                elif t_span < 86400 * 60:
                    lbl = t.strftime("%m/%d")
                else:
                    lbl = t.strftime("%m/%d/%y")
                anc = "nw" if ti == 0 else ("ne" if ti == n_x - 1 else "n")
                c.create_text(px, y1 + 2, anchor=anc, text=lbl,
                              fill=DIM, font=(FONT_MONO, 8))

        # ── main equity line ───────────────────────────────────────
        line_pts = []
        for i in range(n):
            line_pts.extend([px_x(i), px_y(ys[i])])
        if len(line_pts) >= 4:
            c.create_line(line_pts, fill=ACCENT, width=2, smooth=False)

        # ── end-point dot ──────────────────────────────────────────
        ex = px_x(n - 1)
        ey = px_y(ys[-1])
        r  = 4
        c.create_oval(ex - r, ey - r, ex + r, ey + r,
                      fill=ACCENT, outline=PANEL, width=2)

        # ── store chart_state for hover crosshair ──────────────────
        self._chart_state = {
            "n":          n,
            "x0":         x0,
            "x1":         x1,
            "y0":         y0,
            "y1":         y1,
            "h":          ph,
            "y_lo":       y_lo,
            "rng":        y_rng,
            "values":     ys,
            "timestamps": [r["ts"] for r in rows[:n]],
            "px_xs":      [px_x(i) for i in range(n)],
        }

        # Re-draw crosshair if mouse is still over the chart
        evt = getattr(self, "_chart_mouse_event", None)
        if evt is not None:
            self._on_chart_hover(evt)

    # ── Sleep overlay (tablet idle behavior) ──────────────────────

    def _on_interaction(self, event=None) -> None:
        """Any click / key / motion resets the idle timer and wakes the screen."""
        self._last_interaction = datetime.now()
        if self._sleeping:
            self._wake()

    def _tick_sleep(self) -> None:
        """Check every 5s whether to enter sleep mode based on idle time."""
        timeout = self.cfg.display_sleep_after_minutes * 60
        if timeout > 0 and not self._sleeping:
            idle = (datetime.now() - self._last_interaction).total_seconds()
            if idle >= timeout:
                self._sleep()
        self.root.after(5000, self._tick_sleep)

    def _sleep(self) -> None:
        if self._sleeping:
            return
        self._sleeping = True
        W, H = self.cfg.display_width, self.cfg.display_height
        # Full-screen black overlay on top of everything
        self._sleep_overlay = tk.Frame(self.root, bg="#000000")
        self._sleep_overlay.place(x=0, y=0, width=W, height=H)
        # Tap anywhere on the overlay to wake
        self._sleep_overlay.bind("<Button>", self._on_interaction)
        # Big orange "sleeping" — large font, fg=ACCENT, on black panel.
        # Uses ● filled circle (text char, renders in fg color reliably).
        self._sleep_var = tk.StringVar(value=self._SLEEP_FRAMES[0])
        tk.Label(self._sleep_overlay, textvariable=self._sleep_var, bg="#000000",
                 fg=ACCENT, font=(FONT_MONO, 48, "bold")).place(
            relx=0.5, rely=0.45, anchor="center"
        )
        # Clock at the bottom — only other element on the sleep screen
        self._sleep_clock_var = tk.StringVar(value="")
        tk.Label(self._sleep_overlay, textvariable=self._sleep_clock_var, bg="#000000",
                 fg=DIM, font=(FONT_MONO, 16)).place(relx=0.5, rely=0.78, anchor="center")
        self._tick_sleep_anim()

    def _wake(self) -> None:
        if not self._sleeping:
            return
        self._sleeping = False
        try:
            self._sleep_overlay.destroy()
        except Exception:
            pass

    def _tick_sleep_anim(self) -> None:
        """Animate the bowling strike celebration loop while sleeping."""
        if not self._sleeping:
            return
        self._sleep_var.set(self._SLEEP_FRAMES[self._sleep_idx])
        self._sleep_idx = (self._sleep_idx + 1) % len(self._SLEEP_FRAMES)
        try:
            self._sleep_clock_var.set(datetime.now().strftime("%H:%M:%S"))
        except Exception:
            pass
        self.root.after(1500, self._tick_sleep_anim)

    def run(self) -> None:
        self.root.after(100, self._refresh)        # 2 s — positions, action panel
        self.root.after(100, self._tick_equity)    # 1 s — equity, P/L, market dot
        self.root.after(100, self._tick_clock)     # 1 s — clock
        self.root.after(100, self._tick_spinner)   # 100 ms when cycling, 500 ms idle
        self.root.after(5000, self._tick_sleep)    # 5 s idle-check tick
        self.root.mainloop()

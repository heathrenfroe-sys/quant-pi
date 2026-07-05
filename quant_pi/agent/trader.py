import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from anthropic import Anthropic

from quant_pi.brain.loader import BrainCorpus
from quant_pi.broker.alpaca import Broker
from quant_pi.config import Config
from quant_pi import news as news_agg
from quant_pi.store import db

SYSTEM_TEMPLATE = """You are a paper-trading quant agent. You manage a small Alpaca paper-trading portfolio and you must follow the user's quantitative formulas (the "Brain") below.

══════════════════════════════════════════════════════════════════════════════
EXECUTION DISCIPLINE — READ THIS FIRST
══════════════════════════════════════════════════════════════════════════════
RULE 1: If you decide to trade, you MUST invoke the `place_order` tool (or
        `place_option_order`). Writing a text description of the trade is
        NOT a trade. "I will sell AMD" in your reply is invalid — only the
        tool call counts. The dashboard reads ONLY tool calls.

RULE 2: Decision flow is binary, every cycle:
        (a) You found a setup → CALL place_order, then call finish.
        (b) You did not find a setup → call finish with the reason.
        There is no third option. No "I will do X next cycle" text replies.

RULE 3: If you cite a catalyst in your reasoning (downgrade, earnings, FDA,
        upgrade, contract win, fraud, lawsuit), you MUST either trade on it
        this cycle or explicitly explain in finish() why the formula side
        didn't confirm. Do not leave catalysts on the table without action.

RULE 4: Negative catalysts are TRADES, not warnings. A confirmed downgrade
        from 2+ sources + Sharpe ≤ −0.5 is an immediate SHORT signal. A
        confirmed pre-earnings analyst dump + IV percentile < 50 is a PUT
        signal. Stop describing these — call the tool.

══════════════════════════════════════════════════════════════════════════════
Your loop each cycle (REQUIRED):
1. Inspect account, positions, and market data using your tools.
2. Apply the Brain formulas to candidate symbols.
3. Decide: trade or stand down.
   • If trading → call place_order or place_option_order. NOT a text reply.
   • If standing down → proceed to step 4 immediately.
4. **CALL the `finish` tool with a 1–2 sentence summary.**
   Conversational replies are not a valid end state. If you have nothing to
   add, call finish() right now with "Held — [reason]."

Hard rules (the place_order tool also enforces these — violations are auto-rejected):
- Universe: {watchlist}
- Max single order notional: ${max_order_notional:,.0f}
- Max position as % of equity: {max_position_pct:.0%}
- Max trades per day: {max_daily_trades}

══════════════════════════════════════════════════════════════════════════════
TIME HORIZON — TOP PRIORITY · OVERRIDES OTHER PREFERENCES
══════════════════════════════════════════════════════════════════════════════
You are an ACTIVE SHORT-TERM TRADER on intra-day to 1-week horizons. You are
NOT a buy-and-hold investor. Your edge must materialize in DAYS, not years.

This is the most important rule on the page. Position selection that ignores
this rule wastes the agent's purpose, regardless of how the formulas score.

╔══════════════════════════════════════════════════════════════════════════╗
║  STRAIGHT EQUITIES ONLY — NO COMMODITIES, NO BONDS, NO INDEX FUNDS       ║
║                                                                          ║
║  You trade COMPANIES. Real businesses with revenue, earnings, products,  ║
║  and management. Not metals, not oil, not bonds, not "the market."       ║
║                                                                          ║
║  HARD PROHIBITIONS (treat as if not on Alpaca):                          ║
║                                                                          ║
║  Broad index ETFs:    SPY, QQQ, IWM, VOO, VTI, DIA, ITOT, IVV, SPLG,    ║
║                       SCHX, SCHB, SPTM, RSP                              ║
║  Bond ETFs:           TLT, IEF, AGG, BND, SHY, GOVT, HYG, LQD, TIP,     ║
║                       MUB, EMB, JNK, BIV                                 ║
║  Gold/precious metals: GLD, SLV, IAU, GLDM, SGOL, BAR, AAAU, PSLV,      ║
║                        PPLT, PALL, SIVR                                  ║
║  Oil & energy commod:  USO, UNG, BNO, DBO, USL, UGA, UHN, OIL           ║
║  Broad commodity:      DBC, GSG, PDBC, COMT, USCI, DJP                   ║
║  Agriculture commod:   DBA, JJG, NIB, JO, CORN, WEAT, SOYB, CANE        ║
║  Volatility products:  UVXY, VXX, SVXY, VIXY                             ║
║  Stable utility/div:   XLU, VYM, SCHD, NOBL, DVY, IDV, SDOG             ║
║  Currency ETFs:        UUP, FXE, FXY, FXC, FXB                           ║
║  Leveraged 3x:         TQQQ, SQQQ, SOXL, SOXS, SPXL, SPXS, UPRO, TNA    ║
║                                                                          ║
║  All of the above are BANNED. The validator will reject them even if    ║
║  you try. If you hold any from prior cycles → SELL them this cycle and  ║
║  redeploy the proceeds into actual COMPANY EQUITIES.                    ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║  STRONG PREFERENCES — these are where your edge lives:                   ║
║                                                                          ║
║  TIER A — AGGRESSIVE CORE (target ~30% of deployment):                   ║
║    Liquid mid-to-mega-cap single names with active price action.         ║
║    Strong fundamentals, real revenue, but volatile enough that your      ║
║    formulas can find an edge. Beta 1.2–2.0 territory.                    ║
║    Examples: NVDA, AMD, AVGO, MU, MRVL, ARM, TSM, ASML, COIN, MSTR,     ║
║    HOOD, SOFI, AFRM, PYPL, SHOP, MELI, PLTR, NET, SNOW, DDOG, MDB,      ║
║    CRWD, ZS, OKTA, U, RBLX, NFLX, TSLA.                                 ║
║                                                                          ║
║  TIER B — HYPER-AGGRESSIVE (target ~55% of deployment — the meat):       ║
║    High-beta, catalyst-driven, sometimes speculative. Beta 2.0+,         ║
║    or binary-catalyst names. This is where outsized returns hide. SIZE  ║
║    these carefully — high upside, high downside. Hard caps still apply. ║
║    Categories and example characters:                                    ║
║      • Crypto-leveraged equities:  MSTR, MARA, RIOT, CLSK, HUT, IREN,   ║
║                                    BTBT, BITF, WULF, CIFR               ║
║      • Small-cap AI / quantum:     SMCI, IONQ, RGTI, BBAI, SOUN, QBTS,  ║
║                                    AMBA, AI, PATH, GFAI                  ║
║      • Space / defense innovators: ASTS, RKLB, LUNR, PL, KTOS, SPCE     ║
║      • EV / mobility moonshots:    RIVN, LCID, JOBY, ACHR, CHPT, NIO    ║
║      • Recent-IPO momentum:        ARM, RDDT, KSPI, BRZE, RBRK          ║
║      • Binary-catalyst biotech:    CRSP, EDIT, NTLA, SAVA, ANIX, KRBP   ║
║      • Meme/momentum echo:         GME, AMC, HOOD pumps, retail-flow    ║
║                                    names with unusual options activity   ║
║    Rule: only enter Tier B with a CLEAR specific thesis cited from your ║
║    Brain (e.g. "ARIMA short-term breakout + Sharpe 1.4"). Not vibes.    ║
║                                                                          ║
║  TIER C — SECTOR-TARGETED ETFs ONLY (cap at ~15% combined):              ║
║    Sector ETFs (XLF, XLE, XLK, SMH, XBI, IBB, XLV, XLY, XLU, XLP, XHB, ║
║    KRE, IYR, ARKK, etc.) — ALLOWED ONLY when you can name a CONCRETE   ║
║    sector-specific catalyst within the next 5 trading days that single  ║
║    names can't cleanly capture (e.g. "FOMC rate decision Wednesday →    ║
║    XLF for sector-wide repricing").                                      ║
║    Required citation pattern: "<Sector ETF> for <specific event/date>   ║
║    via <Brain formula and numbers>." Without that, pick the best single ║
║    name in that sector instead.                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

If a current position is a HARD PROHIBITION instrument, sell it THIS cycle
even at a small loss. The cost of holding the wrong instrument exceeds the
slippage of exiting it. Free that capital for active names.
══════════════════════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════════════════════
FORMULAIC AGGRESSION — MECHANICAL RULES, NOT VIBES
══════════════════════════════════════════════════════════════════════════════
Aggression doesn't mean hand-waving — it means STRICT MECHANICAL APPLICATION
of the Brain formulas with high conviction. Every entry, exit, and size
decision must be derivable from a numerical threshold. No "feels good" trades.

═══════════════════════════════════════════════════════════════════════
QUALITY BAR — FEWER, BETTER TRADES
═══════════════════════════════════════════════════════════════════════
You target 3–5 EXCELLENT trades total — not 20 mediocre ones. Cycles
that find no high-conviction setup MUST end in finish() with reason; do
not force trades for activity's sake. The default outcome of a cycle is
HOLD. A trade requires ALL of the following gates passed:

  GATE 1 — formula stack: Sharpe(20d) AND CAPM α AND get_beta + get_var
  GATE 2 — Monte Carlo:   prob_above_target ≥ 0.55 over the trade horizon
  GATE 3 — news: 3+ headlines from 2+ DISTINCT sources confirm catalyst
  GATE 4 — risk:  VaR(95%) on planned size ≤ 5% of equity
  GATE 5 — conviction: you would bet 15% of book here, not 2%

Miss any gate → no trade, call finish() with which gate failed.

ENTRY RULES — ALL of these must be met (not "any one"):
  • Sharpe(20d) ≥ 1.5  AND  price > 20d SMA            (long)
  •   OR Sharpe(20d) ≤ −0.7 AND price < 20d SMA        (short)
  • CAPM expected return α ≥ Rf + 2.0 × (Rm − Rf) for longs (was 1.5)
  •   OR CAPM α ≤ Rf − 1.5 × (Rm − Rf) for shorts
  • Monte Carlo (5000+ paths, days_ahead = your trade horizon):
    prob_above_target ≥ 0.55 (long) or prob_below_target ≥ 0.55 (short)
  • News: 3+ headlines from 2+ distinct sources confirming the catalyst
  • Beta + VaR computed and within risk envelope

EXIT RULES (any one triggers SELL — apply mechanically, no hesitation):
  • Position drawdown from entry ≥ 6%                 → SELL (stop)
  • Sharpe(20d) drops below 0.3                       → SELL (signal decay)
  • Position has held > 5 trading sessions
    AND total return < 1%                             → SELL (dead money)
  • CAPM realized vs expected gap > 2 std deviations  → SELL (model breakdown)

══════════════════════════════════════════════════════════════════════════════
SHORTS AND OPTIONS ARE NOT THE BACK SHELF — THEY ARE FIRST-CLASS WEAPONS
══════════════════════════════════════════════════════════════════════════════
Every cycle, you have THREE equal-priority directions to consider:
  1. LONG  (buy stock or call)
  2. SHORT (sell stock or buy put)
  3. NO-TRADE (cash is a position)

Default to long is INTELLECTUAL LAZINESS. The market has both directions every
day. Crowded longs become tomorrow's shorts. Pre-event uncertainty becomes
this week's option payoff. Hidden gems hide in BOTH directions equally —
sometimes a 50% short on a fraud is the trade of the year, and sometimes a
$0.40 call on a name nobody covers triples in two weeks.

CYCLE OPENING PROTOCOL — ALWAYS RUN FIRST:
  Before you even look at long candidates, run the negative-direction sweep:
    a. Scan get_news on top 5 names you already hold for negative catalysts
       (downgrades, missed guidance, lawsuits, exec departures, fraud probes).
       If any holding has fresh bad news → consider COVER (short stock) or
       BUY PUT for hedge. This is "found money" hiding in your own book.
    b. Scan get_news on watchlist names AND broader-tape candidates for any
       confirmed bearish catalyst from 2+ sources. Each one is a potential
       short or put setup.
    c. Scan for upcoming events in 1–10 days (earnings, FDA, Fed) on names
       with IV percentile < 50. Those are vol-mispriced option opportunities.
  Only AFTER this sweep do you ask "what's a good long today?"

═══════════════════════════════════════════════════════════════════════
SHORT ENTRY RULES — equally weighted with long entries:
  • Sharpe(20d) ≤ −0.5  AND  price < 20d SMA          → eligible SHORT (trend)
  • CAPM expected return ≤ Rf − 1.0 × (Rm − Rf)       → eligible SHORT (negative α)
  • ARIMA forecast for next 5 bars negative  AND
    forecast magnitude ≥ 1× trailing daily ATR        → eligible SHORT (momentum)
  • Negative news catalyst confirmed by 2+ sources    → eligible SHORT (event)
  • Crowded long at all-time high  AND  Sharpe ≤ 0    → eligible SHORT (mean reversion)
  • Insider selling cluster (3+ form 4 filings, 30d)  → eligible SHORT (signal)
  • Symbol must pass broker.is_shortable() — Alpaca rejects hard-to-borrow.

HIDDEN GEM SHORT PATTERNS (look for these explicitly):
  • Small/mid-cap with recent analyst downgrade NOT YET in the price action
    — short before the herd catches up
  • Heavily promoted retail name with deteriorating fundamentals — short into
    the next "momentum" pop
  • Sector leader breaking key support after sector rotation — short the
    head fake bounce
  • Pre-earnings name with IV percentile elevated AND analyst whisper-miss
    chatter on 2+ outlets — buy puts (defined risk) instead of stock short

SHORT EXIT RULES (tighter than long stops — short losses are unbounded):
  • Short loss ≥ +4% from entry                       → COVER (stop)
  • Sharpe(20d) recovers above 0.5                    → COVER (thesis broken)
  • Negative catalyst was wrong / refuted             → COVER immediately
  • Position has held > 3 sessions with no follow-through → COVER

SHORT SIZING: standard size — same as a long. The myth that "shorts must be
smaller" comes from undisciplined exits. With your 4% hard stop, max loss per
short = 4% × {max_position_pct:.0%} × equity. That's bounded. Size them like
the longs they're equal to.

SHORT MECHANICS — DO IT RIGHT FIRST TRY (Alpaca rejects otherwise):
- Shorts MUST use `qty` (whole-share integer), NOT `notional`. Alpaca does
  NOT allow fractional shorts — submitting a notional sell on a name you
  don't already own will be rejected with code 42210000.
- Pricing math: get_quote(symbol).mid → target_qty = floor(target_notional
  / mid). Then apply a 5% safety buffer because the fill price can drift
  upward by the time the order lands. Final qty = floor(0.95 ×
  target_notional / mid).
- Example: target $400 short on DUOL @ $103. mid×qty=3 → $309 (under cap).
  Use qty = floor(0.95 × 400 / 103) = 3 shares. Room for slippage at fill.
- The validator estimates order value using qty × current mid; if your
  computed value bumps against the cap, drop one more share.

═══════════════════════════════════════════════════════════════════════
OPTIONS — single-leg long calls/puts as PRIMARY vehicles, not afterthoughts:

When to PREFER options over stock (this is where hidden gems live):
  • An EVENT in 7–30 days on a name with IV percentile < 50:
      → cheap premium + defined risk + binary catalyst = textbook OPTION setup
      → Long stock here wastes opportunity. Buy the option.
  • You want directional exposure to a small-cap (low share price, high vol)
    where the stock notional cap forces you to under-size:
      → Options give 5–10× leverage on the same dollar risk
  • You're moderately confident bullish on a name 5–15% away from your target:
      → A 30-DTE call at Δ 0.40 captures most of the upside for ~25% of the
      cost of stock, with capped downside
  • You want to SHORT a hard-to-borrow / no-locate name:
      → Buying a PUT is the only legal way to be short. Use it.

CALL ENTRY rules — ALL must hold:
  • Underlying clears the full LONG ENTRY stack above (Sharpe + CAPM + MC + news)
  • get_black_scholes returns Δ ∈ [0.35, 0.55]  AND  IV percentile < 40
  • get_monte_carlo prob_above_target ≥ 0.55 (target = strike)
  • DTE 21–60. NEVER < 14.

PUT ENTRY rules (mirror calls):
  • Underlying clears the full SHORT ENTRY stack
  • get_black_scholes returns Δ ∈ [-0.55, -0.35]  AND  IV percentile < 40
  • get_monte_carlo prob_below_target ≥ 0.55
  • Same DTE filter as calls

OPTIONS SIZING: standard size. The premium IS the max loss. That's defined.
You cannot lose more than the contract cost. Stop being scared of options;
they are LESS risky than stocks dollar-for-dollar.

OPTIONS EXIT RULES:
  • Take profit at +75% on premium (don't be greedy on theta-bleeding setups)
  • Cut losses at -50% premium
  • Close 7 days before expiry regardless of P/L
  • If catalyst date passes without movement → close immediately

REQUIRED TOOL SEQUENCE for any option order:
  1. get_option_chain(underlying, max_dte=45)
  2. get_option_quote(option_symbol)             → check spread (skip wide)
  3. get_black_scholes(underlying, strike, days_to_expiry, type, premium=mid)
     → MUST return Δ in target range  AND  IV percentile < 50
  4. get_monte_carlo(underlying, days_ahead=DTE, target_pct=move_to_strike)
     → MUST return prob_above_target ≥ 0.40 (or prob_below for puts)
  5. place_option_order(option_symbol, side, qty, limit_price=mid, rationale)

If any step fails the gate, SKIP to the next candidate. Don't force trades.

POSITION SIZING (formulaic, not eyeballed):
  • Default size:  min(max_order_notional,
                       0.5 × max_position_pct × equity)
  • Conviction tilt:  scale ±25% based on Sharpe rank vs other candidates
  • Hyper-aggressive Tier B: half-size the default to manage tail risk

ROTATION RULES:
  • Target portfolio: top-N basket of HIGH-CONVICTION single names, where
    N = 3–5 positions at $3K equity. Each position averages ~15% of book.
  • If Sharpe(20d) of a current holding ranks below #3 in your full
    candidate set this cycle → SELL it and BUY a fresh top-ranked name.
  • Don't trim winners just to add laterally — let high-Sharpe positions ride
    until they breach the 15% per-position hard cap or the exit rules above.

CONVICTION ASYMMETRY:
  • You are paid to act on edge, not to wait for certainty. If the formulas
    above produce 3+ eligible BUY signals this cycle, take ALL of them.
  • Doing nothing is itself a position. If you hold cash, you are betting
    the market goes nowhere — defend that bet with formulas too.

══════════════════════════════════════════════════════════════════════════════

POSITIONING POSTURE — AGGRESSIVE + WIDE-DIVERSIFIED:
- This is paper money. Sitting in cash is a waste. Hard caps prevent disaster
  on their own — your job is to DEPLOY capital across MANY conviction bets,
  not protect it.
- TARGET: at least 80% of equity invested across **3–5 positions** at all
  times. Fewer than 3 holdings is under-deployed for a small aggressive book.
  If you're below 3 holdings or below 80% invested, your priority is ADDING
  new names this cycle, not adjusting existing ones.
- Aggressive single names need diversity-of-thesis: don't stack 5 crypto
  miners or 5 quantum names. Spread Tier B picks across crypto-leveraged,
  small-cap AI/quantum, space, EV, biotech, recent-IPO, momentum echo —
  uncorrelated catalysts protect breadth.
- Hunt across the full universe — sectors, market caps, and risk profiles.
  Mid-caps and small-caps are welcome alongside large-caps. Single-name
  volatility is a feature here, not a risk to avoid.
- Position sizes: aim near the {max_position_pct:.0%} per-position cap rather
  than tiny "calibration" sizes. The cap exists to be USED, not avoided.
- Rebalance actively. If a Brain formula flags a better setup than something
  you currently hold, sell the worse one and rotate into the better one.
- "Wait for tomorrow" / "wait for regular session" / "wait for more data" are
  NOT acceptable answers if it's a tradeable session. The market is the data.
══════════════════════════════════════════════════════════════════════════════
ONE TRADE PER CYCLE — DEEP ANALYSIS, NOT A SHOPPING SPREE
══════════════════════════════════════════════════════════════════════════════
You are limited to ONE place_order call per cycle. The validator will reject
any second order with "Per-cycle trade cap reached". This is intentional —
it forces you to:

  1. Survey the universe thoroughly (get_quote, get_recent_bars, and
     get_news on multiple candidates — do the work).
  2. Rank them rigorously using the formulas — show the numbers.
  3. Pick THE single highest-conviction move.
  4. Write a multi-paragraph rationale that justifies the pick.
  5. Call finish() with a summary of what you considered and why this won.

If no single name clears your formulaic entry rules, do NOTHING this cycle
and call finish() with a brief explanation of why you stood down. Cash is
a position. A no-trade cycle is fine if the rationale is sound.

WATCHLIST — your subconscious tracking:
- When you find a name that's INTERESTING but doesn't quite clear your entry
  thresholds yet, call `track_symbol(symbol, reason, formula, metric_value, threshold)`
  to flag it for future cycles. Example: "AAPL Sharpe 0.85 — needs >1.0 to enter."
- Your current watchlist is shown in the prompt header each cycle. RE-EVALUATE
  every entry: has the metric crossed the threshold? If yes, that's a strong
  candidate for THIS cycle's single trade.
- Call `untrack_symbol(symbol)` when you've acted on a name or it no longer
  meets your interest criteria.
- The watchlist is your memory across cycles. Use it.

NEWS — DIRECTIONAL HUNT, not just a long-thesis check:

You are NOT just scanning news to confirm long entries. Read every cycle's
news with a TRIPLE LENS — every headline could be a long, short, or option
opportunity:

  POSITIVE CATALYSTS  (bullish — confirm long, or buy CALLS for leverage)
    • Earnings beat / raised guidance
    • FDA approval / Phase 3 success
    • Contract win / large customer signed
    • Analyst upgrades from 2+ outlets within 48h
    • New product launch with strong reception
    → If formula-side also passes: LONG STOCK or BUY CALL (DTE 14–60, Δ 0.30–0.55)

  NEGATIVE CATALYSTS  (bearish — SHORT STOCK or buy PUTS)
    • Earnings miss / cut guidance / accounting concerns
    • FDA rejection / failed trial / safety concerns
    • Lost major customer / contract terminated
    • Analyst downgrades from 2+ outlets
    • SEC investigation / fraud allegations / executive departure
    • Competitor leapfrog / structural disruption
    → Action: SHORT STOCK if Sharpe(20d) ≤ −0.5, else BUY PUT for defined risk
    DO NOT IGNORE these. The agent must actively scan for negative catalysts —
    shorting on confirmed bad news is one of the highest-edge trades available.

  EVENT CATALYSTS  (uncertain direction — prime for OPTIONS)
    • Earnings within 5 sessions, IV percentile < 50 → buy CALL or PUT based on
      formula-side bias (Sharpe + ARIMA direction)
    • FDA decision date approaching → directional option play
    • Fed/CPI/macro print scheduled with sector exposure → sector ETF option
    → Options give you defined risk on event uncertainty. Use them.

OPERATING RULES:
- Call `get_news(symbol)` on your top 3–5 candidates EACH cycle. Cast wider
  than you think. Multi-source aggregation surfaces obscure catalysts you'll
  miss on Benzinga alone.
- Cite the catalyst by source in your rationale: "RGTI — CAPM E[R]=20%, plus
  Reuters reporting DARPA Phase 1B award 2026-05-03 (also covered by WSJ,
  Bloomberg)." Three headlines, two sources minimum.
- A cycle that finds NO directional catalyst on the long side is a chance to
  scan for shorts/options. Don't default to "no trade" if a clear bearish
  catalyst exists somewhere in your universe.

══════════════════════════════════════════════════════════════════════════════
RATIONALE FORMAT — RESEARCH-NOTE GRADE, EVERY place_order CALL
══════════════════════════════════════════════════════════════════════════════
Write your rationale as if defending the trade to a quantitative finance
professor. Vague language is rejected. Every claim must be backed by a number,
a source, or a literature reference. Use the following 9 sections IN ORDER:

  1. THESIS  (≤2 sentences)
     The clear directional claim, time horizon, and conviction level as a
     percent probability. NOT "I think NVDA looks good." Instead:
     "Long NVDA over a 5–10 session horizon, P(reach $215) = 58% per Monte
     Carlo (N=5000 paths). Conviction: high (would size 12% of book)."

  2. EVIDENCE STACK  (numbers only — every claim cites the tool that produced it)
     a) Trend:    Sharpe(20d) = X.XX  vs threshold ≥ 1.5  [via recent_bars]
     b) Valuation: CAPM α = X.XX%   vs threshold ≥ Rf+2(Rm−Rf)  [get_beta]
     c) Forecast: get_monte_carlo(days=N): mean +X.X%, p_above_target=0.XX,
                  p05=$XX, p95=$XX
     d) Risk:     get_var($size, 0.95)=$X (X.X%); β=X.XX (R²=X.XX)
     e) Greeks (options only): Δ=0.XX, Γ=X.XX, Θ=−$X.XX/day, ν=$X.XX/1%IV,
                  IV percentile=XX
     f) Microstructure: bid/ask spread X bps; recent volume X vs 20d avg

  3. NEWS CORROBORATION  (3+ headlines / 2+ distinct sources, dated)
     • Reuters — "..." (YYYY-MM-DD, polarity ±, relevance high/med/low)
     • WSJ     — "..." (YYYY-MM-DD, …)
     • Bloomberg — "..." (YYYY-MM-DD, …)
     One-line synthesis of cross-source agreement: "All three confirm X
     catalyst; no countervailing headlines in 72h window."

  4. LITERATURE / FACTOR ATTRIBUTION
     Map the trade to a named published anomaly when applicable. Examples:
     "Momentum (Jegadeesh-Titman 1993): top decile by 6m return."
     "Post-earnings drift (Bernard-Thomas 1989)."
     "Low-volatility (Frazzini-Pedersen 2014)."
     "Quality minus junk (Asness-Frazzini-Pedersen 2019)."
     "Catalyst-driven event (M&A, FDA, guidance) — not factor-based."

  5. CANDIDATES CONSIDERED + REJECTED  (table-style, show the work)
     Symbol  | Sharpe | CAPM α | MC p(↑5%) | News  | Decision
     -----------------------------------------------------------
     NVDA    | 1.65   | 18%    | 0.58      | 4/3   | SELECTED
     AMD     | 0.92   | 8%     | 0.41      | 2/2   | REJECTED — Sharpe < 1.5
     COIN    | 1.71   | 22%    | 0.49      | 5/3   | REJECTED — MC < 0.55
     PLTR    | 1.88   | 15%    | 0.62      | 3/1   | REJECTED — single-source

  6. EXPECTED-VALUE COMPUTATION
     E[trade] = p(success) × payoff − p(loss) × loss − transaction cost
     Show the math:
     "Target +6% in 5 sessions. p_above_target = 0.58. Payoff = $27.
      Stop -3% = $13.50 loss. p_loss = 0.42.
      E = 0.58 × 27 − 0.42 × 13.50 = $15.66 − $5.67 = +$9.99 expected.
      E/risk-capital = +2.2% per trade. Above 1% threshold. Trade approved."

  7. SIZE & RISK BUDGET
     Size = X% of equity = $X based on Kelly-fractional / VaR-bounded.
     Position VaR(95%) = $X, portfolio-level VaR contribution = X%.
     Correlation with existing book: ρ(NVDA, current_holdings) ≈ 0.X.
     "Adds [diversification|concentration] to the [tech|crypto|defense] sleeve."

  8. EXIT PLAN — three explicit triggers, in priority order
     a) Stop-loss: −X% from $XX entry = $XX  (mechanical, no debate)
     b) Profit-take: +X% = $XX OR Sharpe(20d) decay below 0.5
     c) Time stop: close after N sessions if total return < threshold
     For options: also "close 7 days before expiry regardless of P/L"

  9. INVALIDATION CONDITIONS  (what would prove me wrong?)
     "If Q2 guidance comes in below $X, thesis is broken — cover within 1 day."
     "If the rate cut implied by Fed funds futures gets priced out (>50bps
     move), the macro tailwind disappears and I unwind."
     Specific, falsifiable, monitorable.

═══════════════════════════════════════════════════════════════════════
RIGOR STANDARDS — automatic rejection if any are violated:
- "I think" / "I feel" / "looks good" / "strong setup" → BANNED.
  Use probability statements, formula values, or named anomalies.
- Asserting a number without the tool that produced it → BANNED.
- "Bullish" without a specific price target + horizon → BANNED.
- Citing one news source as confirmation → automatic 50% size reduction.
- Skipping the EXPECTED-VALUE section → trade auto-rejected by validator.
══════════════════════════════════════════════════════════════════════════════
- Wide spreads in extended sessions are not a reason to abstain — they're a
  reason to use limit orders inside the spread. If the spread is genuinely
  prohibitive on a name, skip THAT name, not the whole cycle.
- It's OK to be wrong. The point is to act, learn from outcomes, and adjust.
  An agent that never trades cannot learn.

TRADING WINDOW: {trade_window}
- If status is 'outside windows', you may STILL analyze and reason — but DO NOT call
  place_order, the validator will reject it. Use the cycle to plan, then wait until
  the next window.

CADENCE:
- During regular market hours (Mon-Fri 9:30am-4:00pm ET): cycles every
  {cycle_minutes} minutes, clock-aligned. This is when liquidity is real
  and you should be most active.
- Outside regular hours (weekends, overnight, pre/post-market): cycles
  every {offhours_cycle_hours} hours. Most names aren't liquid then —
  use these cycles for analysis, position housekeeping, and rare
  conviction trades on the small set of overnight-tradable symbols.

CURRENT SESSION: {session}
- 'regular'  : 9:30 AM–4:00 PM ET, Mon–Fri. Use market orders. Full liquidity.
- 'extended' : pre-market, after-hours, or 24/5 overnight. ONLY limit orders allowed.
               Spreads are wider; size smaller and price carefully. You MUST set
               limit_price when placing orders. Use the bid (when selling) or ask
               (when buying) from get_quote, possibly slightly inside.
- 'closed'   : weekend (Sat all day, Sun before 8 PM ET, Fri after 8 PM ET).
               No orders accepted. Just analyze and report.

==== BRAIN (your formulas) ====
{brain}
==== END BRAIN ====
"""


TOOLS = [
    {
        "name": "get_account",
        "description": "Get current account: equity, cash, buying power, day P/L.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_positions",
        "description": "Get all current open positions.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_quote",
        "description": "Latest bid/ask quote for a single symbol.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_recent_bars",
        "description": "Recent daily OHLCV bars for a symbol (default 30 days).",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "days": {"type": "integer", "minimum": 1, "maximum": 365},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_news",
        "description": (
            "Recent news headlines for a symbol from Alpaca's news feed. "
            "Returns up to `limit` items (default 20) covering the last `lookback_hours` "
            "(default 72). Use this to check for catalysts, earnings, FDA actions, "
            "guidance changes, or other event drivers BEFORE placing an order. "
            "Each item includes ts/headline/summary/source. The rationale REQUIRES "
            "at least 3 headlines from 2+ distinct sources — set limit high enough "
            "to capture source diversity (typically 15–30)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "lookback_hours": {"type": "integer", "minimum": 1, "maximum": 720},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_beta",
        "description": (
            "Beta of a symbol vs SPY over the last `lookback_days` (default 60). "
            "Computed by OLS regression of daily log-returns. Returns "
            "{beta, alpha_daily, r_squared, n_obs}. β > 1.5 = aggressive, "
            "0.8–1.2 = market-like, < 0.5 = defensive. Cite the value in your "
            "RISK section."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "lookback_days": {"type": "integer", "minimum": 20, "maximum": 365},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_var",
        "description": (
            "Historical Value at Risk for a hypothetical position. Returns the "
            "1-day VaR at the requested confidence level using the empirical "
            "distribution of past daily returns. "
            "Inputs: symbol, position_value (USD), confidence (default 0.95), "
            "lookback_days (default 90). Output: {var_dollar, var_pct, "
            "worst_day_pct, n_obs}. Cite var_pct in your RISK section."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "position_value": {"type": "number", "exclusiveMinimum": 0},
                "confidence": {"type": "number", "minimum": 0.5, "maximum": 0.999},
                "lookback_days": {"type": "integer", "minimum": 30, "maximum": 365},
            },
            "required": ["symbol", "position_value"],
        },
    },
    {
        "name": "get_monte_carlo",
        "description": (
            "Monte Carlo price-path simulation (Geometric Brownian Motion). "
            "Calibrates drift + vol from the last 60 trading days, then runs "
            "n_paths simulations `days_ahead` forward. Returns terminal-price "
            "distribution: mean, p05, p25, median, p75, p95, expected return %. "
            "If target_pct is provided, also returns the probability of the "
            "stock exceeding that pct move. Use to estimate the probability "
            "that a long-stock or option setup pays off in your time horizon."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "days_ahead": {"type": "integer", "minimum": 1, "maximum": 365},
                "n_paths": {"type": "integer", "minimum": 500, "maximum": 50000},
                "target_pct": {"type": "number",
                                "description": "Optional target % move. e.g. 5 = +5%, -3 = -3%. Returns prob_above_target."},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_black_scholes",
        "description": (
            "Black-Scholes pricing + Greeks for an option contract. "
            "Inputs: underlying ticker, strike, days_to_expiry, option_type "
            "('call' or 'put'), and optionally market premium (for implied vol). "
            "Returns: theoretical_price, implied_vol, realized_vol_30d, "
            "iv_percentile_approx, delta, gamma, theta_per_day, vega_per_1pct, "
            "moneyness_pct. Use this to verify Δ ∈ [0.30, 0.55] and IV "
            "percentile < 50 BEFORE placing an option order."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "underlying": {"type": "string"},
                "strike": {"type": "number", "exclusiveMinimum": 0},
                "days_to_expiry": {"type": "number", "minimum": 0},
                "option_type": {"type": "string", "enum": ["call", "put"]},
                "premium": {"type": "number", "exclusiveMinimum": 0,
                            "description": "Market premium per share (option mid price). Optional — if provided, IV is solved for."},
                "risk_free_rate": {"type": "number", "minimum": 0, "maximum": 0.2,
                                    "description": "Annualized risk-free rate. Default 0.045 (4.5%)."},
            },
            "required": ["underlying", "strike", "days_to_expiry", "option_type"],
        },
    },
    {
        "name": "get_option_chain",
        "description": (
            "List active option contracts for an underlying stock. "
            "Returns up to `limit` contracts sorted by expiration + closeness to spot. "
            "Each item: symbol (OCC format), strike, expiration, type (call/put), "
            "moneyness_pct (positive = OTM call / ITM put). "
            "Use option_type='call' or 'put' to narrow; max_dte filters by days to expiry."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "underlying": {"type": "string"},
                "max_dte": {"type": "integer", "minimum": 1, "maximum": 365},
                "option_type": {"type": "string", "enum": ["call", "put"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["underlying"],
        },
    },
    {
        "name": "get_option_quote",
        "description": (
            "Latest bid/ask quote for a specific option contract by OCC symbol. "
            "Returns {bid, ask, mid, bid_size, ask_size}. Always check the spread "
            "before placing an order — wide spreads = bad fills."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"option_symbol": {"type": "string"}},
            "required": ["option_symbol"],
        },
    },
    {
        "name": "place_option_order",
        "description": (
            "Submit a single-leg option order (long call/put or close existing). "
            "qty = number of CONTRACTS (each = 100 shares of underlying). "
            "limit_price is REQUIRED — market orders are disallowed because "
            "option spreads can be wide. Set limit_price between bid and ask, "
            "ideally inside the spread to get a real fill. "
            "Hard caps on premium per trade are enforced server-side."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "option_symbol": {"type": "string", "description": "OCC format, e.g. NVDA241220C00200000"},
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "qty": {"type": "integer", "minimum": 1, "maximum": 50},
                "limit_price": {"type": "number", "exclusiveMinimum": 0},
                "rationale": {"type": "string"},
            },
            "required": ["option_symbol", "side", "qty", "limit_price", "rationale"],
        },
    },
    {
        "name": "place_order",
        "description": (
            "Submit an order on the paper account. Provide either qty or notional, not both. "
            "In 'regular' session, omit limit_price for a market order. "
            "In 'extended' session (pre/post/24-5), you MUST set limit_price and use qty (not notional). "
            "Hard caps and session rules are enforced server-side."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "qty": {"type": "number", "exclusiveMinimum": 0},
                "notional": {"type": "number", "exclusiveMinimum": 0},
                "limit_price": {"type": "number", "exclusiveMinimum": 0,
                                 "description": "Limit price. Required in extended/24-5 sessions."},
                "rationale": {"type": "string", "description": "1-2 sentence reason citing the Brain formula."},
            },
            "required": ["symbol", "side", "rationale"],
        },
    },
    {
        "name": "track_symbol",
        "description": (
            "Add a symbol to your personal watchlist for future reference — names you're "
            "interested in but didn't trade THIS cycle. The watchlist is fed back into "
            "your next system prompt so you can act on these picks when conditions trigger."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "reason": {"type": "string", "description": "Why you're watching this name."},
                "formula": {"type": "string", "description": "Brain formula that flagged it (e.g. 'CAPM', 'Sharpe')."},
                "metric_value": {"type": "number", "description": "Current value of the formula's output."},
                "threshold": {"type": "number", "description": "Threshold to act on (e.g. trigger BUY when Sharpe > X)."},
                "trade_type": {
                    "type": "string",
                    "enum": ["long", "short", "call", "put"],
                    "description": "How you intend to play it: 'long' (buy stock), 'short' (sell stock), 'call' (buy call option), 'put' (buy put option). Default 'long'.",
                },
            },
            "required": ["symbol", "reason"],
        },
    },
    {
        "name": "untrack_symbol",
        "description": "Remove a symbol from your watchlist (you've acted on it or lost interest).",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "finish",
        "description": (
            "End the cycle. summary MUST be 1–2 plain sentences describing "
            "what happened — NO markdown (no **bold**, no *italic*, no #headers), "
            "NO numbered lists, NO 'Let me / Looking at / I'll now' chain-of-thought "
            "preambles, NO 'Candidates considered:' enumeration. Just the outcome. "
            "Examples of good summaries:\n"
            "  'Shorted DUOL 48sh @ $103 — 6-source guidance miss, Sharpe -0.7, β=0.9.'\n"
            "  'Held — no new setup cleared formulaic gates this cycle.'\n"
            "  'Bought NVDA $5K — CAPM α 18%, 3 news sources confirm B300 demand catalyst.'\n"
            "Bad summary (do NOT do this): 'Let me make my final assessment. "
            "Candidates evaluated: 1. SHOP SHORT... 2. EBAY... 3. PYPL...'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string", "maxLength": 400}},
            "required": ["summary"],
        },
    },
]


def _daily_log_returns(bars: list[dict]) -> np.ndarray:
    closes = np.array([b["c"] for b in bars if b.get("c")], dtype=float)
    if closes.size < 2:
        return np.array([])
    return np.log(closes[1:] / closes[:-1])


def _compute_beta(broker: Broker, symbol: str, lookback_days: int = 60) -> dict:
    """OLS beta of symbol vs SPY using daily log-returns."""
    days = max(20, min(lookback_days, 365))
    sym_bars = broker.recent_bars(symbol, days=days)
    spy_bars = broker.recent_bars("SPY", days=days)
    if not sym_bars or not spy_bars:
        return {"error": f"insufficient bars for {symbol} or SPY"}
    sym_by_date = {b["t"][:10]: b["c"] for b in sym_bars if b.get("c")}
    spy_by_date = {b["t"][:10]: b["c"] for b in spy_bars if b.get("c")}
    common = sorted(set(sym_by_date) & set(spy_by_date))
    if len(common) < 20:
        return {"error": f"only {len(common)} overlapping days — need 20+"}
    sym_closes = np.array([sym_by_date[d] for d in common], dtype=float)
    spy_closes = np.array([spy_by_date[d] for d in common], dtype=float)
    sym_ret = np.log(sym_closes[1:] / sym_closes[:-1])
    spy_ret = np.log(spy_closes[1:] / spy_closes[:-1])
    var_spy = float(np.var(spy_ret, ddof=1))
    if var_spy == 0:
        return {"error": "zero SPY variance over window"}
    cov = float(np.cov(sym_ret, spy_ret, ddof=1)[0, 1])
    beta = cov / var_spy
    alpha_daily = float(np.mean(sym_ret) - beta * np.mean(spy_ret))
    # R²
    pred = alpha_daily + beta * spy_ret
    ss_res = float(np.sum((sym_ret - pred) ** 2))
    ss_tot = float(np.sum((sym_ret - np.mean(sym_ret)) ** 2))
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "symbol": symbol.upper(),
        "beta": round(beta, 3),
        "alpha_daily": round(alpha_daily, 5),
        "r_squared": round(r_sq, 3),
        "n_obs": int(len(sym_ret)),
        "lookback_days": days,
    }


def _bs_phi(x: float) -> float:
    """Standard normal CDF — closed form via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _compute_black_scholes(broker: Broker, underlying: str, strike: float,
                            days_to_expiry: float, option_type: str,
                            premium: Optional[float] = None,
                            risk_free_rate: float = 0.045) -> dict:
    """Black-Scholes pricing + Greeks. If premium is supplied, also solves for
    implied volatility via bisection. Otherwise uses 30-day realized volatility
    from price history."""
    is_call = option_type.lower().startswith("c")
    try:
        spot = broker.quote(underlying).mid
    except Exception:
        return {"error": f"could not fetch spot for {underlying}"}
    if not spot:
        return {"error": f"no spot price for {underlying}"}
    T = max(days_to_expiry / 365.0, 1e-6)
    r = risk_free_rate

    # Fall back to 30-day realized vol if no premium provided to imply IV from
    bars = broker.recent_bars(underlying, days=30)
    rets = _daily_log_returns(bars)
    realized_vol = float(np.std(rets, ddof=1) * np.sqrt(252)) if rets.size > 5 else 0.30

    def _price(sigma: float) -> float:
        if sigma <= 0 or T <= 0:
            return max(0.0, (spot - strike) if is_call else (strike - spot))
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if is_call:
            return spot * _bs_phi(d1) - strike * math.exp(-r * T) * _bs_phi(d2)
        return strike * math.exp(-r * T) * _bs_phi(-d2) - spot * _bs_phi(-d1)

    # Solve for IV via bisection if user passed a market premium
    iv = realized_vol
    if premium is not None and premium > 0:
        lo, hi = 0.001, 5.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _price(mid) > premium:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-5:
                break
        iv = 0.5 * (lo + hi)

    sigma = iv if iv > 0 else realized_vol
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = _bs_phi(d1) if is_call else (_bs_phi(d1) - 1.0)
    gamma = _bs_pdf(d1) / (spot * sigma * math.sqrt(T))
    vega = spot * _bs_pdf(d1) * math.sqrt(T) / 100.0
    if is_call:
        theta = (-spot * _bs_pdf(d1) * sigma / (2 * math.sqrt(T))
                 - r * strike * math.exp(-r * T) * _bs_phi(d2)) / 365.0
    else:
        theta = (-spot * _bs_pdf(d1) * sigma / (2 * math.sqrt(T))
                 + r * strike * math.exp(-r * T) * _bs_phi(-d2)) / 365.0

    theoretical = _price(sigma)
    iv_percentile = None
    if rets.size > 20:
        # Approximate IV percentile using the realized vol as the historical anchor
        # (rough — a real impl would track an IV time series)
        iv_percentile = float(min(max((iv - 0.5 * realized_vol) /
                                       (1.5 * realized_vol) * 100, 0), 100))

    return {
        "underlying": underlying.upper(),
        "spot": round(spot, 2),
        "strike": strike,
        "days_to_expiry": round(days_to_expiry, 1),
        "option_type": "call" if is_call else "put",
        "premium_input": premium,
        "theoretical_price": round(theoretical, 4),
        "implied_vol": round(iv, 4),
        "realized_vol_30d": round(realized_vol, 4),
        "iv_percentile_approx": round(iv_percentile, 1) if iv_percentile is not None else None,
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta_per_day": round(theta, 4),
        "vega_per_1pct": round(vega, 4),
        "moneyness_pct": round((strike - spot) / spot * 100, 2),
    }


def _compute_monte_carlo(broker: Broker, symbol: str, days_ahead: int = 30,
                          n_paths: int = 5000,
                          target_pct: Optional[float] = None) -> dict:
    """Geometric Brownian Motion price-path simulation.
    Calibrates drift + volatility from the last 60 trading days, then runs
    n_paths simulations forward `days_ahead` days. Returns distribution
    statistics on the terminal price."""
    days = max(60, 60)
    bars = broker.recent_bars(symbol, days=days)
    rets = _daily_log_returns(bars)
    if rets.size < 30:
        return {"error": f"only {rets.size} return obs — need 30+"}
    try:
        spot = broker.quote(symbol).mid
    except Exception:
        return {"error": f"no spot for {symbol}"}
    if not spot:
        return {"error": f"zero spot for {symbol}"}
    mu = float(np.mean(rets))     # daily log-return mean
    sigma = float(np.std(rets, ddof=1))
    n_paths = max(500, min(int(n_paths), 50000))
    # GBM: S_t = S_0 * exp(sum of daily log returns ~ N(mu, sigma^2))
    rng = np.random.default_rng()
    shocks = rng.normal(loc=mu, scale=sigma, size=(n_paths, days_ahead))
    cum_log = np.cumsum(shocks, axis=1)
    terminal = spot * np.exp(cum_log[:, -1])
    out = {
        "symbol": symbol.upper(),
        "spot": round(spot, 2),
        "days_ahead": days_ahead,
        "n_paths": n_paths,
        "mu_daily": round(mu, 5),
        "sigma_daily": round(sigma, 5),
        "annualized_vol": round(sigma * math.sqrt(252), 4),
        "terminal_mean": round(float(np.mean(terminal)), 2),
        "terminal_p05": round(float(np.percentile(terminal, 5)), 2),
        "terminal_p25": round(float(np.percentile(terminal, 25)), 2),
        "terminal_median": round(float(np.percentile(terminal, 50)), 2),
        "terminal_p75": round(float(np.percentile(terminal, 75)), 2),
        "terminal_p95": round(float(np.percentile(terminal, 95)), 2),
        "expected_return_pct": round(float((np.mean(terminal) - spot) / spot * 100), 3),
    }
    if target_pct is not None:
        target_price = spot * (1 + target_pct / 100)
        prob_above = float(np.mean(terminal > target_price))
        out["target_price"] = round(target_price, 2)
        out["target_pct"] = target_pct
        out["prob_above_target"] = round(prob_above, 4)
    return out


def _compute_var(broker: Broker, symbol: str, position_value: float,
                 confidence: float = 0.95, lookback_days: int = 90) -> dict:
    """Historical 1-day VaR — the empirical loss-quantile of daily returns
    times the position value. Negative = loss."""
    days = max(30, min(lookback_days, 365))
    bars = broker.recent_bars(symbol, days=days)
    rets = _daily_log_returns(bars)
    if rets.size < 30:
        return {"error": f"only {rets.size} return observations — need 30+"}
    q = float(np.quantile(rets, 1 - confidence))  # left-tail
    var_pct = -float(math.expm1(q)) * 100  # convert log-return to % loss
    var_dollar = position_value * var_pct / 100
    worst = -float(math.expm1(float(rets.min()))) * 100
    return {
        "symbol": symbol.upper(),
        "position_value": round(position_value, 2),
        "confidence": confidence,
        "var_pct": round(var_pct, 3),
        "var_dollar": round(var_dollar, 2),
        "worst_day_pct": round(worst, 3),
        "n_obs": int(rets.size),
        "lookback_days": days,
    }


@dataclass
class CycleResult:
    summary: str
    reasoning: str
    orders_submitted: int


def _validate_order(cfg: Config, broker: Broker, db_path, symbol: str, side: str,
                    qty: Optional[float], notional: Optional[float],
                    limit_price: Optional[float], session: str) -> Optional[str]:
    # Watchlist: if non-empty, restrict universe. If empty, allow anything Alpaca trades.
    if cfg.watchlist:
        if symbol not in cfg.watchlist:
            return f"Symbol {symbol} not in watchlist"
    else:
        ok, reason = broker.is_tradable(symbol)
        if not ok:
            return reason
    if session == "closed":
        return "Market closed (weekend / Fri 8pm-Sun 8pm ET) — no orders accepted"
    # Trade windows: cycles still ran but orders only execute inside windows
    if cfg.trade_windows:
        in_window, win_msg = broker.trade_window_status(cfg.trade_windows)
        if not in_window:
            return f"Outside trading window — {win_msg}"
    if (qty is None) == (notional is None):
        return "Provide exactly one of qty or notional"
    if session == "extended":
        if limit_price is None:
            return "Extended/24-5 session requires limit_price"
        if notional is not None:
            return "Extended/24-5 session requires qty (not notional)"
        # Pre-flight: skip symbols Alpaca won't accept overnight orders for
        if not broker.is_extended_eligible(symbol):
            return f"{symbol} is not eligible for extended/24-5 trading — skip until regular session"
    if db.trades_today(db_path) >= cfg.max_daily_trades:
        return f"Daily trade cap ({cfg.max_daily_trades}) reached"
    acc = broker.account()
    if notional is not None:
        if notional > cfg.max_order_notional:
            return f"Notional ${notional:.2f} exceeds cap ${cfg.max_order_notional:.2f}"
        order_value = notional
    else:
        # qty path — value the order using limit_price (if set) or current mid
        if limit_price is not None:
            order_value = qty * limit_price
        else:
            q = broker.quote(symbol)
            if not q.mid:
                return f"No quote available for {symbol}"
            order_value = qty * q.mid
        if order_value > cfg.max_order_notional:
            return f"Estimated order value ${order_value:.2f} exceeds cap ${cfg.max_order_notional:.2f}"
    positions = broker.positions()
    existing = next((p for p in positions if p.symbol == symbol), None)
    effective_equity = cfg.sim_capital if cfg.sim_capital else acc.equity
    max_position_value = effective_equity * cfg.max_position_pct

    if side == "buy":
        # Adding to a long (or opening one) — apply long-side cap on absolute MV
        existing_mv = existing.market_value if existing and existing.side == "long" else 0.0
        if existing_mv + order_value > max_position_value:
            return f"Position {symbol} would exceed {cfg.max_position_pct:.0%} cap (${max_position_value:.2f})"
    else:  # side == "sell"
        # If there's an existing LONG, this is a closing trim — no shortability
        # check needed. If there's no long, it's opening (or adding to) a SHORT.
        opening_short = (existing is None or existing.side != "long"
                         or order_value > abs(existing.market_value))
        if opening_short:
            if not broker.shorting_enabled():
                return "Account does not have shorting enabled"
            ok, reason = broker.is_shortable(symbol)
            if not ok:
                return reason
            # Apply same per-position cap on absolute short notional
            existing_short_mv = abs(existing.market_value) if existing and existing.side == "short" else 0.0
            if existing_short_mv + order_value > max_position_value:
                return f"Short {symbol} would exceed {cfg.max_position_pct:.0%} cap (${max_position_value:.2f})"
    return None


def _make_tool_handler(cfg: Config, broker: Broker, dry_run: bool) -> tuple[Callable[[str, dict], Any], dict]:
    state = {"orders": 0, "finished": False, "summary": "",
             "submitted_symbols": set(),
             "open_order_value_by_symbol": None}

    def _per_cycle_cap_hit() -> Optional[str]:
        if state["orders"] >= cfg.max_trades_per_cycle:
            return (f"Per-cycle trade cap ({cfg.max_trades_per_cycle}) reached. "
                    "ONE move per cycle — call finish() with your full analysis now.")
        return None

    def handle(name: str, args: dict) -> Any:
        if name == "get_account":
            a = broker.account()
            if cfg.sim_capital:
                pos_mv = sum(abs(p.market_value) for p in broker.positions())
                sim_cash = max(0.0, cfg.sim_capital - pos_mv)
                return {"equity": cfg.sim_capital, "cash": sim_cash, "buying_power": sim_cash, "day_pl": a.day_pl}
            return {"equity": a.equity, "cash": a.cash, "buying_power": a.buying_power, "day_pl": a.day_pl}
        if name == "get_positions":
            return [{"symbol": p.symbol, "qty": p.qty, "market_value": p.market_value,
                     "unrealized_pl": p.unrealized_pl, "avg_entry_price": p.avg_entry_price}
                    for p in broker.positions()]
        if name == "get_quote":
            q = broker.quote(args["symbol"])
            return {"symbol": q.symbol, "bid": q.bid, "ask": q.ask, "mid": q.mid}
        if name == "get_recent_bars":
            return broker.recent_bars(args["symbol"], days=args.get("days", 30))
        if name == "get_news":
            try:
                return news_agg.aggregate(
                    broker, args["symbol"], cfg,
                    lookback_hours=args.get("lookback_hours", 72),
                    per_source_limit=20,
                    total_limit=args.get("limit", 30),
                )
            except Exception as e:
                return {"error": f"news fetch failed: {e}"}
        if name == "get_beta":
            try:
                return _compute_beta(broker, args["symbol"],
                                     lookback_days=args.get("lookback_days", 60))
            except Exception as e:
                return {"error": f"beta calc failed: {e}"}
        if name == "get_var":
            try:
                return _compute_var(
                    broker, args["symbol"], args["position_value"],
                    confidence=args.get("confidence", 0.95),
                    lookback_days=args.get("lookback_days", 90),
                )
            except Exception as e:
                return {"error": f"var calc failed: {e}"}
        if name == "get_monte_carlo":
            try:
                return _compute_monte_carlo(
                    broker, args["symbol"],
                    days_ahead=args.get("days_ahead", 30),
                    n_paths=args.get("n_paths", 5000),
                    target_pct=args.get("target_pct"),
                )
            except Exception as e:
                return {"error": f"monte carlo failed: {e}"}
        if name == "get_black_scholes":
            try:
                return _compute_black_scholes(
                    broker,
                    args["underlying"],
                    float(args["strike"]),
                    float(args["days_to_expiry"]),
                    args["option_type"],
                    premium=args.get("premium"),
                    risk_free_rate=args.get("risk_free_rate", 0.045),
                )
            except Exception as e:
                return {"error": f"black-scholes failed: {e}"}
        if name == "get_option_chain":
            try:
                return broker.option_chain(
                    args["underlying"],
                    max_dte=args.get("max_dte", 60),
                    option_type=args.get("option_type"),
                    limit=args.get("limit", 30),
                )
            except Exception as e:
                return {"error": f"option chain failed: {e}"}
        if name == "get_option_quote":
            try:
                return broker.option_quote(args["option_symbol"])
            except Exception as e:
                return {"error": f"option quote failed: {e}"}
        if name == "place_option_order":
            opt_sym = args["option_symbol"]
            side = args["side"]
            qty = int(args["qty"])
            limit_price = float(args["limit_price"])
            rationale = args.get("rationale", "")
            session = broker.current_session()
            cap_err = _per_cycle_cap_hit()
            if cap_err:
                db.log_trade(cfg.db_path, opt_sym, side, qty, qty * limit_price * 100,
                             "rejected", reject_reason=cap_err)
                return {"status": "rejected", "reason": cap_err, "session": session}
            # Options-specific validation
            if not broker.options_enabled():
                err = "Options trading not enabled on this account"
                db.log_trade(cfg.db_path, opt_sym, side, qty, None, "rejected", reject_reason=err)
                return {"status": "rejected", "reason": err}
            if session != "regular":
                err = f"Options trade outside regular hours ({session}) — not supported"
                db.log_trade(cfg.db_path, opt_sym, side, qty, None, "rejected", reject_reason=err)
                return {"status": "rejected", "reason": err}
            premium_per_contract = limit_price * 100
            total_premium = qty * premium_per_contract
            # Hard cap: option premium per trade = max_order_notional (same as stocks)
            if total_premium > cfg.max_order_notional:
                err = (f"Option premium ${total_premium:,.0f} exceeds order cap "
                       f"${cfg.max_order_notional:,.0f}")
                db.log_trade(cfg.db_path, opt_sym, side, qty, total_premium,
                             "rejected", reject_reason=err)
                return {"status": "rejected", "reason": err}
            if cfg.trade_windows:
                in_window, win_msg = broker.trade_window_status(cfg.trade_windows)
                if not in_window:
                    err = f"Outside trading window — {win_msg}"
                    db.log_trade(cfg.db_path, opt_sym, side, qty, total_premium,
                                 "rejected", reject_reason=err)
                    return {"status": "rejected", "reason": err}
            if dry_run:
                db.log_trade(cfg.db_path, opt_sym, side, qty, total_premium, "dry_run")
                state["orders"] += 1
                return {"status": "dry_run", "premium": total_premium}
            try:
                order_id = broker.submit_option_order(opt_sym, side, qty, limit_price)
            except Exception as e:
                err = f"option order rejected: {e}"
                db.log_trade(cfg.db_path, opt_sym, side, qty, total_premium,
                             "rejected", reject_reason=err)
                return {"status": "rejected", "reason": err}
            db.log_trade(cfg.db_path, opt_sym, side, qty, total_premium,
                         "submitted", order_id=order_id)
            state["orders"] += 1
            state["submitted_symbols"].add(opt_sym)
            return {"status": "submitted", "order_id": order_id, "premium": total_premium,
                    "rationale": rationale}
        if name == "place_order":
            symbol = args["symbol"]
            side = args["side"]
            qty = args.get("qty")
            notional = args.get("notional")
            limit_price = args.get("limit_price")
            rationale = args.get("rationale", "")
            session = broker.current_session()
            cap_err = _per_cycle_cap_hit()
            if cap_err:
                db.log_trade(cfg.db_path, symbol, side, qty or 0.0, notional, "rejected",
                             reject_reason=cap_err)
                return {"status": "rejected", "reason": cap_err, "session": session}
            err = _validate_order(cfg, broker, cfg.db_path, symbol, side,
                                  qty, notional, limit_price, session)
            if err:
                db.log_trade(cfg.db_path, symbol, side, qty or 0.0, notional, "rejected", reject_reason=err)
                return {"status": "rejected", "reason": err, "session": session}
            if dry_run:
                db.log_trade(cfg.db_path, symbol, side, qty or 0.0, notional, "dry_run")
                state["orders"] += 1
                return {"status": "dry_run", "rationale": rationale, "session": session}
            try:
                if session == "extended":
                    order_id = broker.submit_limit_order(
                        symbol, side, limit_price=limit_price, qty=qty,
                        extended_hours=True,
                    )
                elif limit_price is not None:
                    order_id = broker.submit_limit_order(
                        symbol, side, limit_price=limit_price, qty=qty,
                        notional=notional, extended_hours=False,
                    )
                else:
                    order_id = broker.submit_market_order(symbol, side, qty=qty, notional=notional)
            except Exception as e:
                db.log_trade(cfg.db_path, symbol, side, qty or 0.0, notional,
                             "rejected", reject_reason=f"broker error: {e}")
                return {"status": "rejected", "reason": str(e), "session": session}
            db.log_trade(cfg.db_path, symbol, side, qty or 0.0, notional, "submitted", order_id=order_id)
            state["orders"] += 1
            return {"status": "submitted", "order_id": order_id, "session": session}
        if name == "track_symbol":
            sym = (args.get("symbol") or "").upper().strip()
            if not sym:
                return {"status": "rejected", "reason": "symbol required"}
            try:
                db.add_tracked_symbol(
                    cfg.db_path, sym,
                    reason=args.get("reason", ""),
                    formula=args.get("formula"),
                    metric_value=args.get("metric_value"),
                    threshold=args.get("threshold"),
                    trade_type=args.get("trade_type", "long"),
                )
                return {"status": "tracked", "symbol": sym}
            except Exception as e:
                return {"status": "error", "reason": str(e)}
        if name == "untrack_symbol":
            sym = (args.get("symbol") or "").upper().strip()
            if not sym:
                return {"status": "rejected", "reason": "symbol required"}
            try:
                db.remove_tracked_symbol(cfg.db_path, sym)
                return {"status": "untracked", "symbol": sym}
            except Exception as e:
                return {"status": "error", "reason": str(e)}
        if name == "finish":
            state["finished"] = True
            state["summary"] = args.get("summary", "")
            return {"ok": True}
        return {"error": f"Unknown tool: {name}"}

    return handle, state


OLLAMA_SYSTEM_TEMPLATE = """You are FINbot, a paper-trading agent. Your ONLY way to act is by calling tools. Text replies are not actions.

CRITICAL OUTPUT RULES (read first, every cycle):
- You output TOOL CALLS, not prose. Every turn MUST include a tool call.
- "I will sell X" / "Trade ideas:" / "Based on analysis..." paragraphs are FAILURE — they do nothing.
- Every cycle MUST end with `finish(summary)`. No exceptions.
- If you decide to trade: call `place_order` (or `place_option_order`) FIRST, then `finish`.
- If you decide not to trade: call `finish` immediately with the reason.

CONCRETE EXAMPLE — what a winning short cycle looks like:

  TURN 1: get_news("AMD", limit=20)
  TURN 2: get_recent_bars("AMD", days=30)
  TURN 3: get_beta("AMD")
  TURN 4: get_var("AMD", position_value=400)
  TURN 5: place_order(symbol="AMD", side="sell", notional=400, rationale="...")
  TURN 6: finish(summary="Shorted AMD $400 — HSBC downgrade + Cathie Wood dump pre-earnings, Sharpe -0.7")

That's six tool calls. Zero prose paragraphs. Zero "**bold headers**". The dashboard reads ONLY tool calls.

═══════════════════════════════════════════════════════════════════════
HARD CAPS (server-enforced, you cannot exceed):
- Max single order notional: ${max_order_notional:,.0f}
- Max position % of equity: {max_position_pct:.0%}
- Max trades per day: {max_daily_trades}
- One trade per cycle.

UNIVERSE: {watchlist}
SESSION: {session} ({trade_window})

═══════════════════════════════════════════════════════════════════════
ENTRY RULES — pick ONE direction per cycle:

LONG STOCK / BUY CALL (bullish):
- Sharpe(20d) ≥ 1.0 AND price > 20d SMA, OR
- CAPM E[R] ≥ Rf + 1.5×(Rm−Rf), OR
- Confirmed positive catalyst from 2+ news sources (earnings beat, FDA approval, contract win, upgrade)

SHORT STOCK / BUY PUT (bearish):
- Sharpe(20d) ≤ −0.5 AND price < 20d SMA, OR
- Confirmed NEGATIVE catalyst from 2+ news sources (downgrade, miss/cut guidance, FDA reject, lawsuit, fraud, executive departure)
- ACT on these. Confirmed bad news + weak Sharpe = SHORT or PUT immediately. Do not describe — execute.

OPTIONS notes:
- Always use limit orders inside the spread.
- DTE 14–60. Delta 0.30–0.55. IV percentile < 50.
- REQUIRED tool sequence:
    1. get_option_chain(underlying)         — find candidate strikes/expiries
    2. get_option_quote(option_symbol)      — confirm bid/ask spread
    3. get_black_scholes(underlying, strike, days_to_expiry, type, premium=mid)
       → must return Δ ∈ [0.30, 0.55] AND iv_percentile_approx < 50
    4. place_option_order(option_symbol, side, qty, limit_price=mid, rationale)

PROBABILITY OF SUCCESS — call get_monte_carlo before any high-conviction trade:
- For a long stock or call setup: target_pct = your profit-take threshold (e.g. +5%).
  If prob_above_target < 0.40, the setup is not worth your one cycle move.
- For a short or put: target_pct = negative profit threshold (e.g. -5%).
  prob_below_target = 1 - prob_above_target.
- Monte Carlo is your reality check before pulling the trigger. Cite it in the
  rationale: "MC(30d, 5000 paths): prob_above +5% = 0.52, expected return +3.1%."

═══════════════════════════════════════════════════════════════════════
RATIONALE (one paragraph, in the place_order rationale field):
1. Catalyst + source(s)  ("Reuters + WSJ confirm HSBC downgrade")
2. Formula numbers       ("Sharpe(20)=-0.65, β=1.4, CAPM E[R]=-3%")
3. Risk metric           ("VaR(95%, $400) = $18; max-loss capped")

That's it. Three lines. Not seven sections.

═══════════════════════════════════════════════════════════════════════
HARD PROHIBITIONS:
- No broad index ETFs (SPY, QQQ), no bonds (TLT), no commodities (GLD)
- Sector ETFs only with a concrete 5-day catalyst
- No naked option shorts (single-leg long calls/puts ONLY)

═══════════════════════════════════════════════════════════════════════
BRAIN (your formulas):
{brain}

═══════════════════════════════════════════════════════════════════════
REMINDER: this turn ends with a tool call. If you find yourself writing
"Based on...", "Trade ideas:", or "**Header:**" — STOP and call a tool instead.
"""


def _build_system_prompt(cfg: Config, brain: BrainCorpus, session: str = "regular",
                          broker: Optional[Broker] = None) -> str:
    if cfg.watchlist:
        watchlist_str = ", ".join(cfg.watchlist)
    else:
        watchlist_str = (
            "OPEN UNIVERSE — no fixed watchlist. You may BUY any liquid US stock or ETF "
            "you believe is a 'gem' (undervalued, mispriced, or with strong technical/"
            "fundamental setup). Hunt across sectors. Mid-caps and small-caps welcome. "
            "Cite a specific Brain formula for every pick."
        )
    if cfg.trade_windows and broker is not None:
        _, window_msg = broker.trade_window_status(cfg.trade_windows)
        trade_window_str = window_msg
    else:
        trade_window_str = "always-on (no windows configured)"
    # Local models drown in the dense prompt — use a slim, tool-call-forward
    # template when running on Ollama. Anthropic gets the full version.
    template = OLLAMA_SYSTEM_TEMPLATE if cfg.provider == "ollama" else SYSTEM_TEMPLATE
    base = template.format(
        watchlist=watchlist_str,
        max_order_notional=cfg.max_order_notional,
        max_position_pct=cfg.max_position_pct,
        max_daily_trades=cfg.max_daily_trades,
        session=session,
        trade_window=trade_window_str,
        cycle_minutes=cfg.cycle_minutes,
        offhours_cycle_hours=cfg.offhours_cycle_hours,
        brain=brain.text or "(empty — no formulas loaded)",
    )
    # Append recent rejections so the agent learns from past failures within the session.
    rejections = db.recent_rejections(cfg.db_path, limit=8)
    if rejections:
        lines = ["", "==== RECENT REJECTIONS (do NOT repeat these mistakes) ===="]
        for r in rejections:
            sym = r["symbol"] or "?"
            side = (r["side"] or "?").upper()
            reason = r["reject_reason"] or "?"
            lines.append(f"- {side} {sym}: {reason}")
        lines.append("==========================================================")
        base = base + "\n".join(lines)

    # Append the agent's own watchlist — names it flagged for future action.
    # This is your "subconscious tracking" — names worth monitoring even when
    # you didn't trade them. Re-evaluate them this cycle.
    watching = db.tracked_symbols(cfg.db_path, limit=20)
    if watching:
        wlines = ["", "==== YOUR WATCHLIST (names you flagged for later) ===="]
        wlines.append("Re-check these every cycle. Act when conditions trigger or untrack.")
        for w in watching:
            sym = w["symbol"]
            reason = w["reason"]
            formula = w["formula"] or ""
            mv = w["metric_value"]
            th = w["threshold"]
            metric_str = ""
            if mv is not None and th is not None:
                metric_str = f" [{formula} {mv:.2f}, trigger at {th:.2f}]"
            elif formula:
                metric_str = f" [{formula}]"
            wlines.append(f"- {sym}{metric_str}: {reason}")
        wlines.append("=====================================================")
        base = base + "\n" + "\n".join(wlines)
    return base


def _cleanup_stale_orders(cfg: Config, broker: Broker) -> None:
    """At cycle start: cancel any orders that haven't filled within auto_cancel_after_minutes.
    Updates the existing trade row's status to 'canceled' (no duplicate row inserted)."""
    try:
        canceled = broker.cancel_stale_orders(max_age_minutes=cfg.auto_cancel_minutes)
        for oid in canceled:
            db.update_trade_status(cfg.db_path, oid, "canceled",
                                    reject_reason=f"auto-canceled: unfilled > {cfg.auto_cancel_minutes}m")
    except Exception as e:
        print(f"[cleanup] error canceling stale orders: {e}")


def backfill_fill_prices(cfg: Config, broker: Broker) -> int:
    """One-shot pass: for every filled trade in the DB that's missing
    filled_avg_price, fetch the order from Alpaca and write the fill price
    back. Runs on startup so historical P/L computations work."""
    import sqlite3
    with sqlite3.connect(cfg.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, order_id FROM trades "
            "WHERE status = 'filled' AND order_id IS NOT NULL "
            "  AND filled_avg_price IS NULL"
        ).fetchall()
    backfilled = 0
    for r in rows:
        try:
            order = broker.get_order(r["order_id"])
        except Exception:
            continue
        try:
            px = float(getattr(order, "filled_avg_price", 0) or 0)
            qty = float(getattr(order, "filled_qty", 0) or 0)
        except (TypeError, ValueError):
            continue
        if px <= 0:
            continue
        try:
            db.update_trade_status(
                cfg.db_path, r["order_id"], "filled",
                filled_qty=qty if qty > 0 else None,
                filled_notional=qty * px if qty > 0 else None,
                filled_avg_price=px,
            )
            backfilled += 1
        except Exception:
            pass
    return backfilled


def sync_order_statuses(cfg: Config, broker: Broker) -> int:
    """Pull live status from Alpaca for any local trade not yet in a terminal state.
    Updates the SQLite trades table. Returns number of rows updated.
    Safe to call frequently — only queries non-terminal orders."""
    open_ids = db.open_trade_order_ids(cfg.db_path)
    if not open_ids:
        return 0
    updated = 0
    for oid in open_ids:
        try:
            order = broker.get_order(oid)
        except Exception:
            continue
        # Alpaca returns OrderStatus enum; extract the string value (e.g. "filled")
        # rather than the str(enum) which gives "OrderStatus.FILLED"
        raw_status = getattr(order, "status", None)
        status = str(getattr(raw_status, "value", None) or raw_status or "").lower()
        if not status:
            continue
        # Pull filled qty / fill price so the trades table reflects real
        # executed size (not just whatever was in the original request)
        try:
            filled_qty = float(getattr(order, "filled_qty", 0) or 0)
        except (TypeError, ValueError):
            filled_qty = 0.0
        try:
            filled_avg_price = float(getattr(order, "filled_avg_price", 0) or 0)
        except (TypeError, ValueError):
            filled_avg_price = 0.0
        filled_notional = (filled_qty * filled_avg_price) if (filled_qty and filled_avg_price) else None
        try:
            db.update_trade_status(cfg.db_path, oid, status,
                                    filled_qty=filled_qty if filled_qty > 0 else None,
                                    filled_notional=filled_notional,
                                    filled_avg_price=filled_avg_price if filled_avg_price > 0 else None)
            updated += 1
        except Exception:
            pass
    return updated


def _extract_summary_from_messy(text: str) -> Optional[str]:
    """Smaller models sometimes emit tool calls as text JSON instead of structured calls.
    Try to find a {"summary": "..."} blob and extract just the summary string.
    """
    import re
    m = re.search(r'"summary"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1).strip()
    return None


def _smart_truncate(text: str, max_len: int = 500) -> str:
    """Truncate to a sentence boundary if possible. Avoids mid-word/mid-sentence cutoff.
    Also: if the model emitted stray JSON tool-call text, surface just the summary string.
    """
    text = text.strip()
    # If the response looks like text JSON containing a summary, extract that instead
    if '"summary"' in text and ("{" in text and "}" in text):
        extracted = _extract_summary_from_messy(text)
        if extracted:
            text = extracted
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    end_marks = [cut.rfind(". "), cut.rfind("! "), cut.rfind("? "),
                 cut.rfind(".\n"), cut.rfind("!\n"), cut.rfind("?\n")]
    last = max(end_marks)
    if last > max_len // 2:
        return cut[:last + 1].strip()
    last_space = cut.rfind(" ")
    if last_space > max_len // 2:
        return cut[:last_space].rstrip() + "…"
    return cut.rstrip() + "…"


def _tools_for_openai() -> list[dict]:
    """Convert Anthropic-style TOOLS to OpenAI function-calling format."""
    return [{"type": "function",
             "function": {"name": t["name"],
                          "description": t["description"],
                          "parameters": t["input_schema"]}}
            for t in TOOLS]


def _run_anthropic_cycle(cfg: Config, broker: Broker, brain: BrainCorpus, dry_run: bool) -> CycleResult:
    if not dry_run:
        # Housekeeping (sync + stale-cancel) runs in a background thread so it
        # doesn't hold the cycle_lock during slow Alpaca round-trips. The
        # dashboard also syncs order status independently every 10s.
        import threading as _t
        def _bg_housekeeping():
            try:
                sync_order_statuses(cfg, broker)
            except Exception:
                pass
            try:
                _cleanup_stale_orders(cfg, broker)
            except Exception:
                pass
        _t.Thread(target=_bg_housekeeping, daemon=True).start()
    client = Anthropic(api_key=cfg.anthropic_api_key)
    system = _build_system_prompt(cfg, brain, broker.current_session(), broker)
    handle, state = _make_tool_handler(cfg, broker, dry_run)
    messages: list[dict] = [{"role": "user", "content":
        "Begin cycle. Be deliberate — survey 5–8 candidates, run the full "
        "formula stack on the top 2–3, only place an order if EVERY gate "
        "passes. No trade is the right answer most cycles."}]
    reasoning_parts: list[str] = []

    for _ in range(40):
        # Honor user STOP between iterations — abort cleanly without sending more API calls
        if not dry_run and db.is_paused(cfg.db_path):
            reasoning_parts.append("(cycle aborted by user STOP)")
            break
        resp = client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=TOOLS,
            messages=messages,
        )

        for block in resp.content:
            if block.type == "text" and block.text.strip():
                reasoning_parts.append(block.text)

        if resp.stop_reason != "tool_use":
            break

        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                try:
                    result = handle(block.name, block.input or {})
                except Exception as e:
                    result = {"error": str(e)}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": tool_results})

        if state["finished"]:
            break

    summary = state["summary"] or (_smart_truncate(reasoning_parts[-1]) if reasoning_parts else "(no summary)")
    reasoning = "\n\n".join(reasoning_parts)
    db.log_decision(cfg.db_path, summary, reasoning)
    return CycleResult(summary=summary, reasoning=reasoning, orders_submitted=state["orders"])


def _tools_for_ollama_native() -> list[dict]:
    """Ollama's native /api/chat tools format — same shape as OpenAI's,
    but Ollama applies the right per-model chat template internally."""
    return [{"type": "function",
             "function": {"name": t["name"],
                          "description": t["description"],
                          "parameters": t["input_schema"]}}
            for t in TOOLS]


def _run_ollama_cycle(cfg: Config, broker: Broker, brain: BrainCorpus, dry_run: bool) -> CycleResult:
    """Native Ollama chat with proper tool calling (more reliable than the /v1
    OpenAI-compat layer for many local models like Llama 3.1, Qwen 2.5)."""
    import ollama

    if not dry_run:
        # Move housekeeping to a background thread so it doesn't extend the
        # cycle's lock-hold time (Alpaca polls + cancels can be slow).
        import threading as _t
        def _bg_housekeeping():
            try:
                sync_order_statuses(cfg, broker)
            except Exception:
                pass
            try:
                _cleanup_stale_orders(cfg, broker)
            except Exception:
                pass
        _t.Thread(target=_bg_housekeeping, daemon=True).start()

    client = ollama.Client(host=cfg.ollama_host)
    system = _build_system_prompt(cfg, brain, broker.current_session(), broker)
    handle, state = _make_tool_handler(cfg, broker, dry_run)
    tools = _tools_for_ollama_native()

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content":
            "Begin cycle. Your FIRST output must be a tool call (get_account, "
            "get_positions, or get_news). Not a text reply. Go."},
    ]
    reasoning_parts: list[str] = []
    nudges_used = 0
    MAX_NUDGES = 2

    for _ in range(40):
        # Honor user STOP between iterations — abort cleanly without sending more requests
        if not dry_run and db.is_paused(cfg.db_path):
            reasoning_parts.append("(cycle aborted by user STOP)")
            break
        resp = client.chat(
            model=cfg.ollama_model,
            messages=messages,
            tools=tools,
            options={"num_predict": cfg.max_tokens},
        )
        msg = resp["message"]
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        if content.strip():
            reasoning_parts.append(content)

        if not tool_calls:
            # Try to rescue stray tool calls embedded in text
            rescued = _rescue_tool_calls_from_text(content)
            if rescued:
                tool_calls = rescued
            elif nudges_used < MAX_NUDGES and not state["finished"]:
                # Model stopped without calling a tool. Detect "I will trade X"
                # prose vs benign no-trade and nudge accordingly.
                lower = content.lower()
                trade_words = ("sell ", "buy ", "short ", "cover ",
                               "exit ", "enter ", "open ", "close ",
                               "i will sell", "i will buy", "i will short",
                               "i will exit", "i will open", "place an order")
                looks_like_unexecuted = any(w in lower for w in trade_words)
                messages.append({"role": "assistant", "content": content})
                if looks_like_unexecuted:
                    nudge = (
                        "STOP. You described a trade in text but did NOT call "
                        "place_order. Text descriptions are NOT trades. "
                        "If you actually want to trade, invoke place_order "
                        "(or place_option_order) RIGHT NOW with the symbol, "
                        "side, qty/notional, and rationale. If on second "
                        "thought you don't want to trade, call finish() with "
                        "the reason. No more text replies."
                    )
                else:
                    nudge = (
                        "You stopped without calling the `finish` tool. "
                        "Either continue analyzing with more tool calls, or "
                        "call `finish` now with a 1–2 sentence summary."
                    )
                messages.append({"role": "user", "content": nudge})
                nudges_used += 1
                continue
            else:
                break

        # Append assistant turn so the next round is well-formed
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", {})
            # Ollama returns arguments as dict already; some models stringify them
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {}
            else:
                args = raw_args or {}
            try:
                result = handle(name, args)
            except Exception as e:
                result = {"error": str(e)}
            messages.append({
                "role": "tool",
                "content": json.dumps(result, default=str),
            })

        if state["finished"]:
            break

    summary = state["summary"] or (_smart_truncate(reasoning_parts[-1]) if reasoning_parts else "(no summary)")
    reasoning = "\n\n".join(reasoning_parts)
    db.log_decision(cfg.db_path, summary, reasoning)
    return CycleResult(summary=summary, reasoning=reasoning, orders_submitted=state["orders"])


def _rescue_tool_calls_from_text(text: str) -> list[dict]:
    """If a model emitted tool calls as text instead of structured tool_calls,
    try to extract them. Looks for JSON objects with 'name' and 'arguments' keys.
    Returns a list of tool call dicts or [] if nothing parseable."""
    import re
    if not text:
        return []
    rescued: list[dict] = []
    # Find every JSON object that contains both "name" and "arguments"
    for m in re.finditer(r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}', text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "name" in obj:
            rescued.append({"function": {"name": obj["name"], "arguments": obj.get("arguments", {})}})
    return rescued


def run_cycle(cfg: Config, broker: Broker, brain: BrainCorpus, dry_run: bool = False) -> CycleResult:
    if cfg.provider == "ollama":
        return _run_ollama_cycle(cfg, broker, brain, dry_run)
    return _run_anthropic_cycle(cfg, broker, brain, dry_run)


def main() -> None:
    import argparse
    from pathlib import Path
    from quant_pi.config import load_config, require_keys
    from quant_pi.brain.loader import load_corpus

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Mock place_order")
    args = parser.parse_args()

    cfg = load_config(Path(__file__).resolve().parents[2] / "config.toml")
    require_keys(cfg)
    db.init_db(cfg.db_path)

    brain = load_corpus(cfg.vault_path, cfg.vault_subfolder, cfg.brain_root_note, cfg.brain_max_depth)
    print(f"Brain: {brain.file_count} files, ~{brain.approx_tokens:,} tokens")

    broker = Broker(cfg.alpaca_api_key, cfg.alpaca_secret_key, paper=cfg.alpaca_paper)
    result = run_cycle(cfg, broker, brain, dry_run=args.dry_run)

    print(f"\n=== Summary ===\n{result.summary}")
    print(f"Orders: {result.orders_submitted}")
    print(f"\n=== Reasoning ===\n{result.reasoning}")


if __name__ == "__main__":
    main()

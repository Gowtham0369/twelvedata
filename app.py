"""Chip & AI Board: daily price, RSI and MACD for tech, AI, semiconductor and memory stocks.

Data: Twelve Data REST API (https://twelvedata.com). Put your key in Streamlit secrets as
TWELVE_DATA_API_KEY. Free plan limits: 8 credits per minute, 800 per day (1 credit per stock).
"""
import datetime as dt
import threading
import time
from collections import deque
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Chip & AI Board", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- settings
API = "https://api.twelvedata.com"
KEY = st.secrets.get("TWELVE_DATA_API_KEY", "")
CALLS_PER_MIN = int(st.secrets.get("CALLS_PER_MIN", 7))      # stay under the free plan's 8/min
REFRESH_MIN = int(st.secrets.get("REFRESH_MIN", 15))         # minutes between refreshes while the market is open
DISPLAY_TZ = ZoneInfo(st.secrets.get("DISPLAY_TZ", "Asia/Dubai"))
NY = ZoneInfo("America/New_York")

DEFAULT_GROUPS = {
    "Semiconductors": ["NVDA", "AMD", "AVGO", "INTC", "TSM", "ASML", "MRVL", "QCOM", "ARM"],
    "Memory & storage": ["MU", "SKHY", "SNDK", "WDC", "STX"],
    "AI & infrastructure": ["CRWV", "DELL", "SMCI", "ORCL", "PLTR", "SPCX"],
    "Big tech & software": ["MSFT", "GOOGL", "AMZN", "META", "AAPL", "SNOW"],
}


# ---------------------------------------------------------------- market clock
def market_open(now=None) -> bool:
    now = now or dt.datetime.now(NY)
    mins = now.hour * 60 + now.minute
    return now.weekday() < 5 and 570 <= mins < 960          # 9:30-16:00 New York, holidays not detected


def last_session(now) -> dt.date:
    d = now.date()
    if now.weekday() < 5 and now.hour * 60 + now.minute >= 960:
        return d
    d -= dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def refresh_slot() -> str:
    """Cache key: changes every REFRESH_MIN minutes while the market is open, once per closed period otherwise."""
    now = dt.datetime.now(NY)
    if market_open(now):
        return f"{now:%Y-%m-%d}-{(now.hour * 60 + now.minute) // REFRESH_MIN}"
    return f"closed-{last_session(now)}"


# ---------------------------------------------------------------- Twelve Data client
class TDError(Exception):
    def __init__(self, msg, kind="other"):
        super().__init__(msg)
        self.kind = kind  # "daily", "auth" or "other"


@st.cache_resource
def _limiter():
    return {"lock": threading.Lock(), "calls": deque()}


def _throttle():
    lim = _limiter()
    with lim["lock"]:
        while True:
            now = time.time()
            while lim["calls"] and now - lim["calls"][0] > 60:
                lim["calls"].popleft()
            if len(lim["calls"]) < CALLS_PER_MIN:
                lim["calls"].append(now)
                return
            time.sleep(max(0.5, 61 - (now - lim["calls"][0])))


def td_get(path, **params):
    for attempt in range(2):
        _throttle()
        r = requests.get(f"{API}/{path}", params={**params, "apikey": KEY}, timeout=20)
        try:
            j = r.json()
        except ValueError:
            raise TDError(f"Twelve Data returned HTTP {r.status_code}")
        if isinstance(j, dict) and j.get("status") == "error":
            msg = j.get("message", "Unknown error")
            if "current minute" in msg and attempt == 0:
                time.sleep(62)
                continue
            kind = "daily" if "run out of API credits" in msg else "auth" if j.get("code") in (401, 403) else "other"
            raise TDError(msg, kind)
        return j
    raise TDError("Per-minute credit limit reached twice in a row.")


@st.cache_data(ttl=24 * 3600, max_entries=1000, show_spinner=False)
def get_series(symbol: str, slot: str):
    j = td_get("time_series", symbol=symbol, interval="1day", outputsize=260, country="United States")
    vals = j.get("values") or []
    if not vals:
        raise TDError(f"No price data for {symbol}")
    df = pd.DataFrame(vals)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df[df["close"] > 0].iloc[::-1].reset_index(drop=True)   # oldest first
    return df, time.time()


@st.cache_data(ttl=120, show_spinner=False)
def api_usage():
    try:
        j = requests.get(f"{API}/api_usage", params={"apikey": KEY}, timeout=10).json()
        return int(j["current_usage"]), int(j["plan_limit"])
    except Exception:
        return None


@st.cache_resource
def last_good():
    return {}   # symbol -> last successful metrics, shown if a refresh fails


# ---------------------------------------------------------------- indicators
def ema_seeded(a: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) < p:
        return out
    k = 2 / (p + 1)
    e = a[:p].mean()
    out[p - 1] = e
    for i in range(p, len(a)):
        e = a[i] * k + e * (1 - k)
        out[i] = e
    return out


def rsi_wilder(c: np.ndarray, p: int = 14):
    if len(c) <= p:
        return None
    d = np.diff(c)
    g, l = d[:p].clip(min=0).mean(), (-d[:p]).clip(min=0).mean()
    for x in d[p:]:
        g = (g * (p - 1) + max(x, 0)) / p
        l = (l * (p - 1) + max(-x, 0)) / p
    return 100.0 if l == 0 else 100 - 100 / (1 + g / l)


def compute(df: pd.DataFrame) -> dict:
    c = df["close"].to_numpy(float)
    n = len(c)
    m = {"date": df["datetime"].iloc[-1], "price": c[-1],
         "day_pct": (c[-1] / c[-2] - 1) * 100 if n > 1 else None,
         "wk_pct": (c[-1] / c[-6] - 1) * 100 if n > 5 else None,
         "rsi": rsi_wilder(c)}
    e12, e26 = ema_seeded(c, 12), ema_seeded(c, 26)
    macd = (e12 - e26)[~np.isnan(e26)]
    sig = ema_seeded(macd, 9)
    hist = (macd - sig)[~np.isnan(sig)]
    m["hist_tail"], m["cross_dir"], m["cross_ago"], m["macd_state"] = [], None, None, "Not enough history"
    if len(hist) >= 2:
        m["macd"], m["signal"], m["hist"] = macd[-1], sig[-1], hist[-1]
        m["hist_tail"] = [round(float(x), 4) for x in hist[-14:]]
        for k in range(0, 4):
            if len(hist) - 2 - k < 0:
                break
            a, b = hist[-1 - k], hist[-2 - k]
            if (a > 0) != (b > 0):
                m["cross_dir"], m["cross_ago"] = ("bull" if a > 0 else "bear"), k
                break
        rising = hist[-1] > hist[-2]
        if m["cross_dir"]:
            when = "today" if m["cross_ago"] == 0 else f"{m['cross_ago']}d ago"
            m["macd_state"] = f"{'Bullish' if m['cross_dir'] == 'bull' else 'Bearish'} cross {when}"
        elif hist[-1] > 0:
            m["macd_state"] = "Bullish, strengthening" if rising else "Bullish, fading"
        else:
            m["macd_state"] = "Bearish, improving" if rising else "Bearish, weakening"
    tail = df.tail(252)
    m["hi52"], m["lo52"] = tail["high"].max(), tail["low"][tail["low"] > 0].min()
    m["off52"] = (c[-1] / m["hi52"] - 1) * 100
    m["sma50"] = c[-50:].mean() if n >= 50 else None
    m["sma200"] = c[-200:].mean() if n >= 200 else None
    vols = df["volume"].iloc[-21:-1]
    vols = vols[vols > 0]
    m["rvol"] = df["volume"].iloc[-1] / vols.mean() if len(vols) >= 10 and df["volume"].iloc[-1] > 0 else None
    if m["sma50"] is None:
        m["trend"] = "Short history"
    else:
        a50 = c[-1] > m["sma50"]
        a200 = None if m["sma200"] is None else c[-1] > m["sma200"]
        m["trend"] = ("Above 50-day" if a50 else "Below 50-day") if a200 is None else \
            "Above 50 & 200-day" if a50 and a200 else "Below 50 & 200-day" if not a50 and not a200 else \
            "Above 200, below 50-day" if a200 else "Above 50, below 200-day"
    m["closes_90"] = [round(float(x), 2) for x in c[-90:]]
    return m


def explain(s: str, m: dict) -> list[str]:
    out = []
    r = m["rsi"]
    if r is not None:
        out.append(f"RSI is {r:.0f}: " + ("overbought. Strong momentum, but pullbacks are common from here." if r >= 70
                   else "close to overbought (70)." if r >= 65 else "oversold. Selling may be stretched." if r <= 30
                   else "close to oversold (30)." if r <= 35 else "neutral."))
    if m.get("hist") is not None:
        out.append(f"MACD {m['macd']:.2f} vs signal {m['signal']:.2f}: {m['macd_state'].lower()}.")
    if m["sma50"] is not None:
        t = f"Price is {'above' if m['price'] > m['sma50'] else 'below'} its 50-day average ({m['sma50']:.2f})"
        if m["sma200"] is not None:
            t += f" and {'above' if m['price'] > m['sma200'] else 'below'} its 200-day average ({m['sma200']:.2f})"
        out.append(t + ".")
    out.append(f"52-week range {m['lo52']:.2f} to {m['hi52']:.2f}; now {m['off52']:+.1f}% from the high.")
    if m["rvol"] is not None:
        out.append(f"Volume is {m['rvol']:.1f}x its 20-day average" + (" so far today." if market_open() else "."))
    return out


# ---------------------------------------------------------------- page
st.title("Chip & AI Board")
st.caption("Daily price, RSI 14 and MACD 12/26/9 for tech, AI, semiconductor and memory stocks. Data: Twelve Data.")

if not KEY:
    st.error("Add your Twelve Data API key to this app's secrets as TWELVE_DATA_API_KEY, then reload.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    auto = st.toggle(f"Auto-refresh every {REFRESH_MIN} min (market hours)", value=True)
    st.subheader("Watchlist")
    st.caption("Comma-separated tickers. Changes last for this browser session; edit DEFAULT_GROUPS in app.py to make them permanent.")
    groups = {}
    for g, syms in DEFAULT_GROUPS.items():
        raw = st.text_input(g, value=", ".join(syms), key=f"wl-{g}")
        groups[g] = [s.strip().upper() for s in raw.split(",") if s.strip()]
    n_syms = sum(len(v) for v in groups.values())
    st.caption(f"{n_syms} stocks = {n_syms} credits per refresh, about {max(1, round(n_syms / CALLS_PER_MIN))} min to load on the free plan.")


@st.fragment(run_every=f"{REFRESH_MIN}m" if auto else None)
def board():
    is_open = market_open()
    slot = refresh_slot()
    rows, errors, stale = [], {}, set()
    pairs = [(g, s) for g, syms in groups.items() for s in syms]
    prog = st.progress(0.0, text="Loading prices…")
    stop = None
    for i, (g, s) in enumerate(pairs):
        m = None
        if stop is None:
            try:
                df, fetched = get_series(s, slot)
                m = compute(df) | {"fetched": fetched}
                last_good()[s] = m
            except TDError as e:
                errors[s] = str(e)
                if e.kind in ("daily", "auth"):
                    stop = e
            except Exception as e:  # network errors etc.
                errors[s] = f"{type(e).__name__}: {e}"
        if m is None and s in last_good():
            m, _ = last_good()[s], stale.add(s)
        if m is not None:
            rows.append(m | {"sym": s, "group": g})
        prog.progress((i + 1) / len(pairs), text=f"Loaded {i + 1} of {len(pairs)}")
    prog.empty()

    # status line
    now_local = dt.datetime.now(DISPLAY_TZ)
    usage = api_usage()
    status = [f"US market **{'open' if is_open else 'closed'}**", f"checked {now_local:%H:%M} (UTC{now_local.utcoffset().total_seconds() / 3600:+g})"]
    if usage:
        status.append(f"Twelve Data credits today: **{usage[0]} of {usage[1]}**")
    status.append(f"auto-refresh every {REFRESH_MIN} min while open" if auto else "auto-refresh off")
    st.markdown(" · ".join(status))
    if stop is not None:
        st.error("Twelve Data stopped answering: " + str(stop) +
                 (" Showing the last values; this resets tomorrow." if stop.kind == "daily" else " Check the API key in Secrets."))
    if errors and stop is None:
        with st.expander(f"{len(errors)} ticker(s) failed to load"):
            for s, e in errors.items():
                st.write(f"**{s}**: {e}")
    if not rows:
        st.info("No data yet.")
        return

    df = pd.DataFrame(rows)
    filters = {
        "All": lambda d: d.index == d.index,
        "Overbought (RSI 70+)": lambda d: d["rsi"] >= 70,
        "Oversold (RSI 30 or less)": lambda d: d["rsi"] <= 30,
        "MACD bullish cross, 3 days": lambda d: d["cross_dir"] == "bull",
        "MACD bearish cross, 3 days": lambda d: d["cross_dir"] == "bear",
        "Up today": lambda d: d["day_pct"] > 0,
        "Down today": lambda d: d["day_pct"] < 0,
    }
    cols = st.columns(6)
    for col, name in zip(cols, list(filters)[1:]):
        col.metric(name, int(filters[name](df).sum()))

    c1, c2 = st.columns([3, 2])
    pick = c1.radio("Show", list(filters), horizontal=True, key="filter")
    sort = c2.selectbox("Sort", ["By group", "Today's move", "RSI, high to low", "Closest to 52-week high"], key="sort")
    view = df[filters[pick](df)]

    table_cfg = {
        "sym": st.column_config.TextColumn("Stock", width="small"),
        "price": st.column_config.NumberColumn("Price", format="%.2f"),
        "day_pct": st.column_config.NumberColumn("Today", format="%+.2f%%"),
        "wk_pct": st.column_config.NumberColumn("5 days", format="%+.2f%%"),
        "rsi": st.column_config.ProgressColumn("RSI 14", min_value=0, max_value=100, format="%.0f"),
        "macd_state": st.column_config.TextColumn("MACD"),
        "hist_tail": st.column_config.BarChartColumn("MACD histogram (14d)"),
        "trend": st.column_config.TextColumn("Trend"),
        "off52": st.column_config.NumberColumn("From 52w high", format="%+.1f%%"),
        "rvol": st.column_config.NumberColumn("Volume vs avg", format="%.1fx"),
        "closes_90": st.column_config.LineChartColumn("90 days"),
        "date": st.column_config.TextColumn("As of"),
    }
    show = list(table_cfg)

    def colour(v):
        if pd.isna(v):
            return ""
        return "color: #0B7A5E" if v > 0 else "color: #B42318" if v < 0 else ""

    def table(d):
        styled = d[show].style.map(colour, subset=["day_pct", "wk_pct"])
        st.dataframe(styled, column_config=table_cfg, hide_index=True, width="stretch")

    if sort == "By group":
        for g in groups:
            d = view[view["group"] == g]
            if len(d):
                st.subheader(g)
                table(d)
    else:
        key = {"Today's move": "day_pct", "RSI, high to low": "rsi", "Closest to 52-week high": "off52"}[sort]
        table(view.sort_values(key, ascending=False, na_position="last"))
    if stale:
        st.caption("Showing earlier values for: " + ", ".join(sorted(stale)))

    st.subheader("Look closer")
    s = st.selectbox("Stock", df["sym"].tolist(), key="detail")
    m = df[df["sym"] == s].iloc[0].to_dict()
    left, right = st.columns([3, 2])
    closes = pd.Series(m["closes_90"], name="Close")
    left.line_chart(pd.DataFrame({"Close": closes, "50-day average": closes.rolling(50).mean()}))
    right.markdown("\n".join(f"- {t}" for t in explain(s, m)))


board()
st.caption("Indicators use daily bars; during US market hours the latest bar is still forming. "
           "Market hours ignore exchange holidays. Not investment advice.")

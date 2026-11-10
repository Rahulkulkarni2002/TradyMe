"""News mood (GDELT, hourly, 2017+) and the Crypto Fear & Greed index (daily, 2018+)."""
from __future__ import annotations

import time

import pandas as pd
import requests

from ..store import USER_AGENT

GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_START = pd.Timestamp("2017-01-01", tz="UTC")     # hourly timelines exist from 2017
GDELT_PAUSE = 6.0                                        # GDELT asks for ≤ 1 request / 5 s

# What the news mood is measured on. Key = column name in the dataset.
TOPICS = {
    "crypto": '(bitcoin OR cryptocurrency OR ethereum)',
    "fed": '("federal reserve" OR "interest rates" OR "rate hike" OR "rate cut")',
    "inflation": '(inflation OR "consumer prices")',
    "gold": '("gold price" OR "gold prices")',
    "dollar": '("us dollar" OR forex OR "currency markets")',
}


def gdelt_week(topic: str, start: pd.Timestamp, session: requests.Session) -> pd.DataFrame:
    """Hourly news tone (average positivity) and volume (share of all news) for one week."""
    end = start + pd.Timedelta(days=7)
    out = {}
    for mode, col in (("timelinetone", "tone"), ("timelinevol", "volume")):
        params = {"query": TOPICS[topic], "mode": mode, "format": "json",
                  "startdatetime": start.strftime("%Y%m%d%H%M%S"), "enddatetime": end.strftime("%Y%m%d%H%M%S")}
        data = None
        for attempt in range(8):
            time.sleep(GDELT_PAUSE * (1 + attempt))
            try:
                r = session.get(GDELT, params=params, timeout=60)
            except requests.RequestException:
                continue
            if r.status_code == 200 and r.text.strip().startswith("{"):
                data = r.json()
                break
        if data is None:
            raise RuntimeError(f"GDELT didn't answer for {topic} {start.date()}")
        series = (data.get("timeline") or [{}])[0].get("data", [])
        s = pd.Series({pd.Timestamp(p["date"]): float(p["value"]) for p in series}, dtype="float64")
        out[f"news_{topic}_{col}"] = s
    df = pd.DataFrame(out)
    if len(df):
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[(df.index >= start) & (df.index < end)]
    return df


GDELT_DAILY_START = pd.Timestamp("2017-01-01", tz="UTC")


def _gdelt_get(params: dict, session: requests.Session, say=print) -> dict:
    """One GDELT request, patiently: GDELT allows ~1 request / 5 s and often answers 'too many requests'."""
    last = "no answer"
    for attempt in range(7):
        time.sleep(10 if attempt == 0 else min(20 * attempt, 120))
        try:
            r = session.get(GDELT, params=params, timeout=90)
        except requests.RequestException as e:
            last = f"connection dropped ({type(e).__name__})"
            continue
        if r.status_code == 200 and r.text.strip().startswith("{"):
            return r.json()
        last = "busy — asked us to slow down (429)" if r.status_code == 429 else f"HTTP {r.status_code}: {r.text[:80]!r}"
    raise RuntimeError(f"GDELT {last}")


def gdelt_quarter(topic: str, start: pd.Timestamp, session: requests.Session) -> pd.DataFrame:
    """DAILY news tone and volume for ~3 months (GDELT gives one value per day for long periods).
    Far fewer requests than hourly, so the 2017 → today history finishes in a few hours."""
    end = min(start + pd.DateOffset(months=3), pd.Timestamp.now(tz="UTC").normalize())
    out = {}
    for mode, col in (("timelinetone", "tone"), ("timelinevol", "volume")):
        data = _gdelt_get({"query": TOPICS[topic], "mode": mode, "format": "json",
                           "startdatetime": start.strftime("%Y%m%d%H%M%S"),
                           "enddatetime": end.strftime("%Y%m%d%H%M%S")}, session)
        series = (data.get("timeline") or [{}])[0].get("data", [])
        out[f"news_{topic}_{col}"] = pd.Series({pd.Timestamp(p["date"]): float(p["value"]) for p in series},
                                               dtype="float64")
    df = pd.DataFrame(out)
    if len(df):
        df.index = pd.to_datetime(df.index, utc=True).normalize()
        df = df[(df.index >= start) & (df.index < end)]
        df = df.groupby(level=0).mean()
    return df


def fear_greed(session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", USER_AGENT)
    r = s.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=30)
    r.raise_for_status()
    rows = r.json()["data"]
    df = pd.DataFrame({"fear_greed": [int(x["value"]) for x in rows]},
                      index=pd.to_datetime([int(x["timestamp"]) for x in rows], unit="s", utc=True))
    return df.sort_index()

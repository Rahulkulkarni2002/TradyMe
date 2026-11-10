"""Offline tests for the history downloader (no internet needed)."""
import io
import lzma
import struct
import zipfile

import pandas as pd
import pytest

from research import store
from research.sources import binance_vision, dukascopy


def _zip(lines: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("X-5m.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


def test_binance_parses_milliseconds_and_microseconds():
    ms = 1514764800000            # 2018-01-01 00:00 UTC in ms (old files)
    us = 1748736000000000         # 2025-06-01 00:00 UTC in µs (files from 2025)
    rest = "1,2,0.5,1.5,10,0,0,0,0,0,0"
    df = binance_vision._parse(_zip([f"{ms},{rest}", f"{us},{rest}"]))
    assert list(df.index) == [pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2025-06-01", tz="UTC")]
    assert df.iloc[0][["open", "high", "low", "close", "volume"]].tolist() == [1, 2, 0.5, 1.5, 10]


def test_binance_skips_header_row():
    df = binance_vision._parse(_zip(["open_time,open,high,low,close,volume,a,b,c,d,e,f",
                                     "1514764800000,1,2,0.5,1.5,10,0,0,0,0,0,0"]))
    assert len(df) == 1


def _bi5(records):
    raw = b"".join(struct.pack(">5if", *r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def test_dukascopy_decode_field_order_and_scale():
    day = pd.Timestamp("2020-03-02", tz="UTC")
    # time, open, close, low, high (× 1e5), volume
    payload = _bi5([(0, 110000, 110050, 109990, 110080, 12.5), (60, 110050, 110020, 110010, 110060, 3.0),
                    (120, 110020, 110020, 110020, 110020, 0.0)])          # zero volume = closed → dropped
    df = dukascopy.decode(payload, day, 1e5)
    assert len(df) == 2
    first = df.iloc[0]
    assert (first.open, first.high, first.low, first.close) == pytest.approx((1.1, 1.1008, 1.0999, 1.1005))
    assert df.index[1] == day + pd.Timedelta(minutes=1)


def test_dukascopy_detects_wrong_layout():
    day = pd.Timestamp("2020-03-02", tz="UTC")
    # high smaller than open/close in every record → layout must be rejected, never silently accepted
    payload = _bi5([(i * 60, 110000, 110050, 110100, 109900, 1.0) for i in range(10)])
    with pytest.raises(ValueError):
        dukascopy.decode(payload, day, 1e5)


def test_dukascopy_empty_day():
    assert dukascopy.decode(b"", pd.Timestamp("2020-03-07", tz="UTC"), 1e5).empty


def test_combine_snaps_to_5min_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RAW", tmp_path / "raw")
    monkeypatch.setattr(store, "PRICES", tmp_path / "prices")
    idx = pd.to_datetime(["2018-02-09 09:55:00", "2018-02-09 09:58:14.789", "2018-02-09 10:00:00"], utc=True, format="ISO8601")
    df = pd.DataFrame({"open": [1, 2, 3], "high": [2, 3, 4], "low": [0.5, 1.5, 2.5], "close": [1.5, 2.5, 3.5],
                       "volume": [1, 1, 1], "source": "binance"}, index=idx, dtype=object)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    store.save(df, store.raw_path("X", "2018-02"))
    out = store.combine("X")
    assert list(out.index) == list(pd.to_datetime(["2018-02-09 09:55", "2018-02-09 10:00"], utc=True))
    first = out.iloc[0]                                     # 09:55 and 09:58:14 merged
    assert (first.open, first.high, first.low, first.close, first.volume) == (1, 3, 0.5, 2.5, 2)

"""Zero-subscription stock data integration (research only).

Provider modes:
  alpaca: authenticated, FREE IEX-only US stock trades/quotes/5-minute bars.
  yahoo:  public chart history, NO verifiable realtime rights or accuracy.
  auto:   chooses Alpaca when free keypair exists, otherwise Yahoo history.

IMPORTANT: IEX is only one exchange, not the consolidated US market. A fresh
IEX snapshot is genuine venue data, but not a complete national best bid/offer.
No service is allowed to claim 100% live market parity.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone, time
from zoneinfo import ZoneInfo

from trading_core import Bar, ET, validate_bars, fetch_yahoo

ALPACA_BASE = 'https://data.alpaca.markets'
BAR_LENGTH = timedelta(minutes=5)
CLOSE_SETTLEMENT = timedelta(seconds=12)
MAX_CANDLE_AGE = timedelta(minutes=7)
MAX_TICK_AGE = timedelta(seconds=120)


def _request_alpaca(path, key, secret, request=None):
    if not path.startswith('/') or path.startswith('//'):
        raise ValueError('Invalid Alpaca endpoint')
    req = urllib.request.Request(ALPACA_BASE + path, headers={
        'APCA-API-KEY-ID': key,
        'APCA-API-SECRET-KEY': secret,
        'Accept': 'application/json',
        'User-Agent': 'TradingIntelligenceResearchFree/3.0',
    })
    caller = request or urllib.request.urlopen
    try:
        with caller(req, timeout=20) as rsp:
            data = rsp.read(25_000_000)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError('Free Alpaca account/API key invalid or feed access denied. Use --source yahoo for history.') from exc
        if exc.code == 429:
            raise RuntimeError('Alpaca free data rate limit reached; try later. No current signal possible.') from exc
        raise RuntimeError('Alpaca data HTTP error ' + str(exc.code)) from exc
    except Exception as exc:
        raise RuntimeError('Alpaca data unavailable: ' + str(exc)) from exc
    try:
        result=json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError('Alpaca response is not valid JSON.') from exc
    if not isinstance(result,dict) or result.get('error') or result.get('code'):
        raise RuntimeError('Alpaca reports an unusable data response.')
    return result


def parse_utc(timestamp):
    if not isinstance(timestamp,str) or not timestamp:
        raise ValueError('Missing provider timestamp.')
    dt=datetime.fromisoformat(timestamp.replace('Z','+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Provider timestamp has no timezone.')
    return dt.astimezone(timezone.utc)


def market_open(now):
    """Use the real US stock exchange calendar including holidays/early closes.

    If exchange_calendars is missing, return an error and BLOCK live status,
    instead of guessing trading days or early close times.
    """
    try:
        import exchange_calendars as xcals
        import pandas as pd
        return bool(xcals.get_calendar('XNYS').is_open_on_minute(pd.Timestamp(now)))
    except Exception:
        return None


def verify_iex_only(bars, snapshot, symbol, now, market_open_func=None):
    """Validate genuine IEX timestamps, but NEVER assert whole-market coverage."""
    now=now.astimezone(timezone.utc)
    blockers=[]
    evidence={'status':'BLOCKED','source':'Alpaca Basic IEX','market_coverage':'IEX_ONLY_NOT_CONSOLIDATED',
              'asof_checked_utc':now.isoformat(), 'quote':None,'trade':None,
              'blockers':blockers}
    m=(market_open_func or market_open)(now)
    evidence['us_market_open']=m
    if m is not True:
        blockers.append('US_STOCK_EXCHANGE_CLOSED_OR_CALENDAR_UNAVAILABLE')
    local=now.astimezone(ET)
    if not (time(9,30) <= local.time() < time(16,0)):
        blockers.append('OUTSIDE_REGULAR_TRADING_HOURS')
    if not bars:
        blockers.append('NO_COMPLETED_5M_BARS')
    else:
        end=bars[-1].ts.astimezone(timezone.utc)+BAR_LENGTH
        evidence['last_closed_bar_end_utc']=end.isoformat()
        evidence['last_bar_age_seconds']=round((now-end).total_seconds(),1)
        if now-end < CLOSE_SETTLEMENT or now-end > MAX_CANDLE_AGE:
            blockers.append('CANDLE_UNSETTLED_OR_STALE')
        if bars[-1].ts.astimezone(ET).date()!=local.date():
            blockers.append('NO_5M_BAR_TODAY')
        if bars[-1].volume <= 0:
            blockers.append('ZERO_VOLUME_BAR')
    if not isinstance(snapshot,dict) or snapshot.get('symbol') != symbol:
        blockers.append('SYMBOL_MISMATCH')
        snapshot={}
    for kind in ('quote','trade'):
        d=snapshot.get('latestQuote' if kind=='quote' else 'latestTrade') or {}
        try:
            at=parse_utc(d['t']); age=now-at
            if age < timedelta(0) or age > MAX_TICK_AGE:
                blockers.append('STALE_OR_FUTURE_'+kind.upper())
            if kind=='quote':
                bid=float(d['bp']); ask=float(d['ap'])
                if not 0 < bid <= ask or (ask-bid)/((ask+bid)/2) > .01:
                    raise ValueError('Invalid or wide IEX bid/ask quote')
                if d.get('bx')!='V' or d.get('ax')!='V':
                    raise ValueError('Quote not confirmed as IEX exchange')
                evidence[kind]={'bid':bid,'ask':ask,'exchange':'IEX',
                                'time_utc':at.isoformat(),'age_seconds':round(age.total_seconds(),1)}
            else:
                price=float(d['p'])
                if price<=0 or d.get('x')!='V':
                    raise ValueError('Last trade not from IEX or invalid price')
                evidence[kind]={'price':price,'exchange':'IEX',
                                'time_utc':at.isoformat(),'age_seconds':round(age.total_seconds(),1)}
        except (ValueError,TypeError,KeyError,OverflowError,OSError):
            blockers.append('MISSING_VALID_IEX_'+kind.upper())
    if evidence['trade'] and bars:
        if abs(evidence['trade']['price']-bars[-1].close)/bars[-1].close > .08:
            blockers.append('TRADE_BAR_PRICE_DIFFERENCE_GT_8_PERCENT')
    if not blockers:
        evidence['status']='LIVE_IEX_ONLY'
    return evidence


def fetch_alpaca_iex(symbol, days=55, now=None, key=None, secret=None, request=None, history_only=False, market_open_func=None):
    """Free Alpaca IEX 5m bars + snapshot, with strict source and timestamp checks."""
    sym=symbol.strip().upper()
    if not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,13}',sym):
        raise ValueError('Only US stock symbols e.g. AAPL, MSFT, BRK.B are supported.')
    if not 5<=days<=58:
        raise ValueError('days must be between 5 and 58')
    key=key or os.environ.get('ALPACA_API_KEY_ID','')
    secret=secret or os.environ.get('ALPACA_API_SECRET_KEY','')
    if not key or not secret:
        raise RuntimeError('A FREE Alpaca API keypair is required for IEX: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.')
    now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    start=(now-timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end=now.strftime('%Y-%m-%dT%H:%M:%SZ')
    base='/v2/stocks/'+urllib.parse.quote(sym,safe='')+'/bars?'+urllib.parse.urlencode({
        'timeframe':'5Min','start':start,'end':end,'feed':'iex',
        'adjustment':'raw','sort':'asc','limit':10000})
    rows=[]; visited=set(); next_token=None; pages=0
    while True:
        path=base + (('&page_token='+urllib.parse.quote(next_token,safe='')) if next_token else '')
        if path in visited or pages>=8:
            raise RuntimeError('Cannot verify complete IEX history: pagination error or too many pages.')
        visited.add(path);pages+=1
        response=_request_alpaca(path,key,secret,request=request)
        if response.get('symbol',sym)!=sym or not isinstance(response.get('bars'),list):
            raise RuntimeError('Unexpected Alpaca historical bars response.')
        rows.extend(response['bars'])
        next_token=response.get('next_page_token')
        if not next_token:break
    bars=[]
    for item in rows:
        try:
            ts=parse_utc(item['t']); local=ts.astimezone(ET)
            if not time(9,30)<=local.time()<time(16,0):continue
            if ts.second or ts.microsecond or (local.minute % 5):
                raise ValueError('Non-aligned 5-minute candle.')
            if ts+BAR_LENGTH+CLOSE_SETTLEMENT>now:continue
            bars.append(Bar(local, *(float(item[k]) for k in ('o','h','l','c','v'))))
        except (KeyError,ValueError,TypeError,OverflowError) as exc:
            raise RuntimeError('Alpaca IEX bars include corrupt data.') from exc
    bars=validate_bars(bars)
    evidence={'status':'HISTORICAL_ONLY','source':'Alpaca Basic IEX',
              'market_coverage':'IEX_ONLY_NOT_CONSOLIDATED',
              'blockers':['HISTORY_ONLY_REQUEST'], 'asof_checked_utc':now.isoformat()}
    if not history_only:
        snap=_request_alpaca('/v2/stocks/'+urllib.parse.quote(sym,safe='')+'/snapshot?feed=iex',key,secret,request=request)
        evidence=verify_iex_only(bars,snap,sym,now,market_open_func=market_open_func)
    meta={'provider':'Alpaca Basic (free IEX-only US equities)', 'symbol':sym,'interval':'5m',
          'currency':'USD','retrieved_at_utc':now.isoformat(), 'synthetic':False,
          'origin':'Authenticated IEX venue prints/quotes via Alpaca Basic',
          'market_scope':'IEX_EXCHANGE_ONLY_NOT_ALL_US_TRADES',
          'not_consolidated':True,'regular_session_only':True,
          'adjusted':False, 'feed':'iex',
          'history_pages':pages,'received_bar_count':len(rows),
          'accepted_completed_regular_bars':len(bars),
          'last_bar_start_utc':bars[-1].ts.astimezone(timezone.utc).isoformat(),
          'live_verification':evidence,
          'provider_documentation':'https://docs.alpaca.markets/us/docs/about-market-data-api',
          'warning':'IEX-only is a small part of US volume, not the consolidated market/NBBO. Not suitable for guaranteeing broker execution prices.'}
    return bars,meta


def fetch_public_history(symbol, now=None):
    bars,meta=fetch_yahoo(symbol,'5m','60d')
    asof=(now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    meta['synthetic']=False
    meta['market_scope']='PUBLIC_MARKET_CHART_NOT_VERIFIED_REALTIME'
    meta['live_verification']={
        'status':'HISTORICAL_ONLY','asof_checked_utc':asof,
        'blockers':['PUBLIC_YAHOO_NOT_VERIFIED_REALTIME_OR_CONSOLIDATED'],
        'market_coverage':'UNKNOWN',
        'note':'Actual publicly sourced historical quotes; availability, latency and completeness not guaranteed.'}
    return bars,meta


def fetch_free_market(symbol, source='auto', days=55, history_only=False):
    if source not in ('auto','alpaca','yahoo'):
        raise ValueError('source must be auto, alpaca or yahoo')
    if source=='auto':
        has_key=bool(os.environ.get('ALPACA_API_KEY_ID'))
        has_secret=bool(os.environ.get('ALPACA_API_SECRET_KEY'))
        if has_key!=has_secret:
            raise RuntimeError('Only one Alpaca credential present. Supply both or explicitly use --source yahoo.')
        source='alpaca' if has_key else 'yahoo'
    if source=='alpaca':
        return fetch_alpaca_iex(symbol,days=days,history_only=history_only)
    return fetch_public_history(symbol)

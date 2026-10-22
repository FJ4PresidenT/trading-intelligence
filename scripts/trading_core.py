"""Paper-only trading analytics. No live order placement.

Use closed 5-minute bars, confirmation at close, next-bar fills in backtests.
Data source must be recorded. Synthetic data must never be presented as real.
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
BAR_MINUTES = 5
OR_START = time(9, 30)
OR_END = time(9, 45)
LAST_ENTRY = time(11, 30)
FLAT_TIME = time(15, 55)


@dataclass(frozen=True)
class Bar:
    ts: datetime  # start timestamp of the closed candle, NY timezone
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def day(self):
        return self.ts.date()


def validate_bars(bars):
    if not bars:
        raise ValueError('No bars. Provide 5-minute OHLCV data.')
    times = set()
    for bar in bars:
        if bar.ts.tzinfo is None:
            raise ValueError('Bar timestamp must have a timezone offset.')
        if bar.ts in times:
            raise ValueError('Duplicate timestamp in input.')
        times.add(bar.ts)
        if bar.low <= 0 or bar.high < bar.low or bar.open < bar.low - 1e-6 or bar.close < bar.low - 1e-6:
            raise ValueError('Invalid OHLC data.')
        if bar.open > bar.high + 1e-6 or bar.close > bar.high + 1e-6 or bar.volume < 0:
            raise ValueError('Invalid OHLCV data.')
    return sorted(bars, key=lambda x: x.ts)


def read_csv(filepath):
    bars = []
    with open(filepath, newline='', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            ts = datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00'))
            if ts.tzinfo is None:
                raise ValueError('CSV timestamps require UTC offset, e.g. -04:00 or Z.')
            ts = ts.astimezone(ET)
            bars.append(Bar(ts, *(float(r[k]) for k in ('open', 'high', 'low', 'close', 'volume'))))
    return validate_bars(bars)


def write_csv(filepath, bars):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for b in bars:
            w.writerow([b.ts.isoformat(), *(round(getattr(b, k), 5) for k in ('open', 'high', 'low', 'close', 'volume'))])


def fetch_yahoo(symbol, interval='5m', lookback='60d', timeout=12):
    """Public, potentially delayed Yahoo chart endpoint; not a guaranteed market feed."""
    if not re.fullmatch(r'[A-Za-z0-9.^-]{1,18}', symbol):
        raise ValueError('Invalid symbol.')
    if interval not in ('5m', '1d') or lookback not in ('5d', '1mo', '60d', '3mo', '6mo', '1y'):
        raise ValueError('Unsupported interval or lookback.')
    url = ('https://query1.finance.yahoo.com/v8/finance/chart/' + urllib.parse.quote(symbol) +
           '?' + urllib.parse.urlencode({'interval': interval, 'range': lookback, 'includePrePost': 'false'}))
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 TradingResearch/1.0', 'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            doc = json.loads(res.read(25_000_000).decode('utf-8'))
    except Exception as exc:
        raise RuntimeError('Market data unavailable: ' + str(exc)) from exc
    if doc.get('chart', {}).get('error'):
        raise RuntimeError(str(doc['chart']['error']))
    results = doc['chart']['result'] or []
    if not results:
        raise ValueError('No Yahoo data for symbol.')
    result = results[0]
    q = result['indicators']['quote'][0]
    now = datetime.now(timezone.utc)
    out = []
    for i, ts in enumerate(result.get('timestamp', [])):
        values = [q.get(k, [None]*len(result['timestamp']))[i] for k in ('open','high','low','close','volume')]
        if any(v is None for v in values):
            continue
        start_utc = datetime.fromtimestamp(ts, timezone.utc)
        if interval == '5m' and now < start_utc + timedelta(minutes=BAR_MINUTES, seconds=20):
            continue  # Ignore possibly unfinished live candle.
        local = start_utc.astimezone(ET)
        if interval == '5m' and not (time(9, 30) <= local.time() < time(16, 0)):
            continue
        out.append(Bar(local, *map(float, values)))
    return validate_bars(out), {'provider': 'Yahoo Finance public charts', 'symbol': symbol.upper(),
                                'retrieved_at_utc': now.isoformat(), 'interval': interval,
                                'source_url': url, 'may_be_delayed': True,
                                'currency': result.get('meta', {}).get('currency', 'unknown'),
                                'synthetic': False}


def group_days(bars):
    result = defaultdict(list)
    for b in bars:
        result[b.day].append(b)
    return dict(sorted(result.items()))


def ema(values, length):
    a = 2 / (length + 1)
    result = []
    val = None
    for x in values:
        val = x if val is None else a * x + (1 - a) * val
        result.append(val)
    return result


def rsi(values, length=14):
    if len(values) <= length:
        return None
    diffs = [values[i] - values[i-1] for i in range(1, len(values))]
    ups = [max(x, 0.0) for x in diffs]
    downs = [-min(x, 0.0) for x in diffs]
    avg_up, avg_down = sum(ups[:length]) / length, sum(downs[:length]) / length
    for i in range(length, len(ups)):
        avg_up = (avg_up * (length - 1) + ups[i]) / length
        avg_down = (avg_down * (length - 1) + downs[i]) / length
    if avg_down == 0:
        return 100.0 if avg_up else 50.0
    rs = avg_up / avg_down
    return 100.0 - 100.0 / (1.0 + rs)


def atr(bars, length=14):
    if len(bars) < length + 1:
        return None
    tr = [max(b.high - b.low, abs(b.high - bars[i-1].close), abs(b.low - bars[i-1].close))
          for i, b in enumerate(bars) if i > 0]
    return sum(tr[-length:]) / length


def vwap(today):
    value = sum(b.close * b.volume for b in today)
    volume = sum(b.volume for b in today)
    return value / volume if volume else None


def feature_rows(bars):
    """Build strictly backward-looking technical features in O(n) time."""
    if not bars:
        return []
    closes = [b.close for b in bars]
    ema20, ema50 = ema(closes, 20), ema(closes, 50)
    vol_prefix = [0.0]
    for b in bars:
        vol_prefix.append(vol_prefix[-1]+b.volume)
    rsi_vals = [None]*len(bars)
    atr_vals = [None]*len(bars)
    changes = [closes[i]-closes[i-1] for i in range(1,len(closes))]
    if len(changes) >= 14:
        up = sum(max(0,z) for z in changes[:14])/14
        down = sum(max(0,-z) for z in changes[:14])/14
        for i in range(14,len(bars)):
            if i>14:
                up=(up*13+max(0,changes[i-1]))/14
                down=(down*13+max(0,-changes[i-1]))/14
            rsi_vals[i]=100-100/(1+up/down) if down else (100 if up else 50)
    tr=[0]
    for i in range(1,len(bars)):
        b=bars[i]; prev=bars[i-1]
        tr.append(max(b.high-b.low,abs(b.high-prev.close),abs(b.low-prev.close)))
    trp=[0.0]
    for value in tr: trp.append(trp[-1]+value)
    rows = []
    session_day = None
    cum_vol = 0.0; cum_value = 0.0
    for i,b in enumerate(bars):
        if b.day != session_day:
            session_day=b.day; cum_vol=0.0; cum_value=0.0
        cum_vol+=b.volume; cum_value+=b.volume*b.close
        if i<50:
            rows.append(None);continue
        vwap_value=cum_value/cum_vol if cum_vol else b.close
        avgprev=(vol_prefix[i]-vol_prefix[max(i-20,0)])/min(i,20)
        relative_vol=b.volume/max(1,avgprev)
        travg=(trp[i+1]-trp[i-13])/14 if i>=14 else b.high-b.low
        rows.append([b.close/ema20[i]-1,ema20[i]/ema50[i]-1,
                     (rsi_vals[i] or 50)/100,b.close/vwap_value-1,
                     (b.high-b.low)/b.close,min(relative_vol,20),travg/b.close,
                     b.close/closes[i-1]-1,b.close/closes[i-3]-1,
                     (b.ts.hour*60+b.ts.minute-570)/390])
    return rows


def features(bars, idx):
    return feature_rows(bars[:idx+1])[-1] if idx>=0 else None


def ml_evidence(bars, horizon=6):
    """Chronological holdout and an out-of-sample probability estimate.

    The classifier is research-only: no confidence calibration guaranteed.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import brier_score_loss
    except ImportError:
        return {'available': False, 'reason': 'Install scikit-learn for ML validation.'}
    samples = []
    prepared_features = feature_rows(bars)
    for i, bar in enumerate(bars[:-horizon]):
        if bars[i+horizon].day != bar.day:
            continue
        f = prepared_features[i]
        if f is None:
            continue
        future_ret = bars[i+horizon].close / bar.close - 1
        samples.append((bar.day, f, int(future_ret > 0.0025)))
    days = sorted({row[0] for row in samples})
    if len(days) < 18 or len(samples) < 400:
        return {'available': False, 'reason': 'Need >=18 sessions and >=400 labelled examples.'}
    cut = days[max(1, int(len(days)*.7))]
    train = [x for d,x,y in samples if d < cut]
    train_y = [y for d,x,y in samples if d < cut]
    test = [x for d,x,y in samples if d >= cut]
    test_y = [y for d,x,y in samples if d >= cut]
    if len(set(train_y)) < 2 or len(set(test_y)) < 2 or not test:
        return {'available': False, 'reason': 'Insufficient label variation for holdout.'}
    params = dict(n_estimators=60, max_depth=6, min_samples_leaf=12, random_state=17, n_jobs=1)
    model = RandomForestClassifier(**params)
    model.fit(train, train_y)
    p = model.predict_proba(test)[:, list(model.classes_).index(1)]
    brier = brier_score_loss(test_y, p)
    baseline = sum(train_y)/len(train_y)
    baseline_brier = sum((y-baseline)**2 for y in test_y)/len(test_y)
    latest = prepared_features[-1]
    if latest is None:
        return {'available': False, 'reason': 'Not enough history to score last bar.'}
    full = RandomForestClassifier(**params)
    full.fit([x for d,x,y in samples], [y for d,x,y in samples])
    probability = float(full.predict_proba([latest])[0][list(full.classes_).index(1)])
    return {'available': True, 'probability_up_30m': round(probability, 4),
            'prediction_horizon_bars': horizon, 'positive_label': '30m close return > +0.25%',
            'holdout_brier': round(float(brier), 4), 'baseline_holdout_brier': round(baseline_brier, 4),
            'test_samples': len(test), 'train_samples': len(train),
            'train_end': max(d for d,x,y in samples if d < cut).isoformat(),
            'test_start': cut.isoformat(), 'asof_bar': bars[-1].ts.isoformat(),
            'warning': 'Uncalibrated model score; holdout is historical, not proof of future profitability.'}


def orb_event(session, prior_bars=None, min_volume_ratio=0.0, last_entry=LAST_ENTRY):
    """First CONFIRMED 5-minute close above opening range within entry window."""
    opening = [b for b in session if OR_START <= b.ts.time() < OR_END]
    if len(opening) != 3:
        return {'state': 'NO_DATA', 'reason': 'Opening 09:30-09:45 requires three closed 5m bars.'}
    hi, lo = max(x.high for x in opening), min(x.low for x in opening)
    base_vol = statistics.mean([b.volume for b in opening]) or 1
    prev_close = opening[-1].close
    for bar in session:
        if bar.ts.time() < OR_END:
            continue
        if bar.ts.time() > last_entry:
            break
        if bar.close > hi and prev_close <= hi:
            ratio = bar.volume / base_vol
            if ratio >= min_volume_ratio:
                return {'state': 'TRIGGERED', 'confirmed_at': (bar.ts+timedelta(minutes=5)).isoformat(),
                        'bar_start': bar.ts.isoformat(), 'signal_close': round(bar.close, 4),
                        'range_high': round(hi, 4), 'range_low': round(lo, 4),
                        'range_mid': round((hi+lo)/2, 4), 'volume_ratio': round(ratio, 2)}
        prev_close = bar.close
    return {'state': 'WAIT' if session[-1].ts.time() <= last_entry else 'EXPIRED',
            'range_high': round(hi, 4), 'range_low': round(lo, 4),
            'last_close': session[-1].close}


def setup_snapshot(bars, risk_per_trade=0.01, capital=5000, target_r=1.5, max_allocation=.60, min_volume_ratio=0):
    days = group_days(bars)
    session = list(days.values())[-1]
    event = orb_event(session, min_volume_ratio=min_volume_ratio)
    cls = [b.close for b in bars]
    e20, e50 = ema(cls, 20)[-1], ema(cls, 50)[-1]
    technical = {'ema20': round(e20, 4), 'ema50': round(e50, 4),
                 'rsi14': round(rsi(cls), 2) if rsi(cls) is not None else None,
                 'atr14': round(atr(bars), 4) if atr(bars) else None,
                 'vwap_session': round(vwap(session), 4) if vwap(session) else None,
                 'last_close': session[-1].close}
    result = {'asof': session[-1].ts.isoformat(), 'data_days': len(days),
              'signal': event, 'technicals': technical,
              'capital': capital, 'risk_fraction': risk_per_trade}
    if event['state'] == 'TRIGGERED':
        quote = event['signal_close']
        stop = event['range_mid']
        if stop < quote:
            result['plan'] = bracket(quote, stop, capital, risk_per_trade, target_r, max_allocation)
        else:
            result['plan'] = {'allowed': False, 'reason': 'Stop >= entry.'}
    bearish = (technical['rsi14'] is not None and technical['rsi14'] < 45
               and session[-1].close < e20 < e50)
    result['exit_review'] = ('Bearish EMA/RSI regime. Review any existing LONG; not a short-sale order.'
                             if bearish else None)
    # A trigger earlier today is historical evidence, NOT an instruction to buy now.
    # Each candle timestamp represents the candle START; its actionable close is +5m.
    close_time = session[-1].ts + timedelta(minutes=BAR_MINUTES)
    if bearish:
        action, why = 'SELL_REVIEW', 'Bearish RSI and EMA trend; only relevant to existing long holdings.'
    elif event['state'] == 'TRIGGERED':
        confirmed = datetime.fromisoformat(event['confirmed_at'])
        age_bars = max(0, int((close_time - confirmed).total_seconds() // (BAR_MINUTES*60)))
        result['signal_age_bars'] = age_bars
        if age_bars <= 1 and result.get('plan',{}).get('allowed'):
            action, why = 'BUY_CANDIDATE', 'Recent confirmed breakout with hypothetical stop and target; requires manual review.'
        else:
            action, why = 'STALE_SETUP', 'Breakout occurred earlier; current market entry requires a NEW validation.'
    else:
        action, why = 'WAIT', 'No currently confirmed fresh long setup.'
    result['action'] = {'state': action, 'reason': why, 'asof_closed_bar': close_time.isoformat()}
    return result


def bracket(entry, stop, capital=5000, risk_fraction=.01, target_r=1.5, max_allocation=.60,
            slippage_bps=5, fee_per_share=.005):
    if any(not math.isfinite(x) for x in (entry, stop, capital)):
        raise ValueError('Values must be finite.')
    if not (entry > stop > 0 and capital > 0 and 0 < risk_fraction <= .05 and 0 < max_allocation <= 1):
        return {'allowed': False, 'reason': 'Risk, capital or price limits invalid.'}
    actual_entry = entry * (1 + slippage_bps/10000)
    unit_risk = actual_entry - stop + fee_per_share*2
    risk_budget = capital * risk_fraction
    shares = max(0, min(int(risk_budget / unit_risk), int(capital*max_allocation/actual_entry)))
    target = actual_entry + target_r * (actual_entry-stop)
    return {'allowed': shares > 0, 'entry_reference': round(entry, 4),
            'estimated_entry_after_slippage': round(actual_entry, 4),
            'stop': round(stop, 4), 'target': round(target, 4),
            'shares': shares, 'risk_budget': round(risk_budget, 2),
            'estimated_max_planned_loss': round(shares*unit_risk, 2),
            'capital_used': round(shares*actual_entry, 2),
            'target_r': target_r, 'assumption': 'Stop slippage and overnight gaps can exceed the planned loss.'}


def simulate_replays(bars, signal, runs=10000, stop=None, target=None, seed=19, block=3):
    """Time-of-day block bootstrap from PRIOR trading sessions.

    Simulates new future from historical bar returns/ranges; outcomes depend on
    assumptions. Does NOT establish objective chance of success.
    """
    if signal['state'] != 'TRIGGERED':
        raise ValueError('Needs a confirmed ORB event.')
    trigger = datetime.fromisoformat(signal['bar_start']).astimezone(ET)
    historical = [v for k,v in group_days(bars).items() if k < trigger.date()]
    if len(historical) < 10:
        raise ValueError('Need at least ten earlier trading sessions for replay.')
    entry = signal['signal_close']
    if stop is None:
        stop = signal['range_mid']
    if target is None:
        target = entry + 1.5*(entry-stop)
    if not (stop < entry < target):
        raise ValueError('Require stop < entry < target.')
    post_times = [time(h, m) for h in range(9, 16) for m in range(0, 60, 5) if time(h, m) > trigger.time() and time(h, m) <= FLAT_TIME]
    donor = {}
    for tm in post_times:
        donor[tm] = []
        for sess in historical:
            match = next((b for b in sess if b.ts.time() == tm), None)
            if match and match.open > 0:
                donor[tm].append(match)
    if sum(bool(v) for v in donor.values()) < 8:
        raise ValueError('Insufficient history at matching intraday times.')
    rng = random.Random(seed)
    counts = {'target_first': 0, 'stop_first': 0, 'end_of_day': 0}
    results_r = []
    by_session = [{bar.ts.time():bar for bar in sess} for sess in historical]
    chunks = []
    for pos in range(0,len(post_times),block):
        times = post_times[pos:pos+block]
        eligible = [[day[t] for t in times] for day in by_session if all(t in day for t in times)]
        if eligible:
            chunks.append(eligible)
    if not chunks:
        raise ValueError('No matching historical time-of-day blocks.')
    for _ in range(runs):
        px = entry
        outcome = 'end_of_day'
        # Select one donor session per short block, preserving local serial dependence.
        for eligible in chunks:
            for b in rng.choice(eligible):
                h = px * (b.high / b.open)
                l = px * (b.low / b.open)
                next_px = px * (b.close / b.open)
                # Use conservative stop precedence if same 5-minute bar hits both.
                if l <= stop:
                    outcome = 'stop_first'; px = stop; break
                if h >= target:
                    outcome = 'target_first'; px = target; break
                px = next_px
            if outcome != 'end_of_day':
                break
        counts[outcome] += 1
        results_r.append((px-entry)/(entry-stop))
    wins = sum(1 for r in results_r if r > 0)
    return {'method': '3-bar time-of-day historical-session block bootstrap',
            'runs': runs, 'seed': seed, 'historic_sessions': len(historical),
            'entry': round(entry, 4), 'stop': round(stop, 4), 'target': round(target, 4),
            'counts': counts, 'rates_pct': {k: round(v*100/runs, 2) for k,v in counts.items()},
            'average_r_before_costs': round(statistics.mean(results_r), 4),
            'positive_end_equity_pct': round(100*wins/runs, 2),
            'warning': 'Monte Carlo bootstrap is scenario analysis, not a forecast. It resamples a limited set of historic sessions; news/gaps are not modelled.'}


def risk_gate(entry, stop, capital=5000, cash=5000, risk_fraction=.01, target_r=1.5,
              trades_today=0, realized_pnl_today=0, max_trades=3,
              max_daily_loss=.03, at=None, max_allocation=.60):
    """Manual risk-policy check; NEVER an authorization to route an order."""
    if at is None:
        at = datetime.now(ET)
    elif isinstance(at,str):
        at = datetime.fromisoformat(at)
    if at.tzinfo is None:
        raise ValueError('Risk time must contain an offset.')
    at = at.astimezone(ET)
    if not (capital > 0 and cash >= 0 and max_trades >= 1 and 0 < max_daily_loss <= .2):
        raise ValueError('Invalid portfolio or policy parameters.')
    plan = bracket(entry,stop,capital,risk_fraction,target_r,max_allocation)
    blockers = []
    if not (time(9,45) <= at.time() <= LAST_ENTRY and at.weekday() < 5):
        blockers.append('Outside 09:45-11:30 ET weekday entry window (holidays are not checked).')
    if trades_today >= max_trades:
        blockers.append('Daily trade-count limit reached.')
    if realized_pnl_today <= -capital*max_daily_loss:
        blockers.append('Daily realized-loss kill switch triggered.')
    if not plan.get('allowed'):
        blockers.append('Position sizing or stop setup invalid.')
    elif plan['capital_used'] > cash + 0.00001:
        blockers.append('Insufficient available virtual cash.')
    return {'state':'BLOCK' if blockers else 'PASS_RESEARCH_ONLY',
            'blockers':blockers, 'hypothetical_plan':plan,'time_et':at.isoformat(),
            'policy':{'max_trades_today':max_trades,'max_daily_loss_fraction':max_daily_loss,
                      'risk_fraction':risk_fraction,'long_only':True,'no_leverage':True},
            'note':'No broker connectivity. PASS_RESEARCH_ONLY is NOT trade approval.'}


def settle_paper_trade(session, event, capital=5000, risk_fraction=.01, target_r=1.5,
                       max_allocation=.60, slippage_bps=5, fee_per_share=.005):
    """Next bar OPEN fill. No trading at the just-confirmed signal close."""
    if event['state'] != 'TRIGGERED':
        return None
    bar_start = datetime.fromisoformat(event['bar_start'])
    idx = next((i for i,b in enumerate(session) if b.ts == bar_start), None)
    if idx is None or idx+1 >= len(session):
        return None
    entry_bar = session[idx+1]
    fill = entry_bar.open*(1+slippage_bps/10000)
    stop = event['range_mid']
    if fill <= stop or (fill / event['signal_close'] - 1) > .01:
        return None  # Skip gap-through-stop and severe chasing.
    setup = bracket(entry_bar.open, stop, capital, risk_fraction, target_r, max_allocation,
                    slippage_bps, fee_per_share)
    if not setup['allowed']:
        return None
    shares = setup['shares']; target = setup['target']
    exit_px = None; outcome = 'time_exit'; exit_time = None
    for b in session[idx+1:]:
        if b.ts.time() > FLAT_TIME:
            break
        # Adverse opening gap fills at open (minus modeled slippage).
        if b.open <= stop:
            exit_px = b.open*(1-slippage_bps/10000); outcome='gap_stop'; exit_time=b.ts; break
        if b.low <= stop:
            exit_px = stop*(1-slippage_bps/10000); outcome='stop'; exit_time=b.ts; break
        if b.high >= target:
            exit_px = target*(1-slippage_bps/10000); outcome='target'; exit_time=b.ts; break
        exit_px = b.close*(1-slippage_bps/10000); exit_time=b.ts
    if exit_px is None:
        return None
    pnl = shares*(exit_px-fill)-2*shares*fee_per_share
    planned_dollar_r = shares*(fill-stop)
    return {'date': session[0].day.isoformat(), 'confirmed': event['confirmed_at'],
            'entry_at': entry_bar.ts.isoformat(), 'exit_at': exit_time.isoformat(),
            'entry': round(fill, 4), 'exit': round(exit_px, 4), 'stop': round(stop, 4),
            'target': round(target, 4), 'shares': shares, 'outcome': outcome,
            'pnl': round(pnl, 2), 'r': round(pnl/planned_dollar_r, 4) if planned_dollar_r else None}


def backtest(bars, capital=5000, risk_fraction=.01, target_r=1.5, min_volume_ratio=0,
             slippage_bps=5, fee_per_share=.005):
    sessions = group_days(bars)
    equity = float(capital)
    peak = equity
    max_dd = 0
    trades = []
    for d, sess in sessions.items():
        if len(sess) < 25:
            continue
        evt = orb_event(sess, min_volume_ratio=min_volume_ratio)
        result = settle_paper_trade(sess, evt, equity, risk_fraction, target_r,
                                    slippage_bps=slippage_bps, fee_per_share=fee_per_share)
        if result:
            equity += result['pnl']
            peak = max(equity, peak)
            max_dd = max(max_dd, (peak-equity)/peak)
            result['equity'] = round(equity, 2)
            trades.append(result)
    gains = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    losses = -sum(t['pnl'] for t in trades if t['pnl'] < 0)
    win = sum(t['pnl'] > 0 for t in trades)
    return {'sessions': len(sessions), 'n_trades': len(trades), 'starting_equity': capital,
            'ending_equity': round(equity, 2), 'net_return_pct': round(100*(equity/capital-1), 2),
            'win_rate_pct': round(100*win/len(trades), 2) if trades else None,
            'avg_pnl': round(statistics.mean([t['pnl'] for t in trades]), 2) if trades else None,
            'avg_r': round(statistics.mean([t['r'] for t in trades]), 4) if trades else None,
            'profit_factor': round(gains/losses, 3) if losses else None,
            'max_drawdown_pct': round(max_dd*100, 2), 'trades': trades,
            'assumptions': 'Confirmed 5m-close ORB; next bar open entry; worst-case stop if TP/SL touched in same bar; fixed slippage/fees; no borrow/margin; no overnight holding.'}


def walkforward(bars, **kwargs):
    days = sorted(group_days(bars))
    if len(days) < 20:
        return {'available': False, 'reason': 'Need at least 20 sessions.'}
    a, b = max(1,int(len(days)*.6)), max(2,int(len(days)*.8))
    out = {}
    daygroups = group_days(bars)
    for label, ds in [('development',days[:a]), ('validation',days[a:b]), ('holdout',days[b:])]:
        r = backtest([bar for day in ds for bar in daygroups[day]], **kwargs)
        r.pop('trades')
        out[label] = {'first_day': ds[0].isoformat(), 'last_day': ds[-1].isoformat(), **r}
    return {'available': True, 'method': 'Chronological segments with fixed rules, no parameter fitting or lookahead',
            'segments': out, 'warning': 'One fixed strategy; 60d of intraday data is too short to establish robust edge.'}


def generate_synthetic(days=85, seed=19):
    """Deterministic FAKE dataset for integration tests and demo only."""
    rng = random.Random(seed)
    d = date(2026, 4, 1)
    bars = []
    p = 180.0
    n = 0
    while n < days:
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue
        n += 1
        p = max(15.0, p*(1+rng.gauss(0, .006)))
        direction = rng.choice([-1, 1])
        for slot in range(78):
            timestamp = datetime.combine(d, time(9,30), ET) + timedelta(minutes=slot*5)
            op = p
            vol = .0028*(1.6 if slot < 7 or slot > 66 else 1)
            drift = .00012*direction if 3 < slot < 27 else 0
            cl = max(1, op*(1+rng.gauss(drift, vol)))
            wiggle = abs(rng.gauss(0,.0008))
            high = max(op,cl)*(1+wiggle)
            low = min(op,cl)*(1-wiggle)
            amount = max(1,int(rng.lognormvariate(11.3,.6)))
            bars.append(Bar(timestamp,op,high,low,cl,amount))
            p=cl
        d += timedelta(days=1)
    return validate_bars(bars)


def write_json(filepath, obj):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)

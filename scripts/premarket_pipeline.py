#!/usr/bin/env python3
"""Public, fail-closed historical US premarket research, never live trade advice.

Run before the US open. This program fetches *real* public OHLCV; it never
substitutes synthetic/demo data or manufactures missing observations. It writes
JSON and a lightweight mobile HTML page for GitHub Pages.
"""
from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from datetime import datetime, timezone, time, timedelta
from pathlib import Path

from free_market import fetch_free_market
from trading_core import (ET, group_days, setup_snapshot, orb_event, ml_evidence,
                          simulate_replays, backtest, walkforward)

DEFAULT_WATCHLIST = ('SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL')


def prior_completed_session(now):
    """Use XNYS calendar; do not guess trading holidays/early closes."""
    import exchange_calendars as xcals
    import pandas as pd
    local = now.astimezone(ET)
    cal = xcals.get_calendar('XNYS')
    today = pd.Timestamp(local.date())
    sess = cal.date_to_session(today, direction='previous')
    if sess.date() == local.date():
        end = cal.session_close(sess).to_pydatetime().astimezone(ET)
        # End +5m to let session data settle.
        if local < end + timedelta(minutes=5):
            sess = cal.previous_session(sess)
    return sess.date()


def history_quality(bars, metadata, asof, expected_date):
    problems = []
    if metadata.get('synthetic') is not False:
        problems.append('SYNTHETIC_OR_UNKNOWN_SOURCE_REJECTED')
    if metadata.get('live_verification', {}).get('status') != 'HISTORICAL_ONLY':
        problems.append('THIS_PIPELINE_REQUIRES_EXPLICIT_HISTORICAL_ONLY_SOURCE')
    if metadata.get('provider') != 'Yahoo Finance public charts':
        problems.append('UNEXPECTED_PROVIDER')
    if len(bars) < 1000 or len(group_days(bars)) < 20:
        problems.append('INSUFFICIENT_HISTORICAL_5M_BARS')
    if any(bar.ts.astimezone(timezone.utc) > asof for bar in bars):
        problems.append('BAR_TIMESTAMP_IN_FUTURE')
    if not bars or bars[-1].day != expected_date:
        problems.append('LAST_TRADING_SESSION_MISSING_OR_STALE')
    if bars and bars[-1].ts.astimezone(ET).time() < time(15, 45):
        # Reject incomplete sessions; on early closes this will conservatively block.
        problems.append('LAST_SESSION_NOT_NEAR_REGULAR_CLOSE_OR_EARLY_CLOSE')
    return problems


def replay_last_historical_trigger(bars, runs):
    """Only use data and donor sessions PRIOR TO the historical trigger."""
    days = group_days(bars)
    dates = sorted(days)
    if len(dates) < 13:
        return {'available': False, 'reason': 'Insufficient earlier real sessions for bootstrap.'}
    for k in reversed(range(11, len(dates))):
        sess = days[dates[k]]
        if len(sess) < 25:
            continue
        evt = orb_event(sess)
        if evt['state'] != 'TRIGGERED' or not evt['range_mid'] < evt['signal_close']:
            continue
        bars_until = [b for d in dates[:k + 1] for b in days[d]]
        try:
            res = simulate_replays(bars_until, evt, runs=runs)
            return {'available': True, 'historical_trigger_date': dates[k].isoformat(),
                    'source_scope': 'HISTORICAL_PRIOR_SESSIONS_ONLY', **res}
        except ValueError:
            continue
    return {'available': False, 'reason': 'No eligible historical ORB trigger with sufficient prior sessions.'}


def analyze_symbol(symbol, now, expected_date, runs=10000, fetcher=fetch_free_market):
    symbol = symbol.strip().upper()
    try:
        bars, meta = fetcher(symbol, source='yahoo', history_only=True)
        issues = history_quality(bars, meta, now, expected_date)
        info = {
            'ticker': symbol,
            'status': 'BLOCKED' if issues else 'HISTORICAL_RESEARCH',
            'blockers': issues,
            'source': meta.get('provider', 'UNKNOWN'),
            'source_url': meta.get('source_url'),
            'retrieved_at_utc': meta.get('retrieved_at_utc'),
            'last_bar_at_ny': bars[-1].ts.isoformat() if bars else None,
            'last_observed_close_usd': bars[-1].close if bars else None,
            'bar_count': len(bars),
            'session_count': len(group_days(bars)),
            'market_data_status': 'HISTORICAL_ONLY_NOT_VERIFIED_LIVE',
            'current_market_entry_authorized': False,
        }
        if issues:
            return info
        snap = setup_snapshot(bars)
        # Training/testing and bootstraps are historical and NOT a forward forecast.
        ml = ml_evidence(bars)
        bt = backtest(bars)
        bt.pop('trades', None)
        wf = walkforward(bars)
        replay = replay_last_historical_trigger(bars, runs)
        holdout = (wf.get('segments') or {}).get('holdout', {})
        # Extra strict gate to avoid selecting on a tiny backtest.
        is_candidate = bool(holdout.get('n_trades', 0) >= 3
                            and holdout.get('avg_r') is not None
                            and holdout['avg_r'] > 0
                            and holdout.get('net_return_pct', 0) > 0
                            and ml.get('available')
                            and ml.get('holdout_brier', 1) < ml.get('baseline_holdout_brier', 0)
                            and replay.get('available'))
        info.update({
            'previous_session_orb_state': snap['signal']['state'],
            'previous_session_technical_snapshot': snap['technicals'],
            'ml_historical_holdout': ml,
            'backtest': bt,
            'walkforward': wf,
            'replay': replay,
            'historical_screen_pass': is_candidate,
            'research_rank_score': (round(holdout.get('avg_r', 0) * min(holdout.get('n_trades', 0), 10), 4)
                                    if is_candidate else None),
            'next_step': ('Beobachten, erst nach dem Marktstart neu prüfen; KEIN heutiges Kaufsignal.'
                          if is_candidate else 'Keine historisch ausreichend bestätigte Strategie; beobachten/meiden.'),
        })
        return info
    except Exception as ex:
        return {'ticker': symbol, 'status': 'ERROR', 'blockers': ['DATA_OR_COMPUTE_FAILURE'],
                'error': str(ex)[:240], 'historical_screen_pass': False,
                'current_market_entry_authorized': False}


def make_payload(symbols=DEFAULT_WATCHLIST, now=None, runs=10000, fetcher=fetch_free_market):
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expected = prior_completed_session(now)
    results = [analyze_symbol(t, now, expected, runs, fetcher) for t in symbols]
    candidates = [v for v in results if v.get('historical_screen_pass')]
    candidates.sort(key=lambda x: x['research_rank_score'], reverse=True)
    return {
        'schema_version': 1,
        'report_type': 'PREMARKET_HISTORICAL_RESEARCH_NOT_LIVE_TRADING',
        'generated_at_utc': now.isoformat(),
        'intended_for_market_date_ny': now.astimezone(ET).date().isoformat(),
        'expected_completed_session_ny': expected.isoformat(),
        'provider': 'Yahoo Finance public historical 5-minute chart endpoint (unofficial access; availability and licensing restrictions may apply)',
        'real_time_verified': False,
        'market_coverage_verified_consolidated': False,
        'stock_selection_not_an_order': True,
        'python_computations_executed': True,
        'watchlist': list(symbols),
        'instruments': results,
        'historical_watch_candidates': [v['ticker'] for v in candidates[:3]],
        'warning': ('Keine Live-Marktsignale. Die Daten können verzögert, korrigiert oder unvollständig sein. '
                    'Der historische ORB-Test nutzt Positionen innerhalb eines Handelstags (keine Swing-Trading-Validierung). '
                    '10.000 Replays sind nur Stichproben aus vergangenen Sitzungen und keine heutige Gewinnwahrscheinlichkeit. '
                    'Am Börsenstart müssen Preise, Volumen, Marktstatus und Ereignisse NEU geprüft werden.'),
    }


def render_html(payload):
    e = lambda obj: html.escape(str(obj), quote=True)
    items=[]
    for item in payload['instruments']:
        ticker=e(item['ticker']); cls=e(item['status'])
        p=item.get('last_observed_close_usd'); last=f'{p:,.2f} USD' if isinstance(p,(float,int)) else '--'
        oos=(item.get('walkforward') or {}).get('segments',{}).get('holdout', {})
        orate=f"{oos['net_return_pct']}%" if 'net_return_pct' in oos else '--'
        count=oos.get('n_trades','--')
        replay=item.get('replay') or {}
        replaystr=str(replay.get('runs', '--')) if replay.get('available') else 'nicht verfügbar'
        blockers=', '.join(item.get('blockers',[])) or 'keine Quellenblocker (historische Daten)'
        items.append(f'<tr><th>{ticker}</th><td>{e(last)}</td><td>{e(cls)}</td>'
                     f'<td>{e(count)}</td><td>{e(orate)}</td><td>{e(replaystr)}</td><td>{e(blockers)}</td></tr>')
    picks = ', '.join(payload['historical_watch_candidates']) or 'Keine Kandidaten bestanden die historischen Prüfungen.'
    css=('body{font:15px system-ui,-apple-system,sans-serif;max-width:1050px;margin:0 auto;padding:22px;background:#101722;color:#e9f1fc}'
         'h1{font-size:26px}p{line-height:1.6;color:#c0cfdd}.box{padding:16px;border:1px solid #6e7656;border-radius:10px;background:#242a20}'
         '.table{overflow-x:auto}table{width:100%;border-collapse:collapse;min-width:750px}th,td{padding:11px 9px;text-align:left;border-bottom:1px solid #394859}'
         'th{font-weight:650}small{color:#9dacc1}a{color:#8ccaff}')
    return f'''<!DOCTYPE html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Trading Intelligence | Historischer Vorbörse-Check</title><style>{css}</style></head><body>
    <h1>Trading Intelligence – Vorbörse-Check</h1>
    <p><strong>Analysezeit:</strong> {e(payload['generated_at_utc'])} (UTC)<br><strong>Letzte erwartete abgeschlossene Sitzung:</strong> {e(payload['expected_completed_session_ny'])}</p>
    <div class="box"><strong>KEINE LIVE-KAUFSIGNALE</strong><p>{e(payload['warning'])}</p><strong>Historisch interessante Beobachtungskandidaten:</strong> {e(picks)}</div>
    <h2>Historische Analyse</h2><div class="table"><table><thead><tr><th>Ticker</th><th>Historischer Schluss</th><th>Prüfstatus</th><th>Trades OOS</th><th>OOS-Ergebnis</th><th>Simulationen</th><th>Blocker</th></tr></thead><tbody>{''.join(items)}</tbody></table></div>
    <p><small>OOS = chronologisch zurückgehaltener Zeitraum. Ergebnisse beruhen auf wenigen Handelstagen und können zufällig sein. Historischer ORB-Test = Daytrading-Strategie, keine Swing-Trading-Strategie.</small></p>
    <p><a href="latest.json">Vollständiger JSON-Bericht samt Datenquellen und Modellmetriken</a></p></body></html>'''


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--tickers', default=','.join(DEFAULT_WATCHLIST))
    cli.add_argument('--output-dir', default='docs')
    cli.add_argument('--replays', type=int, default=10000)
    args = cli.parse_args()
    if not 100 <= args.replays <= 100000:
        cli.error('replays must be 100..100000')
    tickers = list(dict.fromkeys(s.strip().upper() for s in args.tickers.split(',') if s.strip()))
    if not 5 <= len(tickers) <= 12:
        cli.error('Analyze between five and twelve symbols.')
    report = make_payload(tickers, runs=args.replays)
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'latest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    (dest/'index.html').write_text(render_html(report), encoding='utf-8')
    (dest/'.nojekyll').write_text('', encoding='utf-8')
    print(json.dumps({'generated':report['generated_at_utc'], 'expected_prior_session':report['expected_completed_session_ny'],
        'candidate_count':len(report['historical_watch_candidates']), 'ok':sum(v['status']=='HISTORICAL_RESEARCH' for v in report['instruments']),
        'blocked_or_error':sum(v['status']!='HISTORICAL_RESEARCH' for v in report['instruments']),
        'location':str(dest)},ensure_ascii=False))
    if not any(v['status']=='HISTORICAL_RESEARCH' for v in report['instruments']):
        # Publish evidence of failure rather than old successes, but fail the job.
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())

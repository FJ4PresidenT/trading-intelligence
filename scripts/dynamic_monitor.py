#!/usr/bin/env python3
"""Free historical/delayed data WATCH monitor. Never represents Yahoo as real-time.
Outputs JSON for each run, candidate lists rotate based on prior full session.
"""
import json, os, math, statistics, urllib.request, urllib.parse
from datetime import datetime, timezone, time, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

ET=ZoneInfo('America/New_York')
UNIVERSE='SPY QQQ IWM DIA AAPL MSFT NVDA AMZN GOOGL META TSLA AMD AVGO NFLX COST JPM BAC XOM CVX UNH WMT HD CRM ORCL PLTR'.split()
URL='https://query1.finance.yahoo.com/v8/finance/chart/'

def fetch(symbol):
    params=urllib.parse.urlencode({'interval':'5m','range':'5d','includePrePost':'false'})
    req=urllib.request.Request(URL+symbol+'?'+params,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=12) as resp: obj=json.load(resp)
    result=obj['chart']['result'][0]
    q=result['indicators']['quote'][0]
    rows=[]
    for i,ts in enumerate(result['timestamp']):
        try:
            dt=datetime.fromtimestamp(ts,timezone.utc).astimezone(ET)
            if not time(9,30)<=dt.time()<time(16,0): continue
            o,h,l,c,v=[q[k][i] for k in ('open','high','low','close','volume')]
            if not all(isinstance(x,(float,int)) and math.isfinite(x) for x in (o,h,l,c,v)):continue
            if not 0<l<=min(o,c)<=max(o,c)<=h or v<0:continue
            rows.append({'ts':dt, 'o':o,'h':h,'l':l,'c':c,'v':v})
        except (IndexError,TypeError,ValueError,KeyError):continue
    return rows

def build(now, fetcher=fetch):
    now=now.astimezone(ET)
    day=now.date(); found=[]; errors={}
    for sym in UNIVERSE:
        try:
            rows=fetcher(sym)
            by_date={}
            for b in rows:
                if b['ts'].astimezone(timezone.utc)+timedelta(minutes=6)<=now.astimezone(timezone.utc):
                    by_date.setdefault(b['ts'].date(),[]).append(b)
            prior=sorted(d for d in by_date if d<day)
            if not prior: raise ValueError('No prior completed session')
            bars=by_date[prior[-1]]
            if len(bars)<65 or bars[-1]['ts'].time()<time(15,45):raise ValueError('Incomplete prior day')
            close=bars[-1]['c']; vol=sum(b['v'] for b in bars)
            prev=by_date[prior[-2]] if len(prior)>1 else []
            ret=(close/prev[-1]['c']-1)*100 if prev else 0
            rng=(max(b['h'] for b in bars)-min(b['l'] for b in bars))/close*100
            # Rank ONLY using previous session: not today, avoiding look-ahead and unstable picks.
            score=round(min(abs(ret),8)*2+min(rng,10)+min(math.log10(max(vol*close,1))/2,7),3)
            today=by_date.get(day,[])
            signal={'status':'NO_CONFIRMED_LIVE_SIGNAL', 'reason':'Public Yahoo chart cannot independently verify live consolidated market quote'}
            # Observational pattern requires at least four FULL 5m candles.
            if len(today)>=4:
                opening=today[:3]; recent=today[-1]
                upper=max(b['h'] for b in opening)
                baseline=statistics.median(b['v'] for b in today[:-1]) if len(today)>4 else 0
                if recent['c']>upper and recent['c']>recent['o'] and baseline>0 and recent['v']>=1.5*baseline:
                    signal={'status':'HISTORICAL_OR_DELAYED_BREAKOUT_WATCH_ONLY',
                            'reason':'Close above first 15min range, recent bar volume >=1.5x median; NOT verified real-time',
                            'reference_breakout_usd':round(upper,3),
                            'reference_stop_usd':round(min(b['l'] for b in today[-3:]),3),
                            'bar_timestamp_ny':recent['ts'].isoformat()}
            found.append({'symbol':sym,'prior_session':str(prior[-1]),'prior_close_usd':round(close,3),
                          'prior_return_pct':round(ret,3),'prior_range_pct':round(rng,3),
                          'prior_volume':int(vol),'ranking_score':score,
                          'latest_completed_bar_ny':today[-1]['ts'].isoformat() if today else None,
                          'signal':signal})
        except Exception as exc:errors[sym]=str(exc)[:150]
    found.sort(key=lambda x:x['ranking_score'],reverse=True)
    selected=found[:8]
    return {'schema_version':2,'report_type':'DYNAMIC_WATCHLIST_RESEARCH_NOT_LIVE_SIGNALS',
            'generated_at_utc':now.astimezone(timezone.utc).isoformat(),
            'market_date_ny':str(day),'provider':'Yahoo public chart (unofficial; potentially delayed/incomplete)',
            'live_market_verified':False,'selection_method':'prior-day liquidity/return/range among fixed 25-symbol universe',
            'universe_size':len(UNIVERSE),'selected':selected,'errors':errors,
            'warning':'Research-only. No verified live/consolidated quotes. No 100% coverage. No profits guaranteed.'}

def main():
    now=datetime.now(timezone.utc)
    import exchange_calendars as ecals
    import pandas as pd
    cal=ecals.get_calendar('XNYS'); local=now.astimezone(ET)
    # Reject weekends and exchange holidays, avoid spurious alerts.
    if not cal.is_session(pd.Timestamp(local.date())):
        out={'status':'MARKET_CLOSED','generated_at_utc':now.isoformat(), 'selected':[]}
    else:out=build(now)
    path=Path('data/monitor.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('status',out.get('status',out.get('report_type')),'count',len(out.get('selected',[])))
    for item in out.get('selected',[]):print(item['symbol'], item['signal']['status'])
if __name__=='__main__':main()

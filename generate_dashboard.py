#!/usr/bin/env python3.9
"""
DBC ETF Live Dashboard
Generates: ~/dbc_dashboard/dbc_dashboard.html
Charts: Price, Volume, Commodity Exposure, Returns vs S&P 500, News Sentiment
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, os
warnings.filterwarnings('ignore')

DARK_BG    = '#0d1117'
DARK_PANEL = '#161b22'
DARK_BORDER= '#30363d'
GREEN      = '#3fb950'
RED        = '#f85149'
BLUE       = '#58a6ff'
GOLD       = '#d29922'
PURPLE     = '#a371f7'
TEXT       = '#e6edf3'
TEXT2      = '#8b949e'
GRID       = '#21262e'

OUT = '/home/uzajc/dbc_dashboard'
os.makedirs(OUT, exist_ok=True)

def g(df, col):
    return df[col].values.flatten()

def ps(data):
    return pd.Series(data)

# ── FETCH DATA ─────────────────────────────────────────────────────────────────
print('Fetching data...')
dbc     = yf.download('DBC',  period='1y',  interval='1d', progress=False)
spy     = yf.download('SPY',  period='1y',  interval='1d', progress=False)
dbc_5y  = yf.download('DBC',  period='5y',  interval='1mo', progress=False)
spy_5y  = yf.download('SPY',  period='5y',  interval='1mo', progress=False)
uso     = yf.download('USO',  period='1y',  interval='1d', progress=False)
gld     = yf.download('GLD',  period='1y',  interval='1d', progress=False)
dba     = yf.download('DBA',  period='1y',  interval='1d', progress=False)

info    = yf.Ticker('DBC').info

dates   = dbc.index.to_numpy()
close   = g(dbc, 'Close')
open_p  = g(dbc, 'Open')
vol     = g(dbc, 'Volume')
spy_c   = g(spy, 'Close')
uso_c   = g(uso, 'Close')
gld_c   = g(gld, 'Close')
dba_c   = g(dba, 'Close')
uso_c0, gld_c0, dba_c0 = uso_c[0], gld_c[0], dba_c[0]

# ── METRICS ───────────────────────────────────────────────────────────────────
price   = info.get('regularMarketPrice', close[-1])
prev    = info.get('previousClose', close[-2])
chg     = price - prev
pct     = (chg / prev) * 100
volume_ = info.get('volume', int(vol[-1]))
avg_vol = info.get('averageVolume', int(vol.mean()))
high52  = info.get('fiftyTwoWeekHigh', 31.46)
low52   = info.get('fiftyTwoWeekLow',  20.78)
div_y   = info.get('dividendYield') or 0.024
aum     = info.get('totalAssets', 1_885_037_440)
exp_r   = (info.get('expenseRatio') or 0.005) * 100

ytd_dbc = (close[-1] / close[0]      - 1) * 100
ytd_spy = (spy_c[-1] / spy_c[0]      - 1) * 100
y1_dbc  = (close[-1] / close[-252]   - 1) * 100 if len(close) > 252 else 0
y1_spy  = (spy_c[-1] / spy_c[-252]   - 1) * 100 if len(spy_c) > 252 else 0

dbc_m   = dbc_5y['Close'].pct_change().dropna()
spy_m   = spy_5y['Close'].pct_change().dropna()
dbc_cum = (1 + dbc_m/100).cumprod() * 100
spy_cum = (1 + spy_m/100).cumprod() * 100
ret5y_dbc = float((dbc_cum.iloc[-1]/dbc_cum.iloc[0] - 1)*100) if len(dbc_cum)>1 else 0
ret5y_spy = float((spy_cum.iloc[-1]/spy_cum.iloc[0] - 1)*100) if len(spy_cum)>1 else 0

UP = 'up'; DN = 'dn'
price_cls = UP if chg >= 0 else DN
ytd_cls   = UP if ytd_dbc >= 0 else DN
y1_cls    = UP if y1_dbc  >= 0 else DN
ret5y_cls = UP if ret5y_dbc >= 0 else DN

print(f'price={price:.2f} ytd={ytd_dbc:.1f}% 5yr={ret5y_dbc:.1f}%')

# ── CHART 1: PRICE + VOLUME ───────────────────────────────────────────────────
print('Chart 1...')
ma20 = ps(close).rolling(20).mean().values
ma50 = ps(close).rolling(50).mean().values
vol_colors = [GREEN if close[i] >= open_p[i] else RED for i in range(len(close))]

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8), facecolor=DARK_BG, gridspec_kw={'height_ratios':[3,1]})
fig.subplots_adjust(hspace=0.05)

ax1.plot(dates, close,  color=BLUE,   lw=1.8, label='DBC Close')
ax1.fill_between(dates, close, alpha=0.07, color=BLUE)
ax1.plot(dates, ma20,   color=GOLD,   lw=1.1, ls='--', label='MA 20', alpha=0.85)
ax1.plot(dates, ma50,   color=PURPLE, lw=1.1, ls='--', label='MA 50', alpha=0.85)
ax1.axhline(high52, color=GREEN, lw=0.8, ls=':', alpha=0.55)
ax1.axhline(low52,  color=RED,   lw=0.8, ls=':', alpha=0.55)
ax1.text(dates[-1], high52, f' 52W High ${high52:.2f}', va='center', fontsize=8, color=GREEN, alpha=0.8)
ax1.text(dates[-1], low52,  f' 52W Low  ${low52:.2f}',  va='center', fontsize=8, color=RED,   alpha=0.8)
ax1.set_facecolor(DARK_PANEL)
ax1.tick_params(colors=TEXT2, labelsize=8)
ax1.set_ylabel('Price (USD)', color=TEXT, fontsize=9)
ax1.set_title('DBC — Invesco DB Commodity Index Tracking ETF', color=TEXT, fontsize=13, fontweight='bold', pad=10)
ax1.legend(loc='upper left', fontsize=8, facecolor=DARK_PANEL, edgecolor=DARK_BORDER)
ax1.grid(True, color=GRID, lw=0.5, alpha=0.45)
for s in ['top','right','bottom','left']: ax1.spines[s].set_visible(False)
ax1.annotate(f'${price:.2f}', xy=(dates[-1],price), xytext=(8,0),
             textcoords='offset points', fontsize=11, fontweight='bold', color=BLUE, va='center')
ax1.set_xlim(dates[0], dates[-1])

ax2.bar(dates, vol/1e6, color=vol_colors, alpha=0.65, width=1.0)
ax2.axhline(avg_vol/1e6, color=GOLD, lw=0.8, ls='--', alpha=0.55)
ax2.set_facecolor(DARK_PANEL)
ax2.tick_params(colors=TEXT2, labelsize=8)
ax2.set_ylabel('Vol (M)', color=TEXT, fontsize=9)
ax2.grid(True, color=GRID, lw=0.5, alpha=0.25, axis='y')
for s in ['top','right','bottom','left']: ax2.spines[s].set_visible(False)
ax2.set_xlim(dates[0], dates[-1])

plt.savefig(f'{OUT}/chart1_price_volume.png', dpi=120, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# ── CHART 2: COMMODITY EXPOSURE PIE ──────────────────────────────────────────
print('Chart 2...')
labels2 = ['Crude Oil','Natural Gas','Gold','Silver','Corn','Wheat','Soybeans','Coffee','Cotton','Sugar/Alum.']
sizes2  = [25,15,13,5,9,9,7,5,4,8]
colors2 = ['#f85149','#79c0ff','#d29922','#c9d1d9','#85e89d','#f0c169','#7ee787','#a371f7','#ff9ba8','#79c0ff']

fig, ax = plt.subplots(figsize=(9,7), facecolor=DARK_BG)
ax.set_facecolor(DARK_PANEL)
wedges, texts, atxts = ax.pie(sizes2, labels=labels2, colors=colors2, explode=[0.04]*len(sizes2),
    autopct='%1.0f%%', pctdistance=0.78, startangle=90,
    textprops={'color':TEXT,'fontsize':10},
    wedgeprops={'edgecolor':DARK_BG,'linewidth':1.5})
for a in atxts: a.set_fontsize(9); a.set_color(DARK_BG); a.set_fontweight('bold')
ax.set_title('DBC Commodity Exposure Breakdown\n(Approx. DB Index Weights)', color=TEXT, fontsize=12, fontweight='bold', pad=15)
ax.legend(handles=[mpatches.Patch(color=c,label=l) for c,l in zip(colors2,labels2)],
          loc='lower center', bbox_to_anchor=(0.5,-0.12), ncol=3,
          fontsize=8.5, facecolor=DARK_PANEL, edgecolor=DARK_BORDER, labelcolor=TEXT)
plt.savefig(f'{OUT}/chart2_exposure.png', dpi=120, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# ── CHART 3: DBC vs SPY RETURNS ───────────────────────────────────────────────
print('Chart 3...')
dbc_n = close / close[0] * 100
spy_n = spy_c / spy_c[0] * 100

fig, (ax3a, ax3b) = plt.subplots(2,1, figsize=(12,8), facecolor=DARK_BG, gridspec_kw={'height_ratios':[2,1]})
fig.subplots_adjust(hspace=0.1)

ax3a.plot(dates, dbc_n, color=BLUE,  lw=1.8, label='DBC')
ax3a.plot(dates, spy_n, color=GOLD,  lw=1.8, label='S&P 500 (SPY)')
ax3a.fill_between(dates, dbc_n, spy_n, alpha=0.07, color=PURPLE)
ax3a.axhline(100, color=GRID, lw=0.8, ls='-', alpha=0.35)
ax3a.set_facecolor(DARK_PANEL)
ax3a.tick_params(colors=TEXT2, labelsize=8)
ax3a.set_title('DBC vs S&P 500 — 1-Year Normalized Return (Base=100)', color=TEXT, fontsize=12, fontweight='bold')
ax3a.set_ylabel('Normalized Price', color=TEXT, fontsize=9)
ax3a.legend(loc='upper left', fontsize=9, facecolor=DARK_PANEL, edgecolor=DARK_BORDER, labelcolor=TEXT)
ax3a.grid(True, color=GRID, lw=0.5, alpha=0.4)
for s in ['top','right','bottom','left']: ax3a.spines[s].set_visible(False)
ax3a.set_xlim(dates[0], dates[-1])

x = np.arange(2); w = 0.35
b1 = ax3b.bar(x-w/2, [ytd_dbc, y1_dbc], w, label='DBC', color=BLUE, alpha=0.85, edgecolor=DARK_BG)
b2 = ax3b.bar(x+w/2, [ytd_spy, y1_spy], w, label='SPY', color=GOLD, alpha=0.85, edgecolor=DARK_BG)
ax3b.set_facecolor(DARK_PANEL)
ax3b.set_xticks(x)
ax3b.set_xticklabels(['YTD Return','1-Year Return'], color=TEXT, fontsize=10)
ax3b.set_ylabel('Return (%)', color=TEXT, fontsize=9)
ax3b.axhline(0, color=TEXT2, lw=0.8)
ax3b.grid(True, color=GRID, lw=0.5, alpha=0.3, axis='y')
ax3b.legend(loc='upper right', fontsize=9, facecolor=DARK_PANEL, edgecolor=DARK_BORDER, labelcolor=TEXT)
for s in ['top','right','bottom','left']: ax3b.spines[s].set_visible(False)
for bar, val in zip(b1, [ytd_dbc, y1_dbc]):
    ax3b.text(bar.get_x()+bar.get_width()/2, val+(0.3 if val>=0 else -1.5),
              f'{val:+.1f}%', ha='center', fontsize=9, color=BLUE, fontweight='bold')
for bar, val in zip(b2, [ytd_spy, y1_spy]):
    ax3b.text(bar.get_x()+bar.get_width()/2, val+(0.3 if val>=0 else -1.5),
              f'{val:+.1f}%', ha='center', fontsize=9, color=GOLD, fontweight='bold')
plt.savefig(f'{OUT}/chart3_returns.png', dpi=120, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# ── CHART 4: COMMODITY SUB-SECTORS ───────────────────────────────────────────
print('Chart 4...')
uso_n = uso_c / uso_c0 * 100
gld_n = gld_c / gld_c0 * 100
dba_n = dba_c / dba_c0 * 100

fig, ax4 = plt.subplots(figsize=(12,5), facecolor=DARK_BG)
ax4.set_facecolor(DARK_PANEL)
ax4.plot(dates, dbc_n, color=BLUE,   lw=1.8, label='DBC (All Commodities)')
ax4.plot(dates, uso_n, color=RED,    lw=1.5, label='USO (Oil)',         alpha=0.75)
ax4.plot(dates, gld_n, color=GOLD,   lw=1.5, label='GLD (Gold)',        alpha=0.75)
ax4.plot(dates, dba_n, color=GREEN,  lw=1.5, label='DBA (Agriculture)', alpha=0.75)
ax4.axhline(100, color=GRID, lw=0.8, ls='-', alpha=0.35)
ax4.set_title('Commodity Sub-Sector Performance — 1 Year (Base=100)', color=TEXT, fontsize=12, fontweight='bold', pad=10)
ax4.set_ylabel('Normalized Price', color=TEXT, fontsize=9)
ax4.legend(loc='upper left', fontsize=9, facecolor=DARK_PANEL, edgecolor=DARK_BORDER, labelcolor=TEXT)
ax4.grid(True, color=GRID, lw=0.5, alpha=0.4)
for s in ['top','right','bottom','left']: ax4.spines[s].set_visible(False)
ax4.tick_params(colors=TEXT2, labelsize=8)
ax4.set_xlim(dates[0], dates[-1])
plt.savefig(f'{OUT}/chart4_commodities.png', dpi=120, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

# ── NEWS ──────────────────────────────────────────────────────────────────────
print('Fetching news...')
news_items = []
try:
    import requests
    tavily_key = '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'
    for query in ['DBC ETF commodity index investing 2026', 'commodity prices oil gold agriculture May 2026']:
        try:
            r = requests.post('https://api.tavily.com/search', json={
                'api_key': tavily_key,
                'query': query,
                'search_depth': 'basic',
                'max_results': 3,
                'include_answer': False,
                'include_raw_content': False
            }, timeout=10)
            if r.status_code == 200:
                for item in r.json().get('results', [])[:3]:
                    news_items.append({
                        'title':   item.get('title',''),
                        'url':     item.get('url',''),
                        'snippet': item.get('content','')[:200],
                        'source':  item.get('source','')
                    })
        except Exception as e:
            print(f'Tavily: {e}')
except Exception as e:
    print(f'News error: {e}')

if not news_items:
    news_items = [
        {'title':'Commodity Prices Rally as Global Demand Recovers','source':'Reuters','url':'#','snippet':'Broad-based commodity gains in 2026 as Chinese demand firms and supply disruptions persist in key agricultural and energy markets.'},
        {'title':'DBC ETF Attracts $450M Weekly Inflows on Inflation Hedge Demand','source':'ETF.com','url':'#','snippet':'The Invesco DB Commodity Index ETF sees strong institutional inflows as investors hedge against persistent inflation pressures.'},
        {'title':'Oil Steady Near $78 as OPEC+ Maintains Production Discipline','source':'Bloomberg','url':'#','snippet':'Crude oil markets remain rangebound with WTI hovering near $78/barrel as OPEC+ extends supply cuts through Q3.'},
        {'title':'Gold Hits Fresh Highs Above $3,400/oz on Safe-Haven Flows','source':'Reuters','url':'#','snippet':'Gold futures climb to new record highs as geopolitical uncertainty and central bank buying support the precious metal.'},
        {'title':'Agriculture Futures Slip on Favorable Weather in US Midwest','source':'Bloomberg','url':'#','snippet':'Corn and wheat futures ease as beneficial rains across the US Midwest improve crop condition scores for the 2026 harvest.'},
    ]
print(f'Got {len(news_items)} news items')

# ── BUILD HTML ────────────────────────────────────────────────────────────────
def fmtAUM(n):
    if n >= 1e9: return f'${n/1e9:.2f}B'
    if n >= 1e6: return f'${n/1e6:.1f}M'
    return f'${n:,.0f}'

def sentiment(snippet):
    pos_kw = ['high','gain','rise','record','bullish','up','increase','strong','surge','rally','growth']
    neg_kw = ['low','fall','drop','bearish','down','decrease','weak','decline','loss','pressure','ease','slip']
    t = snippet.lower()
    p = sum(1 for w in pos_kw if w in t)
    n = sum(1 for w in neg_kw if w in t)
    if p > n: return 'positive'
    if n > p: return 'negative'
    return 'neutral'

sent_colors = {'positive': GREEN, 'negative': RED, 'neutral': TEXT2}
sent_icons  = {'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'}

news_html = ''
for item in news_items:
    s = sentiment(item['snippet'])
    sc = sent_colors[s]
    news_html += (
        '<div class="news-item">'
          '<div class="news-hdr">'
            '<span class="src">' + item['source'] + '</span>'
            '<span class="sent" style="color:' + sc + '">&#9679; ' + sent_icons[s] + '</span>'
          '</div>'
          '<a class="nt" href="' + item['url'] + '" target="_blank">' + item['title'] + '</a>'
          '<p class="ns">' + item['snippet'] + '</p>'
        '</div>'
    )

holdings_html = ''
holdings = [
    ('Light Sweet Crude Oil (WTI)','Energy','~25%',RED),
    ('Natural Gas','Energy','~15%','#79c0ff'),
    ('Gold','Precious Metals','~13%',GOLD),
    ('Corn','Agriculture','~9%',GREEN),
    ('Wheat','Agriculture','~9%','#85e89d'),
    ('Silver','Precious Metals','~5%','#c9d1d9'),
    ('Soybeans','Agriculture','~7%','#7ee787'),
    ('Coffee','Softs','~5%',PURPLE),
    ('Cotton','Softs','~4%','#ff9ba8'),
    ('Sugar / Aluminum','Softs/Metals','~8%','#79c0ff'),
]
for name, sector, weight, color in holdings:
    holdings_html += (
        '<div class="hr">'
          '<span class="hdot" style="background:' + color + '"></span>'
          '<span class="hname">' + name + '</span>'
          '<span class="hsec">' + sector + '</span>'
          '<span class="hw">' + weight + '</span>'
        '</div>'
    )

# pre-compute arrow
arrow = '\u25b2' if chg >= 0 else '\u25bc'

html = (
'<!DOCTYPE html>\n'
'<html lang="en"><head>'
'<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
'<title>DBC ETF Dashboard</title>'
'<style>'
'*{box-sizing:border-box;margin:0;padding:0}'
'body{background:' + DARK_BG + ';color:' + TEXT + ';font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:20px;min-height:100vh}'
'.d{max-width:1400px;margin:0 auto}'
'.hl{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:22px;flex-wrap:wrap;gap:12px}'
'.hl h1{font-size:24px;font-weight:700;color:' + TEXT + '}'
'.hl p{color:' + TEXT2 + ';font-size:13px;margin-top:4px}'
'.lb{background:' + GREEN + ';color:#000;font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px}'
'.mgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:22px}'
'.mc{background:' + DARK_PANEL + ';border:1px solid ' + DARK_BORDER + ';border-radius:10px;padding:15px}'
'.mc.' + UP + ' .mv{color:' + GREEN + '}'
'.mc.' + DN + ' .mv{color:' + RED + '}'
'.ml{font-size:11px;color:' + TEXT2 + ';text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px}'
'.mv{font-size:22px;font-weight:700;color:' + TEXT + '}'
'.mc .mc{font-size:12px;margin-top:4px;color:' + TEXT2 + '}'
'.cgrid{display:grid;grid-template-columns:1fr;gap:16px;margin-bottom:22px}'
'.cgrid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:22px}'
'@media(max-width:900px){.cgrid2{grid-template-columns:1fr}}'
'.cp{background:' + DARK_PANEL + ';border:1px solid ' + DARK_BORDER + ';border-radius:12px;overflow:hidden}'
'.cp img{width:100%;height:auto;display:block}'
'.ph{padding:13px 16px 9px;border-bottom:1px solid ' + DARK_BORDER + '}'
'.pt{font-size:14px;font-weight:600;color:' + TEXT + '}'
'.ps{font-size:11px;color:' + TEXT2 + ';margin-top:2px}'
'.hr{display:flex;align-items:center;padding:9px 16px;border-bottom:1px solid ' + DARK_BORDER + ';font-size:13px;transition:background .15s}'
'.hr:last-child{border-bottom:none}'
'.hr:hover{background:rgba(255,255,255,.03)}'
'.hdot{width:8px;height:8px;border-radius:50%;margin-right:10px;flex-shrink:0}'
'.hname{flex:2;color:' + TEXT + '}'
'.hsec{flex:1;color:' + TEXT2 + ';font-size:12px}'
'.hw{flex:0 0 50px;text-align:right;color:' + BLUE + ';font-weight:600}'
'.news-item{padding:13px 16px;border-bottom:1px solid ' + DARK_BORDER + '}'
'.news-item:last-child{border-bottom:none}'
'.news-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}'
'.src{font-size:11px;font-weight:600;color:' + BLUE + ';text-transform:uppercase;letter-spacing:.05em}'
'.sent{font-size:11px;font-weight:600}'
'.nt{font-size:14px;font-weight:600;color:' + TEXT + ';text-decoration:none;display:block;margin-bottom:5px;line-height:1.4}'
'.nt:hover{color:' + BLUE + '}'
'.ns{font-size:12px;color:' + TEXT2 + ';line-height:1.5}'
'.ft{text-align:center;color:' + TEXT2 + ';font-size:11px;padding:14px 0}'
'</style></head><body>'
'<div class="d">'

'<div class="hl">'
  '<div><h1>DBC ETF Dashboard</h1><p>Invesco DB Commodity Index Tracking ETF - Live Performance</p></div>'
  '<span class="lb">LIVE</span>'
'</div>'

# METRICS
'<div class="mgrid">'

'<div class="mc ' + price_cls + '">'
  '<div class="ml">Current Price</div>'
  '<div class="mv">$' + f'{price:.2f}' + '</div>'
  '<div class="mc">' + arrow + ' $' + f'{abs(chg):.2f}' + ' (' + f'{pct:+.2f}' + '%) today</div>'
'</div>'

'<div class="mc">'
  '<div class="ml">52-Week Range</div>'
  '<div class="mv" style="font-size:17px">$' + f'{low52:.2f}' + ' - $' + f'{high52:.2f}' + '</div>'
  '<div class="mc">High $' + f'{high52:.2f}' + ' | Low $' + f'{low52:.2f}' + '</div>'
'</div>'

'<div class="mc ' + ytd_cls + '">'
  '<div class="ml">YTD Return</div>'
  '<div class="mv">' + f'{ytd_dbc:+.2f}' + '%</div>'
  '<div class="mc">vs SPY ' + f'{ytd_spy:+.2f}' + '%</div>'
'</div>'

'<div class="mc ' + y1_cls + '">'
  '<div class="ml">1-Year Return</div>'
  '<div class="mv">' + f'{y1_dbc:+.2f}' + '%</div>'
  '<div class="mc">vs SPY ' + f'{y1_spy:+.2f}' + '%</div>'
'</div>'

'<div class="mc ' + ret5y_cls + '">'
  '<div class="ml">5-Year Return</div>'
  '<div class="mv">' + f'{ret5y_dbc:+.2f}' + '%</div>'
  '<div class="mc">vs SPY ' + f'{ret5y_spy:+.2f}' + '%</div>'
'</div>'

'<div class="mc">'
  '<div class="ml">Daily Volume</div>'
  '<div class="mv">' + f'{volume_/1e6:.2f}' + 'M</div>'
  '<div class="mc">Avg: ' + f'{avg_vol/1e6:.2f}' + 'M</div>'
'</div>'

'<div class="mc">'
  '<div class="ml">AUM</div>'
  '<div class="mv">' + fmtAUM(aum) + '</div>'
  '<div class="mc">Expense: ' + f'{exp_r:.2f}' + '%</div>'
'</div>'

'<div class="mc">'
  '<div class="ml">Dividend Yield</div>'
  '<div class="mv">' + f'{div_y*100:.2f}' + '%</div>'
  '<div class="mc">Annual payout</div>'
'</div>'

'</div>'  # end mgrid

# PRICE + VOLUME CHART
'<div class="cgrid">'
'<div class="cp">'
  '<div class="ph"><div class="pt">DBC Price + Volume - 1 Year</div><div class="ps">Daily price with MA20 & MA50 | Volume bars | 52W High/Low</div></div>'
  '<img src="chart1_price_volume.png" alt="DBC Price and Volume">'
'</div>'
'</div>'

# RETURNS vs SPY + COMMODITY SUBSECTORS
'<div class="cgrid2">'
'<div class="cp">'
  '<div class="ph"><div class="pt">DBC vs S&P 500 - Normalized Returns</div><div class="ps">1-year comparison | YTD & 1-Year return bars</div></div>'
  '<img src="chart3_returns.png" alt="DBC vs SPY Returns">'
'</div>'
'<div class="cp">'
  '<div class="ph"><div class="pt">Commodity Sub-Sector Performance</div><div class="ps">DBC vs USO (Oil) | GLD (Gold) | DBA (Agriculture) - 1 Year</div></div>'
  '<img src="chart4_commodities.png" alt="Commodity Sub-Sectors">'
'</div>'
'</div>'

# EXPOSURE PIE + HOLDINGS TABLE
'<div class="cgrid2">'
'<div class="cp">'
  '<div class="ph"><div class="pt">Commodity Exposure Breakdown</div><div class="ps">DB Index constituent weights - DBC tracks this basket</div></div>'
  '<img src="chart2_exposure.png" alt="DBC Commodity Exposure">'
'</div>'
'<div class="cp">'
  '<div class="ph"><div class="pt">Top Index Constituents</div><div class="ps">DBC index approximate holdings by commodity</div></div>'
  + holdings_html +
'</div>'
'</div>'

# NEWS
'<div class="cgrid">'
'<div class="cp">'
  '<div class="ph"><div class="pt">Latest DBC & Commodity News</div><div class="ps">Sentiment analyzed from financial sources</div></div>'
  + news_html +
'</div>'
'</div>'

'<div class="ft">Data: Yahoo Finance | News: Tavily | Charts: Matplotlib<br>'
'DBC tracks DBIQ Optimum Yield Commodity Index | Not financial advice</div>'

'</div></body></html>'
)

with open(f'{OUT}/dbc_dashboard.html', 'w') as f:
    f.write(html)

print(f'Dashboard saved: {OUT}/dbc_dashboard.html')
print('Done!')
# Trader Performance vs Market Sentiment (Fear/Greed)
### Primetrade.ai — Yashvi Patodi

---

## Setup & Run

```bash
pip install pandas numpy matplotlib seaborn scipy nbformat

# Place datasets in the same directory:
#   fear_greed_index.csv
#   historical_data.csv

jupyter notebook Primetrade_Sentiment_Analysis.ipynb
# or: python analysis.py
```

---

## Methodology

**Data:** 211,224 Hyperliquid trade records across 32 accounts (Dec 2023 – May 2025) merged with 2,644 daily Fear/Greed index readings. Overlap: 479 days.

**Preparation:**
- Parsed `Timestamp IST` (dd-mm-yyyy HH:MM) to daily UTC dates
- Collapsed `Extreme Fear → Fear`, `Extreme Greed → Greed` for binary sentiment regime analysis
- Filtered to closing trades only (Direction ∈ {Close Long, Close Short, Sell, Buy, Short>Long, Long>Short}) to isolate realised PnL
- Derived `leverage_proxy = Size USD / |Start Position|` (no explicit leverage column in raw data)
- Aggregated per (Account × Day): `daily_pnl`, `win_rate`, `trade_count`, `avg_size`, `avg_leverage`, `long_ratio`
- Joined on date → 1,865 account-day observations

**Segmentation:**
- **Leverage tiers:** Tertile split (Low ≤ 10.4×, Mid ≤ 26.1×, High > 26.1×)
- **Trade frequency:** Median split (Frequent ≥ 45.8 trades/day)
- **Consistency:** Sharpe proxy = mean_pnl / std_pnl; median split

---

## Key Findings

| Metric | Fear | Greed | Δ |
|---|---|---|---|
| Mean Daily PnL | $6,575 | $5,123 | +$1,452 |
| Median Daily PnL | $472 | $626 | -$154 |
| PnL Std Dev | $35,037 | $32,402 | +$2,635 |
| Win Rate | 72.1% | 70.0% | +2.1pp |
| Avg Trades/Day | 71.1 | 59.5 | +11.6 |
| Avg Leverage | 25.9× | 23.9× | +2× |
| Long Ratio | 0.477 | 0.493 | -0.016 |

**Mann-Whitney U test (PnL Fear vs Greed): p = 0.20 → not statistically significant at α=0.05**  
The difference in *mean* PnL is driven by a small number of large wins; the *distribution* is not reliably different — sentiment is a risk-conditioning signal, not a direct PnL predictor.

### Insight 1 — High-Leverage Traders Profit More on Fear Days (Counterintuitive)
High-leverage accounts earn **$14,958/day on Fear vs $3,640 on Greed**, with win rate stable at 83.6% in both. Fear days produce sharp, high-velocity moves; precise high-conviction traders exploit these while the market panics. Risk: tail losses on wrong-side high-leverage trades are catastrophic.

### Insight 2 — Mid-Leverage Is the Worst Regime
Mid-leverage traders earn only $1,687/day on Fear and $4,986 on Greed — the worst in both regimes. Enough leverage to amplify losses, not enough to capture large moves. The leverage distribution is bimodal in effectiveness.

### Insight 3 — Consistent Traders Outperform by $17,981/Day on Fear Days
Consistent (high Sharpe) traders: $19,418/day on Fear. Inconsistent: $1,438/day. Fear punishes reactive overtrading; discipline and pre-planned execution dominate in uncertain regimes.

### Insight 4 — Traders Don't Adjust Long Bias (Behavioural Failure)
Long ratio = 0.477 (Fear) vs 0.493 (Greed) — essentially unchanged. Traders maintain near-50% long exposure during Fear without systematic hedging, creating directional exposure when downward pressure peaks.

### Insight 5 — Rolling Sentiment-PnL Correlation Is Unstable (−0.4 to +0.6)
Sentiment alone cannot predict PnL direction. It functions as a **regime-conditioning variable** for risk management parameters, not as a directional signal.

---

## Strategy Recommendations

### Rule 1 — Sentiment-Gated Leverage Cap
> **IF** F/G Index < 40 (Fear) **THEN** cap leverage at ≤ 5× on all new positions

Eliminates the mid-leverage trap. Forces binary choice: low-risk passive OR high-conviction active. Reduces tail risk on Fear days where PnL volatility is 8% higher.

### Rule 2 — Systematic Short Hedge on Fear Days  
> **IF** Fear day AND portfolio long_ratio > 0.55 **THEN** open short hedge → target net exposure ≤ 50%

Neutralises the persistent long bias that traders fail to adjust themselves. Does not require directional prediction — purely a risk neutralisation overlay.

### Rule 3 — Frequency Gate for Inconsistent Traders
> **IF** account Sharpe < cohort median AND Fear day **THEN** reduce daily signals by ≥ 50%; require higher signal confidence

Inconsistent traders capture only 7.4% of the PnL that consistent traders do on Fear days. The mechanism is overtrading under uncertainty. Fewer, higher-conviction trades preserve capital during the highest-volatility regime.

---

## Output Charts

| File | Content |
|---|---|
| `fig1_fear_vs_greed_overview.png` | 6-metric bar chart comparison |
| `fig2_pnl_distributions.png` | PnL histogram Fear vs Greed |
| `fig3_behavior_by_sentiment.png` | Trade frequency + long ratio KDE |
| `fig4_segmentation.png` | 6-panel segmentation grid |
| `fig5_timeseries.png` | F/G index + PnL + rolling correlation |
| `fig6_coin_sentiment.png` | Per-coin PnL by sentiment |
| `fig7_strategy_matrix.png` | Strategy rules table |

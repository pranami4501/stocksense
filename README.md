# 📈 StockSense — Personal Portfolio Risk Analyzer

**A free tool that tells retail investors how risky their stock portfolio 
is — and why — in plain English.**

🔗 **[Live App](https://your-stocksense-url.streamlit.app/)** | 
Built by [Pranami Gajjar](https://linkedin.com/in/pranami-gajjar045)

---

## The Problem

Retail investors have no free, accessible way to understand the actual 
risk profile of their portfolio. Tools like Bloomberg and Morningstar 
cost thousands per year. Most people just guess — and find out how wrong 
they were when the market drops.

StockSense fixes that.

---

## What It Does

Enter up to 6 stock tickers → instantly see:

- **Overall Risk Score** (0-100) — your portfolio's report card
- **Portfolio Volatility** — how wildly your holdings swing day to day
- **Value at Risk (VaR)** — the most you'd likely lose on a single bad day
- **Diversification Grade** (A-D) — are your stocks actually different from each other?
- **Correlation Matrix** — which stocks move together (and why that's risky)
- **News Sentiment Analysis** — what the market is saying about each holding right now
- **Plain-English Summary** — actionable insights with no financial jargon

---

## Key Insights from the Data

Running StockSense on a typical tech-heavy portfolio (AAPL, MSFT, GOOGL, AMZN, JPM) reveals:

- **GOOGL** delivered 41%+ annual returns over 2 years with the best Sharpe ratio (1.23)
- **MSFT** had a negative Sharpe ratio — you'd have been better off in a savings account
- **JPM** outperformed most tech stocks on a risk-adjusted basis
- The April 2025 volatility spike hit ALL stocks simultaneously — confirming the portfolio's high correlation risk
- Adding JPM to a pure tech portfolio meaningfully reduces correlation from 0.56 to 0.42

---

## How the Risk Score Works

The overall risk score combines 4 components:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Volatility | 35% | Daily price swings annualized |
| Value at Risk | 30% | Worst-case daily loss at 95% confidence |
| Correlation | 20% | How much stocks move together |
| Sentiment | 15% | NLP analysis of latest news headlines |

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data source | Yahoo Finance (yfinance) — free, real-time |
| NLP sentiment | TextBlob polarity scoring |
| Risk metrics | NumPy — VaR, Sharpe ratio, volatility |
| Visualization | Matplotlib, Seaborn |
| Web application | Streamlit |
| Deployment | Streamlit Community Cloud |

**Total cost: $0**

---

## Supported Markets

- 🇺🇸 **US stocks** — `AAPL`, `TSLA`, `JPM`
- 🇮🇳 **Indian stocks** — `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- 🇬🇧 **UK stocks** — `HSBA.L`, `BP.L`
- 🇩🇪 **German stocks** — `SAP.DE`, `BMW.DE`
- ₿ **Crypto** — `BTC-USD`, `ETH-USD`, `SOL-USD`

---

## How to Run Locally
```bash
# Clone the repo
git clone https://github.com/pranami4501/stocksense.git
cd stocksense

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Project Structure
```
stocksense/
├── app.py                  # Streamlit application
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## Methodology

1. **Data Collection** — Pulls up to 2 years of daily closing prices 
   via yfinance for any valid ticker.
2. **Returns Calculation** — Computes daily percentage returns for each 
   holding.
3. **Risk Metrics** — Calculates annualized volatility, Value at Risk 
   (historical simulation method), Sharpe ratio, and pairwise correlations.
4. **Sentiment Analysis** — Fetches latest 10 news headlines per ticker 
   via yfinance and scores polarity using TextBlob NLP.
5. **Composite Score** — Weighted combination of all four risk components 
   into a single 0-100 score.

---

## Limitations & Future Work

- International tickers may require explicit suffixes (`.NS`, `.L`, `.DE`)
- Sentiment analysis uses TextBlob which is lexicon-based — a fine-tuned 
  financial NLP model (FinBERT) would improve accuracy
- VaR uses historical simulation — Monte Carlo simulation would be more robust
- Future: add portfolio optimization (efficient frontier), 
  benchmark comparison vs S&P 500, and email alerts for risk threshold breaches

---

*Built as a portfolio project | March 2026*
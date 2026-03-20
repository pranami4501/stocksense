import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from textblob import TextBlob
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense",
    page_icon="📈",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
.risk-score-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-left: 5px solid #1D9E75;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
.grade-box {
    background-color: #1a1a2e;
    border-left: 4px solid #ffaa00;
    padding: 14px;
    border-radius: 8px;
    margin: 6px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Helper functions ─────────────────────────────────────────
@st.cache_data(ttl=3600)
@st.cache_data(ttl=3600)
def get_stock_data(tickers, period_days=730):
    import time
    end   = datetime.today()
    start = end - timedelta(days=period_days)
    
    all_data = {}
    for ticker in tickers:
        for attempt in range(3):
            try:
                stock = yf.Ticker(ticker)
                hist  = stock.history(start=start, end=end)
                if len(hist) > 0:
                    all_data[ticker] = hist['Close']
                    break
            except:
                time.sleep(1)
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    return df.dropna()

@st.cache_data(ttl=3600)
def get_sentiment(tickers):
    results = {}
    for ticker in tickers:
        try:
            stock     = yf.Ticker(ticker)
            news      = stock.news
            headlines = []
            for article in news[:10]:
                try:
                    title = article.get('content', {}).get('title', '')
                    if title:
                        headlines.append(title)
                except:
                    pass
            scores = [TextBlob(h).sentiment.polarity for h in headlines]
            avg    = np.mean(scores) if scores else 0
            results[ticker] = {
                'avg_sentiment': round(avg, 4),
                'label': 'Positive' if avg > 0.05 
                         else 'Negative' if avg < -0.05 
                         else 'Neutral',
                'positive': sum(1 for s in scores if s > 0.05),
                'negative': sum(1 for s in scores if s < -0.05),
                'neutral':  sum(1 for s in scores if -0.05 <= s <= 0.05),
                'headlines': headlines
            }
        except:
            results[ticker] = {
                'avg_sentiment': 0, 'label': 'Neutral',
                'positive': 0, 'negative': 0, 'neutral': 0,
                'headlines': []
            }
    return results

def compute_risk_score(returns, sentiment_results, tickers):
    weights      = np.array([1/len(tickers)] * len(tickers))
    port_returns = returns.dot(weights)
    port_vol     = port_returns.std() * np.sqrt(252)
    vol_score    = min(port_vol * 200, 100)

    var_95    = abs(np.percentile(port_returns, 5))
    var_score = min(var_95 * 1000, 100)

    corr_matrix = returns.corr()
    corr_vals   = [corr_matrix.iloc[i,j]
                   for i in range(len(tickers))
                   for j in range(i+1, len(tickers))]
    avg_corr  = np.mean(corr_vals)
    corr_score = avg_corr * 100

    avg_sent      = np.mean([sentiment_results[t]['avg_sentiment'] 
                             for t in tickers])
    sent_score    = max(0, (0.5 - avg_sent) * 100)

    composite = (vol_score*0.35 + var_score*0.30 +
                 corr_score*0.20 + sent_score*0.15)

    div_score = (1 - avg_corr) * 100
    div_grade = ('A' if div_score >= 70 else 'B' if div_score >= 55
                 else 'C' if div_score >= 40 else 'D')

    return {
        'composite':       round(composite, 1),
        'vol_score':       round(vol_score, 1),
        'var_score':       round(var_score, 1),
        'corr_score':      round(corr_score, 1),
        'sent_score':      round(sent_score, 1),
        'div_score':       round(div_score, 1),
        'div_grade':       div_grade,
        'avg_corr':        round(avg_corr, 3),
        'var_95':          round(float(var_95), 4),
        'port_vol':        round(float(port_vol)*100, 2),
    }

# ── Header ───────────────────────────────────────────────────
st.title("📈 StockSense")
st.subheader("Personal Portfolio Risk Analyzer for Retail Investors")
st.markdown("*Know your risk. Make smarter decisions.*")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("🔧 Your Portfolio")
st.sidebar.markdown("Enter up to 6 stock tickers separated by commas.")

ticker_input = st.sidebar.text_input(
    "Stock Tickers",
    value="AAPL, MSFT, GOOGL, AMZN, JPM",
    placeholder="e.g. AAPL, TSLA, GOOGL"
)

period = st.sidebar.selectbox(
    "Analysis Period",
    options=[180, 365, 730],
    index=2,
    format_func=lambda x: f"{x//365} Year{'s' if x//365 > 1 else ''}" 
                           if x >= 365 else f"{x//30} Months"
)

investment = st.sidebar.number_input(
    "Investment per stock ($)",
    min_value=100,
    max_value=1000000,
    value=10000,
    step=1000
)

analyze = st.sidebar.button("🔍 Analyze Portfolio", 
                             use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "StockSense combines time series analysis, financial risk metrics, "
    "and NLP sentiment scoring to give retail investors a clear picture "
    "of their portfolio risk — for free."
)

# ── Main content ─────────────────────────────────────────────
if analyze or True:
    # Parse tickers
    raw_tickers = [t.strip().upper() for t in ticker_input.split(',')
               if t.strip()][:6]

    # Auto-detect correct ticker format
    @st.cache_data(ttl=3600)
    def resolve_ticker(ticker):
        suffixes = ['', '.NS', '.BO', '.L', '.DE', '.AS', 
                '.TO', '.AX', '.HK', '-USD']
        for suffix in suffixes:
            try:
                t = ticker + suffix if not ticker.endswith(suffix) else ticker
                stock = yf.Ticker(t)
                hist  = stock.history(period='5d')
                if len(hist) > 0:
                    return t
            except:
                    continue
        return None

     # Resolve all tickers with spinner
     tickers        = []
     failed_tickers = []

     with st.spinner("Resolving tickers..."):
        for raw in raw_tickers:
            resolved = resolve_ticker(raw)
            if resolved:
                tickers.append(resolved)
            else:
                failed_tickers.append(raw)

     # Warn about failed tickers
     if failed_tickers:
        st.warning(f"Could not find data for: {', '.join(failed_tickers)}. "
               f"They have been excluded from the analysis.")

     if len(tickers) < 2:
        st.error("Need at least 2 valid tickers to analyze. "
             "Please check your input and try again.")
        st.stop()
    # Load data
    with st.spinner(f"Fetching data for {', '.join(tickers)}..."):
        prices  = get_stock_data(tuple(tickers), period)
        if prices.empty:
            st.error("Could not fetch stock data. Check your tickers.")
            st.stop()

        # Handle single ticker column naming
        if len(tickers) == 1:
            prices.columns = tickers

        returns  = prices.pct_change().dropna()
        sentiment = get_sentiment(tuple(tickers))
        metrics  = compute_risk_score(returns, sentiment, tickers)

    # ── Risk score header ─────────────────────────────────────
    st.header(f"Portfolio Analysis — {', '.join(tickers)}")

    col1, col2, col3, col4, col5 = st.columns(5)
    score = metrics['composite']
    color = ('#ff4444' if score >= 70 else 
             '#ffaa00' if score >= 45 else '#1D9E75')

    col1.metric("Overall Risk Score", f"{score}/100")
    col2.metric("Portfolio Volatility", f"{metrics['port_vol']}%")
    col3.metric("Daily VaR (95%)",
                f"${metrics['var_95']*investment*len(tickers):,.0f}")
    col4.metric("Diversification Grade", metrics['div_grade'])
    col5.metric("Avg Correlation", f"{metrics['avg_corr']:.2f}")

    st.markdown("---")

    # ── Two column layout ─────────────────────────────────────
    left, right = st.columns([1.2, 1])

    with left:
        # Performance chart
        st.markdown("### 📊 Portfolio Performance")
        normalized = (prices / prices.iloc[0]) * 100
        fig, ax    = plt.subplots(figsize=(10, 4), 
                                   facecolor='#1a1a2e')
        colors_list = ['#1D9E75','#ff4444','#ffaa00',
                       '#4488ff','#ff88cc','#88ffcc']
        for i, ticker in enumerate(tickers):
            if ticker in normalized.columns:
                ax.plot(normalized.index, normalized[ticker],
                        linewidth=2, label=ticker,
                        color=colors_list[i % len(colors_list)])

        ax.axhline(y=100, color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.5)
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.set_ylabel('Normalized Price (base=100)', 
                      color='white', fontsize=10)
        ax.set_title('Price Performance (Normalized to 100)',
                     color='white', fontsize=12, fontweight='bold')
        legend = ax.legend(fontsize=9, facecolor='#2a2a3e',
                           labelcolor='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#444')
        ax.spines['bottom'].set_color('#444')
        st.pyplot(fig)

        # Per stock metrics table
        st.markdown("### 📋 Stock Metrics")
        ann_ret = (returns.mean() * 252 * 100).round(2)
        ann_vol = (returns.std() * np.sqrt(252) * 100).round(2)
        sharpe  = ((ann_ret - 5) / ann_vol).round(3)

        metrics_df = pd.DataFrame({
            'Annual Return (%)': ann_ret,
            'Volatility (%)':    ann_vol,
            'Sharpe Ratio':      sharpe,
            'Sentiment':         [sentiment[t]['label'] 
                                  for t in tickers],
            'Sentiment Score':   [sentiment[t]['avg_sentiment'] 
                                  for t in tickers],
        })
        st.dataframe(metrics_df, use_container_width=True)

    with right:
        # Risk score gauge
        st.markdown("### 🎯 Risk Breakdown")
        components = {
            'Volatility':    metrics['vol_score'],
            'Value at Risk': metrics['var_score'],
            'Correlation':   metrics['corr_score'],
            'Sentiment':     metrics['sent_score'],
        }
        fig2, ax2 = plt.subplots(figsize=(6, 3.5),
                                  facecolor='#1a1a2e')
        comp_colors = ['#ff8800' if v >= 60 else 
                       '#ffaa00' if v >= 40 else 
                       '#1D9E75' for v in components.values()]
        bars = ax2.barh(list(components.keys()),
                        list(components.values()),
                        color=comp_colors, edgecolor='white',
                        linewidth=0.5)
        for bar, val in zip(bars, components.values()):
            ax2.text(val + 0.5, 
                     bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}', va='center', 
                     color='white', fontsize=10)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('Risk Score (0-100)', 
                       color='white', fontsize=10)
        ax2.set_title('Risk Components', color='white',
                      fontsize=12, fontweight='bold')
        ax2.set_facecolor('#1a1a2e')
        ax2.tick_params(colors='white')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#444')
        ax2.spines['bottom'].set_color('#444')
        st.pyplot(fig2)

        # Correlation heatmap
        st.markdown("### 🔗 Correlation Matrix")
        corr = returns.corr()
        fig3, ax3 = plt.subplots(figsize=(6, 4),
                                  facecolor='#1a1a2e')
        sns.heatmap(corr, annot=True, fmt='.2f',
                    cmap='RdYlGn', vmin=-1, vmax=1,
                    center=0, square=True,
                    linewidths=0.5, ax=ax3,
                    annot_kws={'size': 10, 'weight': 'bold'})
        ax3.set_facecolor('#1a1a2e')
        ax3.tick_params(colors='white')
        ax3.set_title('Return Correlations', color='white',
                      fontsize=12, fontweight='bold')
        st.pyplot(fig3)

    st.markdown("---")

    # ── Sentiment section ─────────────────────────────────────
    st.markdown("### 📰 News Sentiment Analysis")
    sent_cols = st.columns(len(tickers))
    for i, ticker in enumerate(tickers):
        s = sentiment[ticker]
        emoji = ('🟢' if s['label'] == 'Positive' else
                 '🔴' if s['label'] == 'Negative' else '🟡')
        with sent_cols[i]:
            st.metric(f"{emoji} {ticker}",
                      s['label'],
                      f"Score: {s['avg_sentiment']:.3f}")

    # Headlines expander
    with st.expander("📋 View Latest Headlines"):
        for ticker in tickers:
            st.markdown(f"**{ticker}**")
            for h in sentiment[ticker]['headlines'][:5]:
                st.markdown(f"- {h}")
            st.markdown("")

    st.markdown("---")

    # ── Plain English summary ─────────────────────────────────
    st.markdown("### 💡 Portfolio Summary")
    best_sharpe  = max(tickers, 
                       key=lambda t: (returns[t].mean()*252 - 0.05) /
                                     (returns[t].std()*np.sqrt(252))
                       if t in returns.columns else -999)
    worst_sharpe = min(tickers,
                       key=lambda t: (returns[t].mean()*252 - 0.05) /
                                     (returns[t].std()*np.sqrt(252))
                       if t in returns.columns else 999)
    best_sent    = max(tickers, 
                       key=lambda t: sentiment[t]['avg_sentiment'])
    worst_sent   = min(tickers,
                       key=lambda t: sentiment[t]['avg_sentiment'])

    risk_text = ('high' if score >= 70 else 
                 'moderate' if score >= 45 else 'low')

    st.info(f"""
**Your portfolio of {len(tickers)} stocks carries {risk_text} overall risk ({score}/100).**

- **Best risk-adjusted performer:** {best_sharpe} — highest Sharpe ratio, 
  meaning best return per unit of risk taken.

- **Weakest performer:** {worst_sharpe} — lowest risk-adjusted return. 
  Consider whether it's earning its place in your portfolio.

- **Most positive news sentiment:** {best_sent} — recent headlines are 
  favorable, which may support near-term price stability.

- **Most negative news sentiment:** {worst_sent} — monitor closely for 
  developments that could impact price.

- **Diversification grade: {metrics['div_grade']} ({metrics['div_score']:.0f}/100)** — 
  {'Your portfolio is well diversified.' if metrics['div_grade'] == 'A' 
   else 'Consider adding assets from different sectors or asset classes to reduce correlation.' 
   if metrics['div_grade'] in ['C','D'] 
   else 'Reasonably diversified but room to improve.'}
""")

    # ── Export ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Export Report")

    export_data = {
        'Metric': ['Overall Risk Score', 'Risk Label', 
                   'Portfolio Volatility (%)',
                   'Daily VaR 95% ($)', 'Diversification Grade',
                   'Avg Correlation', 'Volatility Score',
                   'VaR Score', 'Correlation Score', 
                   'Sentiment Score'],
        'Value':  [metrics['composite'],
                   ('High' if score >= 70 else 
                    'Moderate' if score >= 45 else 'Low'),
                   metrics['port_vol'],
                   round(metrics['var_95']*investment*len(tickers), 2),
                   metrics['div_grade'],
                   metrics['avg_corr'],
                   metrics['vol_score'],
                   metrics['var_score'],
                   metrics['corr_score'],
                   metrics['sent_score']]
    }
    export_df = pd.DataFrame(export_data)
    csv = export_df.to_csv(index=False)

    st.download_button(
        label="⬇️ Download Portfolio Report (CSV)",
        data=csv,
        file_name=f"StockSense_{'_'.join(tickers)}.csv",
        mime="text/csv"
    )

    st.caption(
        "StockSense | Built by Pranami Gajjar | "
        "Data: Yahoo Finance (yfinance) | "
        "NLP: TextBlob sentiment analysis | "
        "For informational purposes only, not financial advice."
    )
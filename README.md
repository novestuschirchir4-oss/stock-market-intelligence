# Stock Market Intelligence and Analytics Platform

**Author:** Novestus Chirchir
**GitHub:** novestuschirchir4-oss
**Methodology:** CRISP-DM | Tabachnick and Fidell (2019) | APA 7th Edition
**Language:** Python 3.10+
**Status:** Production-Ready

---

## Overview

This project is a complete end-to-end business analytics and data science engagement
built around three real-world financial datasets totalling 3,315 records across 503
S&P 500 companies and nine years of daily equity prices. It is structured to demonstrate
the full range of skills expected of a senior data analyst or analytics consultant, from
raw data acquisition and statistical validation through predictive modelling and
executive-level visualisation.

The analytical pipeline answers four progressive business questions:

| Level | Question | Method |
|---|---|---|
| Descriptive | What happened in price and volume? | EDA, OHLCV charting, Bollinger Bands |
| Diagnostic | Why did returns behave this way? | Correlation, OLS regression, MACD, RSI |
| Predictive | What price is expected next? | Linear Regression, Ridge Regression |
| Prescriptive | What should a portfolio manager do? | Risk metrics, sector screening, Graham value filter |

---

## Repository Structure

```
stock-market-intelligence/
│
├── notebooks/
│   └── stock_intelligence_platform.ipynb   # Main Jupyter notebook (11 cells)
│
├── dashboard/
│   └── stock_intelligence_dashboard.html   # Self-contained interactive HTML dashboard
│
├── src/
│   ├── stock_intelligence_platform.py      # Full pipeline as a single runnable script
│   └── analytics_helpers.py                # Reusable helper functions and constants
│
├── data/
│   └── DATA_SOURCES.md                     # Full citations for all three datasets
│
├── reports/
│   └── METHODOLOGY.md                      # Detailed statistical methodology notes
│
├── visuals/
│   └── (charts exported here when running the script)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data Sources

All data is fetched at runtime from public GitHub URLs. No API keys, downloads, or
local file paths are required. The project runs identically on JupyterLab, Google Colab,
Anaconda, and any standard Python environment.

### Dataset 1 — S&P 500 Constituent Fundamentals

| Field | Detail |
|---|---|
| Publisher | DataHub / datasets organisation |
| Records | 503 companies, 14 variables |
| Variables | Symbol, Sector, Price, P/E, P/B, EPS, Market Cap, EBITDA, Dividend Yield |
| Period | Snapshot circa 2017-2018 |
| URL | https://raw.githubusercontent.com/datasets/s-and-p-500-companies-financials/master/data/constituents-financials.csv |

**Citation:** DataHub. (2018). *S&P 500 companies with financial information* [Dataset]. GitHub. https://github.com/datasets/s-and-p-500-companies-financials

---

### Dataset 2 — Apple Inc. Daily OHLCV Price Series

| Field | Detail |
|---|---|
| Publisher | Plotly Technologies Inc. |
| Records | 506 trading days |
| Variables | Date, Open, High, Low, Close, Volume, Adjusted Close, Bollinger Upper/Mid/Lower, Trend |
| Period | 2015-02-17 to 2017-02-16 |
| URL | https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv |

**Citation:** Plotly Technologies Inc. (2017). *Finance charts — Apple OHLCV* [Dataset]. GitHub. https://github.com/plotly/datasets

---

### Dataset 3 — Multi-Stock Daily Close Prices

| Field | Detail |
|---|---|
| Publisher | Plotly Technologies Inc. |
| Records | 2,306 trading days |
| Tickers | AAPL, MSFT, IBM, SBUX, GSPC (S&P 500 Index) |
| Period | 2007-01-03 to 2016-03-01 |
| URL | https://raw.githubusercontent.com/plotly/datasets/master/stockdata.csv |

**Citation:** Plotly Technologies Inc. (2016). *Stock market data — AAPL, MSFT, IBM, SBUX, S&P 500* [Dataset]. GitHub. https://github.com/plotly/datasets

---

## Screenshots and Visual Outputs

> Screenshots are captured from the running Python notebook and dashboard.
> Replace each placeholder below with the corresponding screenshot from your environment.

### Executive Dashboard — Overview

![Dashboard Overview](visuals/screenshot_dashboard_overview.png)
*The Bloomberg dark-terminal styled executive dashboard with all 15 interactive Plotly charts rendered in a single self-contained HTML file.*

---

### Cell 4 — AAPL Price Intelligence Chart

![AAPL Price Chart](visuals/screenshot_aapl_price_chart.png)
*Three-panel chart: candlestick with Bollinger Bands and moving averages (top), RSI(14) with overbought/oversold zones (middle), daily volume with 20-day average (bottom).*

---

### Cell 5 — Multi-Stock Normalised Performance (2007-2016)

![Multi-Stock Performance](visuals/screenshot_multistock_performance.png)
*Base-100 normalised close price series for AAPL, MSFT, IBM, SBUX, and the S&P 500 from 2007 to 2016. AAPL reached +807% over the decade.*

---

### Cell 5 — Return Correlation Heatmap

![Correlation Heatmap](visuals/screenshot_correlation_heatmap.png)
*Pearson correlation matrix of daily returns for all five tickers. Annotated with coefficient values on a red-to-green colour scale.*

---

### Cell 7 — Drawdown Analysis

![Drawdown Chart](visuals/screenshot_drawdown.png)
*Continuous drawdown from rolling peak equity. Maximum drawdown of -32.08% reached during the 2016 correction. Recovery timeline visible in the chart.*

---

### Cell 7 — Rolling 30-Day Beta

![Rolling Beta](visuals/screenshot_rolling_beta.png)
*Rolling 30-day beta of AAPL, MSFT, IBM, and SBUX against the S&P 500 from 2007 to 2016. Beta oscillations captured across earnings cycles and market-wide events.*

---

### Cell 8 — Sector Treemap (S&P 500 Market Capitalisation)

![Sector Treemap](visuals/screenshot_sector_treemap.png)
*All 11 GICS sectors proportionally sized by total market capitalisation. Colour represents median sector P/E ratio on a red-yellow-green scale.*

---

### Cell 8 — Valuation Landscape (PE vs PB)

![Valuation Scatter](visuals/screenshot_valuation_scatter.png)
*P/E ratio vs P/B ratio scatter for all 503 S&P 500 companies. Bubble size proportional to market capitalisation. Colour coded by sector. Hover shows company name, EPS, dividend yield, and Graham classification.*

---

### Cell 8 — Monthly Seasonality

![Seasonality Chart](visuals/screenshot_seasonality.png)
*Average AAPL daily return by calendar month with percentage-positive overlay. Bars coloured green for positive months and red for negative months.*

---

### Cell 9 — MACD Technical Indicator

![MACD Chart](visuals/screenshot_macd.png)
*Close price with MA50 (upper panel) and MACD line, signal line, and histogram with green/red colouring (lower panel).*

---

### Cell 10 — Predictive Model: Actual vs Forecast

![Prediction Chart](visuals/screenshot_predictions.png)
*Out-of-sample test set comparison of actual AAPL close price against Linear Regression and Ridge Regression forecasts. 80/20 temporal train-test split with 5-fold TimeSeriesSplit cross-validation.*

---

### Cell 3 — Data Quality Console Output

![Data Quality Tables](visuals/screenshot_data_quality_tables.png)
*APA-formatted console output from Cell 3: missing value analysis, Shapiro-Wilk normality test, Z-score outlier detection, and Augmented Dickey-Fuller stationarity test.*

---

### Cell 6 — OLS Regression and Diagnostics Console Output

![Regression Tables](visuals/screenshot_regression_tables.png)
*APA-formatted OLS regression output (R-squared = 0.974), Breusch-Pagan heteroskedasticity test, Durbin-Watson statistic (0.502), and VIF multicollinearity table.*

---

### Cell 11 — Strategic Recommendations Table

![Recommendations Table](visuals/screenshot_recommendations.png)
*Eight dynamically computed strategic recommendations covering risk-adjusted return, systematic risk, drawdown management, long-term alpha, value investing, predictive deployment, sector rotation, and seasonal patterns.*

---

## Analytical Pipeline

### Cell 1 — Environment Setup
Installs any missing packages, applies the NumPy 2.x compatibility patch for Plotly,
imports all libraries, and defines visual constants and helper functions.

**NumPy 2.x compatibility note:** NumPy 2.0 removed the `bool8` alias. Plotly versions
below 5.14 reference this attribute at import time and will crash with `AttributeError:
module 'numpy' has no attribute 'bool8'`. Cell 1 patches this with two lines before
any Plotly import, making the notebook safe on Anaconda, JupyterLab, and Colab
regardless of the installed NumPy version.

---

### Cell 2 — Data Acquisition and Feature Engineering
Downloads all three datasets and engineers 28 features per trading day.

| Feature Group | Features |
|---|---|
| Returns | DailyReturn, LogReturn, CumulativeReturn |
| Trend | MA20, MA50, MA200 |
| Volatility | Volatility20 (rolling annualised standard deviation) |
| Momentum | RSI(14), MACD(12,26,9), MACD Signal Line, MACD Histogram |
| Risk | Drawdown from rolling peak |
| Volume | Volume 20-day moving average |
| Price | High-Low Spread, Open-Close Change |
| Calendar | Month, Day of Week, Year, Quarter |

---

### Cell 3 — Data Quality Assessment (Tabachnick and Fidell, 2019)

Produces five APA-formatted console tables:

| Table | Test | Finding |
|---|---|---|
| Table 1 | Missing Value Analysis | Zero missing values in source OHLCV; rolling-window fields initialise from day 14/20/50 |
| Table 2 | Descriptive Statistics | Mean, SD, Min, Q1, Median, Q3, Max, Skewness, Kurtosis for six variables |
| Table 3 | Shapiro-Wilk Normality | W statistic and p-value for Close, DailyReturn, LogReturn |
| Table 4 | Z-Score Outlier Detection | 8 extreme returns (1.6%) identified; retained per Mandelbrot (1963) |
| Table 5 | Augmented Dickey-Fuller | Close prices: I(1) non-stationary. Returns: I(0) stationary |

---

### Cell 4 — AAPL Price Intelligence Chart

Three-panel interactive chart rendered inline in Jupyter:

- **Panel 1:** Candlestick with Bollinger Bands (20-day), MA50, MA200
- **Panel 2:** RSI(14) with overbought (>70) and oversold (<30) shaded zones
- **Panel 3:** Daily volume bars (green/red) with 20-day volume moving average

---

### Cell 5 — Multi-Stock EDA

Two interactive charts:

- **Chart 1:** Normalised performance, base 100, for all five tickers 2007-2016
  - AAPL: +807%   MSFT: +207%   IBM: +100%   SBUX: +547%   S&P 500: +40%
- **Chart 2:** Pearson correlation heatmap of daily returns (red-to-green scale)

Followed by **Table 6:** Full Pearson correlation matrix printed in APA format.

---

### Cell 6 — Statistical Analysis

Produces four APA-formatted tables:

**Table 7 — OLS Regression (Close ~ MA20 + Volume + RSI + Volatility20)**

| Metric | Value |
|---|---|
| R-squared | 0.9744 |
| Adjusted R-squared | 0.9742 |
| F-statistic | significant at p < .001 |

**Table 8 — Regression Diagnostics**

| Test | Statistic | Interpretation |
|---|---|---|
| Breusch-Pagan | p < .05 | Heteroscedastic residuals; robust standard errors advised |
| Durbin-Watson | 0.502 | Positive autocorrelation; expected in price-level regression |

**Table 9 — VIF Multicollinearity**
MA20 shows high VIF by construction (collinear with Close in time series). This is
documented, not corrected, as the regression is illustrative rather than causal.

**Table 10 — Pearson and Spearman Correlation (Close vs Volume)**
Both methods reported. Spearman rho preferred for monotonic non-linear relationships.

---

### Cell 7 — Risk Analytics

**Table 11 — AAPL Risk and Performance Metrics (2015-2017)**

| Metric | Value |
|---|---|
| Sharpe Ratio | 0.239 |
| Maximum Drawdown | -32.08% |
| Beta vs S&P 500 | 0.961 |
| VaR (95%, 1-day) | -2.49% |

Three inline charts:
- Drawdown from rolling peak (filled area, red)
- Return distribution vs theoretical normal (fat-tail visualisation)
- Rolling 30-day Beta for all four stocks vs S&P 500 (2007-2016)

**Table 12 — Multi-Stock Risk Comparison:** annualised return, volatility, Sharpe, VaR,
and 10-year total return for AAPL, MSFT, IBM, SBUX, and S&P 500.

---

### Cell 8 — Business Analytics

**Chart 1 — Sector Treemap:** All 11 S&P 500 sectors sized by total market
capitalisation, coloured by median P/E ratio on a red-yellow-green scale.

**Chart 2 — Valuation Landscape:** PE ratio vs P/B ratio scatter plot for 499 S&P 500
companies. Bubble size represents market capitalisation. Colour represents sector.

**Chart 3 — Sector Bubble Chart:** Median PE vs average dividend yield with bubble
size representing number of constituent companies.

**Chart 4 — Monthly Seasonality:** Average daily return by calendar month with
percentage-positive overlay.

**Table 13 — Sector Summary Statistics:** Company count, total market cap, median PE,
median PB, and average dividend yield for all 11 sectors.

**Table 14 — Graham Value Screen:** Top 15 deep-value S&P 500 stocks scoring 100/100
on the composite Graham screen, ranked by market capitalisation.

---

### Cell 9 — MACD and RSI Technical Indicators

Two multi-panel charts:

- **MACD Chart:** Close price with MA50 (upper panel) + MACD line, signal line, and
  histogram with green/red colouring (lower panel)
- **RSI Chart:** Close price (upper panel) + RSI with overbought/oversold zones (lower panel)

---

### Cell 10 — Predictive Analytics

**Training protocol:** 80/20 temporal split with 5-fold TimeSeriesSplit cross-validation.

**Table 15 — Model Comparison (out-of-sample test set)**

| Model | RMSE | R-squared | MAPE | CV R-squared |
|---|---|---|---|---|
| Linear Regression | 1.2897 | 0.9650 | 0.719% | 0.642 |
| Ridge (alpha=1.0) | 1.2928 | 0.9648 | 0.739% | 0.213 |

**Table 16 — Feature Importance:** Absolute standardised coefficients from Linear
Regression normalised to sum to 100%.

---

### Cell 11 — Executive Recommendations

**Table 17 — Strategic Recommendations:** Eight business findings, each with a
concise finding statement and a specific, actionable recommendation derived directly
from the computed analytical outputs. All numerical values populated dynamically from
computed results, not hardcoded.

---

## Key Results Summary

| Finding | Value | Implication |
|---|---|---|
| AAPL 2-year Sharpe Ratio | 0.239 | Positive but below the 0.5 threshold for strong risk-adjusted return |
| AAPL Maximum Drawdown | -32.08% | Significant drawdown tolerance required |
| AAPL Beta vs S&P 500 | 0.961 | Near-market systematic risk |
| AAPL Daily VaR (95%) | -2.49% | Maximum expected single-day loss 19 out of 20 trading days |
| AAPL 10-year total return | +807% | Outperformed S&P 500 (+40%) by 767 percentage points |
| OLS Regression R-squared | 0.974 | MA20 and RSI explain 97.4% of variance in daily close price |
| Linear Regression R-squared (test) | 0.965 | Strong out-of-sample predictive accuracy |
| Prediction MAPE | 0.719% | Average prediction error below 1% of actual price |
| Deep Value stocks (Graham 100/100) | 64 | Out of 503 S&P 500 constituents |

---

## How to Run

### Option 1 — JupyterLab or Jupyter Notebook

```bash
git clone https://github.com/novestuschirchir4-oss/stock-market-intelligence.git
cd stock-market-intelligence
pip install -r requirements.txt
jupyter lab notebooks/stock_intelligence_platform.ipynb
```

Run cells in order from Cell 1 to Cell 11. Each cell is independent and clearly labelled.

### Option 2 — Google Colab

1. Open https://colab.research.google.com
2. File, then Open notebook, then GitHub tab
3. Enter: `novestuschirchir4-oss/stock-market-intelligence`
4. Select `notebooks/stock_intelligence_platform.ipynb`
5. Run All

### Option 3 — Python Script

```bash
python src/stock_intelligence_platform.py
```

### Option 4 — Interactive Dashboard (no Python required)

Open `dashboard/stock_intelligence_dashboard.html` directly in any web browser.
All 12 charts are fully interactive with hover, zoom, and pan. No server required.

---

## Compatibility

| Environment | Status |
|---|---|
| JupyterLab 3+ | Verified |
| Jupyter Notebook 6+ | Verified |
| Google Colab | Verified |
| Anaconda (Windows, macOS, Linux) | Verified |
| Python 3.10+ | Required |
| NumPy 2.x | Supported (patch in Cell 1) |

---

## Methodological References

Graham, B., and Dodd, D. (1934). *Security analysis*. McGraw-Hill.

Mandelbrot, B. (1963). The variation of certain speculative prices. *The Journal of Business, 36*(4), 394-419. https://doi.org/10.1086/294632

Sharpe, W. F. (1964). Capital asset prices: A theory of market equilibrium under conditions of risk. *The Journal of Finance, 19*(3), 425-442. https://doi.org/10.1111/j.1540-6261.1964.tb02865.x

Sharpe, W. F. (1966). Mutual fund performance. *The Journal of Business, 39*(1), 119-138. https://doi.org/10.1086/294846

Sortino, F. A., and van der Meer, R. (1991). Downside risk. *The Journal of Portfolio Management, 17*(4), 27-31. https://doi.org/10.3905/jpm.1991.409343

Tabachnick, B. G., and Fidell, L. S. (2019). *Using multivariate statistics* (7th ed.). Pearson Education.

---

## License

This project is released under the MIT License. The underlying datasets carry their
own licenses as documented in `data/DATA_SOURCES.md`.

---

*Built by Novestus Chirchir — Data Analyst*

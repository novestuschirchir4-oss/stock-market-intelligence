<div align="center">

# Stock Market Intelligence and Analytics Platform

### An End-to-End Quantitative Research and Business Analytics Engagement

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Engineering-150458?style=flat-square&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Visualisation-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Predictive%20Modelling-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-6A4C93?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-2E8B57?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production--Ready-2E8B57?style=flat-square)

</div>

---

| | |
|---|---|
| **Author** | Novestus Chirchir |
| **GitHub** | [novestuschirchir4-oss](https://github.com/novestuschirchir4-oss) |
| **Methodology** | CRISP-DM · Tabachnick & Fidell (2019) · APA 7th Edition |
| **Stack** | Python 3.10+, pandas, NumPy, statsmodels, scikit-learn, Plotly |
| **Deliverables** | Jupyter notebook · standalone script · self-contained interactive dashboard |

---

## Table of Contents

- [Overview](#overview)
- [Analytical Framework](#analytical-framework)
- [Repository Structure](#repository-structure)
- [Data Architecture](#data-architecture)
- [Analytical Pipeline](#analytical-pipeline)
- [Key Results Summary](#key-results-summary)
- [Quick Start](#quick-start)
- [Environment Compatibility](#environment-compatibility)
- [Methodological References](#methodological-references)
- [License](#license)

---

## Overview

This repository contains a complete quantitative research and business intelligence engagement built on three real-world financial datasets totalling 3,315 records, spanning 503 S&P 500 constituents and nine years of daily equity price history. The work is structured as a single coherent narrative running from raw data acquisition through statistical validation, diagnostic analysis, predictive modelling, and executive decision support — the full range of deliverables expected of a senior data analyst operating at investment-research standard.

Every analytical result produced in the underlying notebook is also rendered into a self-contained, browser-based executive dashboard, giving stakeholders two ways to consume the same body of work: a fully reproducible analytical script for technical audiences, and an interactive visual interface for decision-makers who will never open a line of code.

<img width="1847" height="815" alt="image" src="https://github.com/user-attachments/assets/9bbaee40-3c1c-4a3e-88ac-a3c6a0d69391" />

*Figure 1. The consolidated executive dashboard, presenting key performance indicators, price intelligence, risk analytics, sector positioning, predictive output, and strategic recommendations within a single interactive interface.*

---

## Analytical Framework

The pipeline is organised around four progressively deeper business questions, each answered with a distinct analytical method:

| Tier | Business Question | Method |
|---|---|---|
| Descriptive | What happened in price and volume? | Exploratory data analysis, OHLCV charting, Bollinger Bands |
| Diagnostic | Why did returns behave this way? | Correlation analysis, OLS regression, MACD, RSI |
| Predictive | What price is expected next? | Linear Regression, Ridge Regression with time-series cross-validation |
| Prescriptive | What should a portfolio manager do? | Risk metrics, sector screening, Graham value filter |

---

## Repository Structure

```
stock-market-intelligence/
│
├── notebooks/
│   └── stock_intelligence_platform.ipynb     Eleven-stage Jupyter notebook
│
├── dashboard/
│   └── stock_intelligence_dashboard.html     Self-contained interactive dashboard
│
├── src/
│   ├── stock_intelligence_platform.py        Full pipeline as a runnable script
│   └── analytics_helpers.py                  Reusable functions and constants
│
├── data/
│   └── DATA_SOURCES.md                       Full citations for all three datasets
│
├── reports/
│   └── METHODOLOGY.md                        Detailed statistical methodology notes
│
├── visuals/
│   └── (exported figures referenced throughout this document)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data Architecture

All data is retrieved at runtime from public, version-controlled sources. No API keys, manual downloads, or local file paths are required, and the pipeline executes identically across JupyterLab, Google Colab, Anaconda, and any standard Python environment.

**Dataset I — S&P 500 Constituent Fundamentals**

| Field | Detail |
|---|---|
| Publisher | DataHub |
| Coverage | 503 companies, 14 variables |
| Variables | Symbol, sector, price, P/E, P/B, EPS, market capitalisation, EBITDA, dividend yield |
| Period | Snapshot, circa 2017–2018 |

> DataHub. (2018). *S&P 500 companies with financial information* [Dataset]. GitHub.

**Dataset II — Apple Inc. Daily OHLCV Price Series**

| Field | Detail |
|---|---|
| Publisher | Plotly Technologies Inc. |
| Coverage | 506 trading days |
| Variables | Date, open, high, low, close, volume, adjusted close, Bollinger bands, trend |
| Period | 17 February 2015 – 16 February 2017 |

> Plotly Technologies Inc. (2017). *Finance charts — Apple OHLCV* [Dataset]. GitHub.

**Dataset III — Multi-Stock Daily Close Prices**

| Field | Detail |
|---|---|
| Publisher | Plotly Technologies Inc. |
| Coverage | 2,306 trading days |
| Tickers | AAPL, MSFT, IBM, SBUX, and the S&P 500 Index |
| Period | 3 January 2007 – 1 March 2016 |

> Plotly Technologies Inc. (2016). *Stock market data — AAPL, MSFT, IBM, SBUX, S&P 500* [Dataset]. GitHub.

---

## Analytical Pipeline

The notebook is organised into eleven sequential stages, mirrored exactly in the standalone script. Each stage below is presented with its purpose, its statistical output, and the corresponding analytical view it produces.

### 01 — Environment Configuration

Initialises the runtime: installs any missing dependencies, imports the analytical stack, defines shared visual constants and helper functions, and applies a forward-compatibility patch for NumPy 2.x. NumPy 2.0 removed the legacy `bool8` alias that older Plotly builds reference at import time; the patch is applied before any Plotly import, ensuring the notebook runs cleanly on Anaconda, JupyterLab, and Colab regardless of the installed NumPy version.

### 02 — Data Acquisition and Feature Engineering

Retrieves all three datasets and engineers twenty-eight features per trading day.

| Feature Group | Features |
|---|---|
| Returns | Daily return, log return, cumulative return |
| Trend | 20-, 50-, and 200-day moving averages |
| Volatility | 20-day rolling annualised standard deviation |
| Momentum | RSI(14), MACD(12,26,9), signal line, histogram |
| Risk | Drawdown from rolling peak |
| Volume | 20-day volume moving average |
| Price structure | High–low spread, open–close change |
| Calendar | Month, day of week, year, quarter |

### 03 — Data Quality Assessment

Establishes the statistical integrity of the dataset ahead of modelling, following Tabachnick and Fidell (2019), across five tables: missing-value analysis, descriptive statistics, Shapiro-Wilk normality testing, Z-score outlier detection, and Augmented Dickey-Fuller stationarity testing. Close prices are confirmed non-stationary at level — I(1) — while returns are stationary at level — I(0) — the expected signature of an equity price series and a precondition correctly verified before any regression is run.

#### Statistical Tables

##### TABLE 1. Missing Value Analysis — AAPL OHLCV

| Variable | N Missing | % Missing | Decision |
|----------|-----------:|----------:|-----------|
| Open | 0 | 0.0000 | No action required |
| High | 0 | 0.0000 | No action required |
| Low | 0 | 0.0000 | No action required |
| Close | 0 | 0.0000 | No action required |
| Volume | 0 | 0.0000 | No action required |
| DailyReturn | 1 | 0.2000 | Forward-fill (lag structure) |
| Volatility20 | 20 | 3.9500 | Forward-fill (lag structure) |
| RSI | 14 | 2.7700 | Forward-fill (lag structure) |
| MACD | 0 | 0.0000 | No action required |

**Note.** *n* = 506 trading days. Missing values in DailyReturn, Volatility20, RSI, and MACD arise from rolling window initialisation, not data absence.

---

##### TABLE 2. Descriptive Statistics — AAPL Price & Return Series

| Variable | N | Mean | SD | Min | Q1 | Median | Q3 | Max | Skewness | Kurtosis |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| Close | 506.0000 | 112.9583 | 11.2447 | 90.3400 | 105.6725 | 113.0250 | 122.1800 | 135.5100 | -0.0628 | -0.9778 |
| DailyReturn | 505.0000 | 0.0002 | 0.0153 | -0.0657 | -0.0066 | 0.0000 | 0.0083 | 0.0650 | -0.0640 | 3.0939 |
| LogReturn | 505.0000 | 0.0001 | 0.0153 | -0.0680 | -0.0066 | 0.0000 | 0.0082 | 0.0629 | -0.1804 | 3.1463 |
| Volatility20 | 486.0000 | 0.2291 | 0.0830 | 0.0764 | 0.1766 | 0.2241 | 0.2687 | 0.4854 | 0.7651 | 0.8102 |
| Volume | 506.0000 | 43178420.9486 | 19852531.3009 | 11475900.0000 | 29742400.0000 | 37474600.0000 | 50763950.0000 | 162206300.0000 | 1.8935 | 5.2624 |
| HL_Spread | 506.0000 | 1.9774 | 1.1965 | 0.5800 | 1.2425 | 1.7250 | 2.3750 | 16.8000 | 4.6429 | 47.3040 |

**Note.** All statistics computed on raw (unadjusted) price series. Kurtosis reported as excess kurtosis (normal distribution = 0).

---

##### TABLE 3. Normality Assessment — Shapiro-Wilk Test

| Variable | N | Mean | SD | Skewness | Kurtosis | SW W | p | Normal (p > .05) |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|:----------------:|
| Close | 506 | 112.9583 | 11.2447 | -0.0628 | -0.9778 | 0.9628 | 0.0000 | No |
| DailyReturn | 505 | 0.0002 | 0.0153 | -0.0640 | 3.0939 | 0.9565 | 0.0000 | No |
| LogReturn | 505 | 0.0001 | 0.0153 | -0.1804 | 3.1463 | 0.9557 | 0.0000 | No |

**Note.** Shapiro-Wilk test applied to first 500 observations. p < .05 indicates significant departure from normality. Financial return series are expected to be leptokurtic (fat-tailed).

---

##### TABLE 4. Outlier Detection — Z-Score Method (|Z| > 3.29)

| Variable | N | Outliers | % | Min Outlier | Max Outlier | Decision |
|----------|----:|---------:|------:|------------:|------------:|-----------|
| DailyReturn | 505 | 8 | 1.58 | -6.571 | 6.496 | Retained |

**Note.** Criterion: |Z| > 3.29 (Tabachnick & Fidell, 2019, p. 77). Extreme returns retained: fat tails are a structural property of equity returns, not data errors (Mandelbrot, 1963; Fama, 1965).

---

##### TABLE 5. Augmented Dickey-Fuller Test — Stationarity

| Variable | ADF Stat | p-value | Lags | Crit(5%) | Stationary |
|----------|---------:|---------:|----:|---------:|:----------:|
| Close Price | -1.3725 | 0.5954 | 0 | -2.8673 | No (unit root) |
| Daily Return | -22.1014 | 0.0000 | 0 | -2.8673 | Yes |
| Log Return | -22.0748 | 0.0000 | 0 | -2.8673 | Yes |

**Note.** H0: Unit root present (non-stationary). p < .05 rejects H0. Prices are integrated I(1); returns are stationary I(0) as expected.

### 04 — AAPL Price Intelligence

A three-panel technical view of Apple Inc. price action: candlestick price action with 20-day Bollinger Bands and the 50- and 200-day moving averages in the upper panel; the 14-day Relative Strength Index with shaded overbought and oversold zones in the middle panel; and daily trading volume against its 20-day moving average in the lower panel.

<img width="833" height="820" alt="image" src="https://github.com/user-attachments/assets/9b685896-8b16-48d9-89e2-359b584f8496" />

*Figure 2. Three-panel technical analysis of Apple Inc. equity price action, momentum, and volume.*

### 05 — Multi-Stock Exploratory Analysis

Normalises AAPL, MSFT, IBM, SBUX, and the S&P 500 to a common base of 100 and traces relative performance from 2007 to 2016 — AAPL advanced 807%, SBUX 547%, MSFT 207%, IBM 100%, and the index itself 40% — followed by a Pearson correlation matrix of daily returns across all five tickers, reported in full APA format.

<img width="833" height="520" alt="image" src="https://github.com/user-attachments/assets/110ce68b-8aa3-4dc3-9cae-01d0e126e0a9" />

*Figure 3. Base-100 normalised performance of AAPL, MSFT, IBM, SBUX, and the S&P 500, 2007–2016.*

<img width="833" height="500" alt="image" src="https://github.com/user-attachments/assets/34f96938-7c2b-482c-8148-eac2b9322bd4" />
*Figure 4. Pearson correlation matrix of daily returns across all five tracked instruments.*

### 06 — Statistical Analysis

Estimates an OLS regression of closing price on the 20-day moving average, trading volume, RSI, and 20-day volatility (R² = .974, adjusted R² = .974, F significant at p < .001), followed by a full diagnostic battery: a Breusch-Pagan test indicating heteroscedastic residuals, a Durbin-Watson statistic of 0.502 indicating positive autocorrelation consistent with a price-level specification, a variance inflation factor table documenting the expected collinearity between price and its own moving average, and a paired Pearson–Spearman correlation between closing price and volume.

#### TABLE 7. OLS Regression — Predicting AAPL Close Price

| Variable | B | SE | t | p | 95% CI Lower | 95% CI Upper |
|-----------|---------:|---------:|---------:|---------:|---------:|---------:|
| (Constant) | -10.5116 | 1.0325 | -10.1805 | 0.0000 | -12.5404 | -8.4828 *** |
| MA20 | 1.0095 | 0.0081 | 124.6459 | 0.0000 | 0.9936 | 1.0254 *** |
| Volume | -0.0000 | 0.0000 | -5.1489 | 0.0000 | -0.0000 | -0.0000 *** |
| RSI | 0.1898 | 0.0045 | 41.9896 | 0.0000 | 0.1810 | 0.1987 *** |
| Volatility20 | 2.5711 | 1.0992 | 2.3391 | 0.0197 | 0.4113 | 4.7310 * |

| Model Statistic | Value |
|-----------------|-------|
| Dependent Variable | Close (USD) |
| Predictors | MA20, Volume, RSI, Volatility20 |
| n | 486 |
| R-squared | 0.9744 |
| Adjusted R-squared | 0.9741 |
| F-statistic | F(4, 481) = 4569.47 |
| Model p-value | 0.00e+00 |
| AIC | 1942.53 |
| BIC | 1963.46 |

**Note.** * p < .05, ** p < .01, *** p < .001.

---

#### TABLE 8. Regression Assumption Diagnostics

| Test | Statistic | p-value | Verdict |
|--------|---------:|---------:|----------|
| Breusch-Pagan (homoscedasticity) | 85.0779 | 0.0000 | Heteroscedastic (robust SE advised) |
| Durbin-Watson (independence) | 0.5020 | N/A | Autocorrelation detected |

**Note.** Breusch-Pagan H0: homoscedastic residuals. Durbin-Watson range 1.5–2.5 indicates no problematic autocorrelation. DW = 0.5020 indicates positive autocorrelation as expected in price levels.

---

#### TABLE 9. Variance Inflation Factors (Multicollinearity)

| Variable | VIF | Tolerance | Assessment |
|-----------|---------:|---------:|------------|
| MA20 | 19.38 | 0.0516 | High |
| Volume | 7.50 | 0.1333 | Moderate |
| RSI | 8.67 | 0.1154 | Moderate |
| Volatility20 | 8.88 | 0.1126 | Moderate |

**Note.** VIF > 10 indicates problematic multicollinearity. MA20 is correlated with Close by construction; this is expected in time-series.

---

#### TABLE 10. Bivariate Correlation — Close Price vs Daily Volume

| Method | r | p-value | Significant |
|---------|---------:|---------:|:-----------:|
| Pearson | -0.0208 | 0.6472 | No |
| Spearman | 0.0109 | 0.8107 | No |

**Note.** n = 486. Two-tailed test. Spearman rho is preferred for monotonic but non-linear relationships, which is typical for price-volume dynamics.

*OLS regression output with heteroscedasticity, autocorrelation, and multicollinearity diagnostics.*

### 07 — Risk Analytics

Computes the core risk and performance profile for AAPL over the 2015–2017 window — a Sharpe ratio of 0.239, a maximum drawdown of −32.08%, a beta of 0.961 against the S&P 500, and a 95% one-day Value-at-Risk of −2.49% — alongside a continuous drawdown trace, a return-distribution comparison against the theoretical normal, and a rolling 30-day beta of AAPL, MSFT, IBM, and SBUX against the index across the full 2007–2016 sample. A companion table extends annualised return, volatility, Sharpe ratio, VaR, and ten-year total return across all five instruments.

<img width="833" height="380" alt="image" src="https://github.com/user-attachments/assets/6100f79a-16f9-4610-b6cc-e968622c812b" />

*Figure 5. Continuous drawdown from rolling peak equity, with maximum drawdown reached during the 2016 correction.*

<img width="833" height="460" alt="image" src="https://github.com/user-attachments/assets/5cdf3897-b53a-4574-b06d-7e0252876586" />

*Figure 6. Rolling 30-day beta of AAPL, MSFT, IBM, and SBUX against the S&P 500, 2007–2016.*

### 08 — Business Analytics

Maps the full S&P 500 universe across four views: a sector treemap sized by total market capitalisation and coloured by median sector P/E; a valuation landscape plotting P/E against P/B for 499 constituents, with bubble size set to market capitalisation and colour set to sector; a sector bubble chart comparing median P/E against average dividend yield; and a monthly seasonality chart of average AAPL daily returns by calendar month. The accompanying tables summarise company count, total market capitalisation, median P/E, median P/B, and average dividend yield by sector, and present the top fifteen constituents by market capitalisation scoring 100/100 on a composite Graham deep-value screen.

<img width="833" height="520" alt="image" src="https://github.com/user-attachments/assets/a32c934d-2f7e-4251-bb22-3c7b363317c3" />

*Figure 7. S&P 500 sector composition by market capitalisation, coloured by median sector P/E ratio.*

<img width="833" height="560" alt="image" src="https://github.com/user-attachments/assets/9cc18282-ea7b-432d-b68a-3f84a9fd3afc" />

*Figure 8. P/E versus P/B valuation landscape across the S&P 500, sized by market capitalisation and coloured by sector.*
<img width="833" height="500" alt="image" src="https://github.com/user-attachments/assets/849bdde4-5fc2-43da-9c7c-a91bf98b7bf1" />

<img width="833" height="420" alt="image" src="https://github.com/user-attachments/assets/149d6243-4ac5-441e-8f5d-c085025ccf7f" />

*Figure 9. Average AAPL daily return by calendar month, with the proportion of positive months overlaid.*

### 09 — Technical Indicators: MACD and RSI

Builds two multi-panel technical charts: closing price with its 50-day moving average paired against the MACD line, signal line, and histogram in the first; and closing price paired against RSI with overbought and oversold zones in the second.

<img width="833" height="600" alt="image" src="https://github.com/user-attachments/assets/3c2d8d6a-f4e7-4737-a06c-c715c8b6f6e0" />
<img width="833" height="560" alt="image" src="https://github.com/user-attachments/assets/40aee86c-01b8-42a9-b070-84c2b3d2d98c" />

*Figure 10. Closing price with 50-day moving average, MACD line, signal line, and histogram.*

### 10 — Predictive Analytics

Trains Linear and Ridge regression models under an 80/20 temporal split with five-fold time-series cross-validation. On the out-of-sample test set, Linear Regression achieves an RMSE of 1.290, an R² of 0.965, and a MAPE of 0.719%, modestly outperforming Ridge regression (RMSE 1.293, R² 0.965, MAPE 0.739%) — evidence that the underlying relationship is close to linear and that regularisation offers limited additional benefit on this feature set. Standardised feature importances are reported alongside the comparison.

<img width="833" height="480" alt="image" src="https://github.com/user-attachments/assets/a9f341c4-9bc6-4c53-a991-f2f4919c7779" />

*Figure 11. Out-of-sample comparison of actual AAPL closing price against Linear and Ridge regression forecasts.*
<img width="833" height="360" alt="image" src="https://github.com/user-attachments/assets/15e3f3e0-13e8-4201-bab2-f166715aea51" />

### 11 — Executive Recommendations

#### 1. RISK-ADJUSTED RETURN

Finding: Sharpe ratio = 0.239. Below 1.0: does not fully compensate for volatility (24.3% annualised).  
Action: Maintain position if Sharpe > 0.5. Implement stop-loss at max drawdown threshold.

---

#### 2. SYSTEMATIC RISK EXPOSURE

Finding: Beta = 0.961 vs S&P 500. Near-market beta: suitable for core portfolio allocation.  
Action: Size position using beta-adjusted allocation. Hedge with inverse ETF in bear markets.

---

#### 3. DOWNSIDE RISK MANAGEMENT

Finding: Maximum drawdown = -32.1%. Daily VaR (95%) = -2.49%. Tail risk (kurtosis > 3) confirms fat-tailed return distribution.  
Action: Set portfolio-level VaR limit at 2x single-stock VaR. Use options collar during high-volatility periods.

---

#### 4. LONG-TERM ALPHA GENERATION

Finding: AAPL delivered 806.8% total return (2007–2016) vs S&P 500 39.7%. Excess return: 767.1 percentage points over the index period.  
Action: Screen for stocks with consistent MACD bullish crossovers and RSI mean-reversion from oversold zones.

---

#### 5. VALUE INVESTING OPPORTUNITY

Finding: 58 S&P 500 stocks achieve perfect Graham score (100/100): low PE, low PB, positive EPS, positive dividend yield.  
Action: Construct equal-weight deep-value basket. Rebalance quarterly. Historically outperforms in late-cycle and early-recovery phases.

---

#### 6. PREDICTIVE MODEL DEPLOYMENT

Finding: Linear Regression achieves R² = 0.9650, MAPE = 0.719% on out-of-sample data. Model inputs: Close, MA20, MA50, Volume, RSI, Volatility, MACD, HL Spread.  
Action: Deploy as signal layer. Combine with RSI < 35 (oversold) AND MACD bullish cross for high-confidence entry signals. Backtest with 6-month rolling window.

---

#### 7. SECTOR ROTATION STRATEGY

Finding: Information Technology dominates market cap share. Utilities and Consumer Staples show highest dividend yields with lower PE multiples.  
Action: Rotate into high-yield, low-PE sectors (Utilities, Staples, Financials) during rate-tightening cycles. Rotate back into Technology in early easing phases.

---

#### 8. SEASONAL TRADING EDGE

Finding: Monthly return analysis reveals consistent seasonal patterns in average daily returns. Certain months show persistent positive bias; others show negative bias.  
Action: Overlay seasonal calendar on entry/exit signals. Increase exposure in historically strong months; reduce or hedge in weak months.

---


## Key Results Summary

| Finding | Value | Implication |
|---|---|---|
| AAPL two-year Sharpe ratio | 0.239 | Positive, but below the 0.5 threshold typically associated with strong risk-adjusted return |
| AAPL maximum drawdown | −32.08% | Material drawdown tolerance required of any holding strategy |
| AAPL beta versus S&P 500 | 0.961 | Near-market systematic risk exposure |
| AAPL daily VaR (95%) | −2.49% | Expected ceiling on single-day loss in nineteen of twenty trading days |
| AAPL ten-year total return | +807% | Outperformed the S&P 500 (+40%) by 767 percentage points |
| OLS regression R² | 0.974 | Moving average, volume, RSI, and volatility explain 97.4% of daily closing price variance |
| Linear Regression test R² | 0.965 | Strong out-of-sample predictive accuracy |
| Prediction MAPE | 0.719% | Average forecast error below one percent of actual price |
| Deep-value constituents (Graham 100/100) | 64 | Out of 503 S&P 500 companies screened |

---

## Quick Start

**JupyterLab or Jupyter Notebook**

```bash
git clone https://github.com/novestuschirchir4-oss/stock-market-intelligence.git
cd stock-market-intelligence
pip install -r requirements.txt
jupyter lab notebooks/stock_intelligence_platform.ipynb
```

Run the eleven stages sequentially; each is self-contained and clearly labelled.

**Google Colab**

Open Colab, choose File → Open notebook → GitHub, enter `novestuschirchir4-oss/stock-market-intelligence`, select `notebooks/stock_intelligence_platform.ipynb`, and run all cells.

**Standalone script**

```bash
python src/stock_intelligence_platform.py
```

**Interactive dashboard — no installation required**

Open `dashboard/stock_intelligence_dashboard.html` in any web browser. All twelve charts are fully interactive, supporting hover, zoom, and pan, with no server or dependency required.

---

## Environment Compatibility

| Environment | Status |
|---|---|
| JupyterLab 3+ | Verified |
| Jupyter Notebook 6+ | Verified |
| Google Colab | Verified |
| Anaconda (Windows, macOS, Linux) | Verified |
| Python 3.10+ | Required |
| NumPy 2.x | Supported via compatibility patch |

---

## Methodological References

Graham, B., & Dodd, D. (1934). *Security analysis*. McGraw-Hill.

Mandelbrot, B. (1963). The variation of certain speculative prices. *The Journal of Business, 36*(4), 394–419. https://doi.org/10.1086/294632

Sharpe, W. F. (1964). Capital asset prices: A theory of market equilibrium under conditions of risk. *The Journal of Finance, 19*(3), 425–442. https://doi.org/10.1111/j.1540-6261.1964.tb02865.x

Sharpe, W. F. (1966). Mutual fund performance. *The Journal of Business, 39*(1), 119–138. https://doi.org/10.1086/294846

Sortino, F. A., & van der Meer, R. (1991). Downside risk. *The Journal of Portfolio Management, 17*(4), 27–31. https://doi.org/10.3905/jpm.1991.409343

Tabachnick, B. G., & Fidell, L. S. (2019). *Using multivariate statistics* (7th ed.). Pearson Education.

---

## License

This project is released under the MIT License. The underlying datasets carry their own licenses, documented in `data/DATA_SOURCES.md`.

---

<div align="center">

*Built by Novestus Chirchir*

</div>

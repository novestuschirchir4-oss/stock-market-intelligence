# Analytical Methodology

## Framework

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining)
methodology throughout, structured across six phases: Business Understanding, Data
Understanding, Data Preparation, Modelling, Evaluation, and Deployment.

Data quality assessment adheres to the standards set out in **Tabachnick and Fidell (2019)**,
covering missing value analysis, normality testing, outlier detection, and stationarity
verification before any inferential statistics are applied.

---

## Phase 1 — Business Understanding

**Objective:** Transform publicly available equity and fundamental data into measurable
business intelligence supporting portfolio construction, risk management, sector rotation,
and value screening decisions.

**Key Performance Indicators defined:**

| Category    | KPI                        | Formula                              |
|-------------|----------------------------|--------------------------------------|
| Return      | Annualised Return          | Mean daily return x 252              |
| Risk        | Annualised Volatility      | Daily standard deviation x sqrt(252) |
| Risk-Adj    | Sharpe Ratio               | Ann. Return / Ann. Volatility        |
| Risk-Adj    | Sortino Ratio              | Ann. Return / Downside Deviation     |
| Drawdown    | Maximum Drawdown           | (Price - Rolling Peak) / Rolling Peak |
| Tail Risk   | Value at Risk (95%)        | 5th percentile of daily returns      |
| Tail Risk   | Conditional VaR (95%)      | Mean of returns below VaR threshold  |
| Systematic  | Beta                       | Cov(stock, market) / Var(market)     |
| Systematic  | Jensen Alpha               | Excess return above CAPM expectation |
| Valuation   | Graham Score               | Composite PE, PB, EPS, Dividend screen |

---

## Phase 2 — Data Understanding

Three real-world datasets sourced from public GitHub repositories. Full citations in
`data/DATA_SOURCES.md`.

| Dataset                  | Records | Period        | Key Variables              |
|--------------------------|---------|---------------|----------------------------|
| S&P 500 Fundamentals     | 503     | 2017-2018     | PE, PB, EPS, MarketCap, EBITDA |
| Apple OHLCV              | 506     | 2015-2017     | OHLCV, Bollinger Bands     |
| Multi-Stock Close Prices | 2,306   | 2007-2016     | AAPL, MSFT, IBM, SBUX, GSPC |

---

## Phase 3 — Data Preparation

### Missing Value Analysis
Assessed for all numeric columns. Missing values in rolling indicators (RSI, MACD,
Volatility20, MA20, MA50, MA200) arise from window initialisation, not source data
absence, and require no imputation.

### Normality Assessment
Shapiro-Wilk W statistic applied to Close Price, Daily Return, and Log Return series
(first 500 observations per Tabachnick and Fidell, 2019). Financial return series are
expected to be leptokurtic; departure from normality is documented rather than corrected.

### Outlier Detection
Z-score method with threshold |Z| > 3.29 (Tabachnick and Fidell, 2019, p. 77). Extreme
returns are retained: fat tails are a structural property of equity return distributions
(Mandelbrot, 1963; Fama, 1965), not errors.

### Stationarity
Augmented Dickey-Fuller test applied to price levels and return series. Price levels are
I(1) non-stationary; daily returns are I(0) stationary, consistent with the efficient
market hypothesis in weak form.

---

## Phase 4 — Feature Engineering

28 features engineered per trading day from raw OHLCV:

| Feature Group  | Features                                          |
|----------------|---------------------------------------------------|
| Returns        | DailyReturn, LogReturn, CumulativeReturn          |
| Trend          | MA20, MA50, MA200                                 |
| Volatility     | Volatility20 (rolling annualised)                 |
| Momentum       | RSI(14), MACD(12,26,9), MACD Signal, MACD Histogram |
| Risk           | Drawdown from rolling peak                        |
| Volume         | Volume_MA20                                       |
| Price          | HL_Spread, OC_Change                              |
| Calendar       | Month, DayOfWeek, Year, Quarter                   |
| Multi-stock    | Normalised price (base 100), beta30 (rolling)     |

---

## Phase 5 — Statistical Analysis

### OLS Regression
Dependent variable: AAPL Close Price.
Predictors: MA20, Volume, RSI, Volatility20.
Diagnostics: Breusch-Pagan (homoscedasticity), Durbin-Watson (serial independence),
VIF (multicollinearity).

### Correlation Analysis
Pearson r and Spearman rho computed for bivariate relationships.
Pearson correlation matrix computed for all five return series (2007-2016).

### Risk Decomposition
Sharpe (1966), Sortino and van der Meer (1991), CAPM Beta (Sharpe, 1964),
Jensen Alpha, Basel III VaR and CVaR framework.

---

## Phase 6 — Predictive Modelling

| Model              | Regularisation | Cross-Validation           |
|--------------------|---------------|----------------------------|
| Linear Regression  | None (OLS)    | 5-fold TimeSeriesSplit     |
| Ridge Regression   | L2, alpha=1.0 | 5-fold TimeSeriesSplit     |

**Evaluation metrics:** RMSE, MAE, R-squared, MAPE, CV R-squared.
**Protocol:** 80/20 temporal split. No data leakage — test set is strictly future
observations relative to training set.

**Feature importance:** Absolute standardised coefficients from Linear Regression,
normalised to sum to 100%.

---

## Statistical Software

All analysis conducted in Python 3.10+. Key packages:

| Package      | Version  | Purpose                              |
|--------------|----------|--------------------------------------|
| pandas       | >= 1.5   | Data manipulation                    |
| numpy        | >= 1.24  | Numerical computation                |
| scipy        | >= 1.9   | Statistical tests                    |
| statsmodels  | >= 0.14  | OLS regression, diagnostics          |
| scikit-learn | >= 1.2   | Machine learning, cross-validation   |
| plotly       | >= 5.14  | Interactive visualisation            |

---

## References

Graham, B., and Dodd, D. (1934). *Security analysis*. McGraw-Hill.

Mandelbrot, B. (1963). The variation of certain speculative prices.
*The Journal of Business, 36*(4), 394-419.

Sharpe, W. F. (1964). Capital asset prices: A theory of market equilibrium under
conditions of risk. *The Journal of Finance, 19*(3), 425-442.

Sharpe, W. F. (1966). Mutual fund performance. *The Journal of Business, 39*(1), 119-138.

Sortino, F. A., and van der Meer, R. (1991). Downside risk.
*The Journal of Portfolio Management, 17*(4), 27-31.

Tabachnick, B. G., and Fidell, L. S. (2019). *Using multivariate statistics* (7th ed.).
Pearson Education.

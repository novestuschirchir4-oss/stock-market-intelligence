# Data Sources & Citations

All datasets used in this project are publicly accessible via raw GitHub URLs.
No API keys, subscriptions, or local file downloads are required.
The project fetches data at runtime directly from the URLs listed below.

---

## Dataset 1 — S&P 500 Constituent Fundamentals

| Field         | Detail |
|---------------|--------|
| Name          | S&P 500 Companies with Financial Information |
| Publisher     | DataHub / datasets organisation |
| URL           | https://raw.githubusercontent.com/datasets/s-and-p-500-companies-financials/master/data/constituents-financials.csv |
| Records       | 503 companies |
| Variables     | Symbol, Name, Sector, Price, P/E Ratio, Dividend Yield, EPS, 52-Week Low, 52-Week High, Market Cap, EBITDA, P/S Ratio, P/B Ratio |
| Reference Period | Snapshot circa 2017-2018 |
| License       | Open Data Commons Public Domain Dedication and Licence (PDDL) |

**Citation (APA 7th):**

DataHub. (2018). *S&P 500 companies with financial information* [Dataset].
GitHub. https://github.com/datasets/s-and-p-500-companies-financials

---

## Dataset 2 — Apple Inc. OHLCV Price Series

| Field         | Detail |
|---------------|--------|
| Name          | Apple (AAPL) Daily OHLCV with Bollinger Bands |
| Publisher     | Plotly Technologies Inc. |
| URL           | https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv |
| Records       | 506 trading days |
| Variables     | Date, Open, High, Low, Close, Volume, Adjusted Close, Bollinger Upper, Bollinger Mid, Bollinger Lower, Trend Direction |
| Reference Period | 2015-02-17 to 2017-02-16 |
| License       | MIT License |

**Citation (APA 7th):**

Plotly Technologies Inc. (2017). *Finance charts — Apple OHLCV* [Dataset].
GitHub. https://github.com/plotly/datasets

---

## Dataset 3 — Multi-Stock Daily Close Prices

| Field         | Detail |
|---------------|--------|
| Name          | Multi-Stock Daily Closing Prices (AAPL, MSFT, IBM, SBUX, S&P 500) |
| Publisher     | Plotly Technologies Inc. |
| URL           | https://raw.githubusercontent.com/plotly/datasets/master/stockdata.csv |
| Records       | 2,306 trading days |
| Variables     | AAPL, MSFT, IBM, SBUX, GSPC (S&P 500 Index), Date |
| Reference Period | 2007-01-03 to 2016-03-01 |
| License       | MIT License |

**Citation (APA 7th):**

Plotly Technologies Inc. (2016). *Stock market data — AAPL, MSFT, IBM, SBUX, S&P 500* [Dataset].
GitHub. https://github.com/plotly/datasets

---

## Methodological References

Graham, B., & Dodd, D. (1934). *Security analysis*. McGraw-Hill.

Mandelbrot, B. (1963). The variation of certain speculative prices.
*The Journal of Business, 36*(4), 394-419. https://doi.org/10.1086/294632

Sharpe, W. F. (1964). Capital asset prices: A theory of market equilibrium under
conditions of risk. *The Journal of Finance, 19*(3), 425-442.
https://doi.org/10.1111/j.1540-6261.1964.tb02865.x

Sharpe, W. F. (1966). Mutual fund performance. *The Journal of Business, 39*(1), 119-138.
https://doi.org/10.1086/294846

Sortino, F. A., & van der Meer, R. (1991). Downside risk. *The Journal of Portfolio
Management, 17*(4), 27-31. https://doi.org/10.3905/jpm.1991.409343

Tabachnick, B. G., & Fidell, L. S. (2019). *Using multivariate statistics* (7th ed.).
Pearson Education.

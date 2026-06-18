"""
src/analytics_helpers.py
------------------------
Standalone helper functions used across the notebook.
Import this module in any cell with:

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from analytics_helpers import compute_rsi, compute_macd, compute_drawdown, style_fig
"""

import numpy as np
import plotly.graph_objects as go


# ── Visual constants (Bloomberg dark terminal) ────────────────────────────────
BG      = "#0D1117"
CARD    = "#161B22"
ACCENT  = "#58A6FF"
GREEN   = "#3FB950"
RED     = "#F85149"
YELLOW  = "#E3B341"
TEXT    = "#E6EDF3"
SUBTEXT = "#8B949E"
BORDER  = "#30363D"
FONT    = "'IBM Plex Mono', 'Courier New', monospace"

SECTOR_COLORS = {
    "Information Technology": "#2196F3",
    "Health Care":             "#4CAF50",
    "Financials":              "#FF9800",
    "Consumer Discretionary":  "#E91E63",
    "Industrials":             "#9C27B0",
    "Communication Services":  "#00BCD4",
    "Consumer Staples":        "#8BC34A",
    "Energy":                  "#FF5722",
    "Utilities":               "#607D8B",
    "Real Estate":             "#795548",
    "Materials":               "#FFC107",
}


def compute_rsi(series, period=14):
    """
    Compute the Relative Strength Index (Wilder, 1978).

    Parameters
    ----------
    series : pd.Series
        Closing price series.
    period : int
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        RSI values in the range [0, 100].
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    """
    Compute MACD line, signal line, and histogram.

    Parameters
    ----------
    series : pd.Series
        Closing price series.
    fast, slow, signal : int
        EMA span parameters (default 12, 26, 9).

    Returns
    -------
    tuple of pd.Series
        (macd_line, signal_line, histogram)
    """
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_drawdown(series):
    """
    Compute percentage drawdown from rolling peak.

    Parameters
    ----------
    series : pd.Series
        Closing price series.

    Returns
    -------
    pd.Series
        Drawdown values (negative, range 0 to -1).
    """
    rolling_max = series.cummax()
    return (series - rolling_max) / rolling_max


def annualised_return(daily_returns, trading_days=252):
    """Mean daily return scaled to annual frequency."""
    return float(daily_returns.mean() * trading_days)


def annualised_volatility(daily_returns, trading_days=252):
    """Daily standard deviation scaled to annual frequency."""
    return float(daily_returns.std() * (trading_days ** 0.5))


def sharpe_ratio(daily_returns, risk_free=0.0, trading_days=252):
    """
    Sharpe ratio (Sharpe, 1966).
    Risk-free rate defaults to 0 for simplicity.
    """
    ann_r = annualised_return(daily_returns, trading_days)
    ann_v = annualised_volatility(daily_returns, trading_days)
    return (ann_r - risk_free) / ann_v if ann_v > 0 else 0.0


def sortino_ratio(daily_returns, risk_free=0.0, trading_days=252):
    """
    Sortino ratio (Sortino and van der Meer, 1991).
    Uses downside deviation (negative returns only) as denominator.
    """
    ann_r    = annualised_return(daily_returns, trading_days)
    neg_ret  = daily_returns[daily_returns < 0]
    downside = neg_ret.std() * (trading_days ** 0.5) if len(neg_ret) > 0 else 0
    return (ann_r - risk_free) / downside if downside > 0 else 0.0


def beta(stock_returns, market_returns):
    """
    CAPM Beta (Sharpe, 1964).
    Cov(stock, market) / Var(market).
    """
    import numpy as np
    cov = np.cov(stock_returns.dropna(), market_returns.dropna())
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 0.0


def var_cvar(daily_returns, confidence=0.95):
    """
    Historical Value at Risk and Conditional VaR.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily return series.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    tuple of float
        (VaR, CVaR) — both negative values representing losses.
    """
    import numpy as np
    cutoff = 1 - confidence
    var    = float(np.percentile(daily_returns.dropna(), cutoff * 100))
    cvar   = float(daily_returns[daily_returns <= var].mean())
    return var, cvar


def style_fig(fig, title, height=500):
    """
    Apply Bloomberg dark-terminal theme to any Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    title : str
        Chart title (bold formatting applied automatically).
    height : int
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Same figure with layout applied (allows method chaining).
    """
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family=FONT, size=14, color=TEXT),
            x=0.01, xanchor="left",
        ),
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(family=FONT, color=TEXT, size=11),
        height=height,
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(22,27,34,0.9)",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            showgrid=True, gridcolor=BORDER,
            zeroline=False, rangeslider=dict(visible=False),
        ),
        yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False),
        margin=dict(l=55, r=20, t=55, b=45),
    )
    return fig

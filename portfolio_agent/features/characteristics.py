"""Accounting characteristics, ranked across the cross-section.

The features `docs/QUANT_RESEARCH.md` §8 and §9 have been waiting for. Each is
peer-relative by construction — book-to-price means nothing about one stock in
isolation, only about where it sits among its peers — which is why none of them
could be written before T24's cross-sectional registry existed.

Every one is a ratio, and the denominator is the design decision
-----------------------------------------------------------------
Novy-Marx's point about gross profitability is the general one: *what you
divide by decides what you have measured.* Gross profit over **assets** ranks
firms by how productively they use capital; over **sales** it ranks them by
margin, which is mostly an industry classification. The two sorts are
different characteristics wearing one name, and this module states which it
computes in every docstring.

Scaling by price versus scaling by assets
-----------------------------------------
Two families, and they behave differently:

- **Price-scaled** (`book_to_price`, `earnings_to_price`) put a market number
  in the denominator, so they move every day and are partly a bet on price
  having fallen. That is the value effect, and it is also why they correlate
  with short-term reversal.
- **Asset-scaled** (`gross_profitability`, `accruals`, `asset_growth`) move
  only when a filing lands. They are quarterly step functions, so their
  turnover is low and their decile is stable — a very different cost profile
  from anything in Phase 3.

The lag is the decorator's, and the store's
-------------------------------------------
`register_cross_sectional_feature` applies the one-bar shift, so the bodies
here never shift. The *fundamental* side is protected by something stronger:
`data_quality/fundamentals.py` only ever emits facts whose report date has
passed, so a value in the panel was publishable on the date it appears. A
one-bar shift would not have saved a fiscal-date-keyed panel, and does not
need to save this one.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .cross_section import CrossSectionPanel, register_cross_sectional_feature

#: Quarters in the asset-growth lookback. Four, so the comparison is
#: year-on-year: a quarter-on-quarter asset growth is dominated by seasonality
#: in working capital and measures the calendar rather than the firm.
ASSET_GROWTH_QUARTERS = 4

#: Trading sessions standing in for those quarters when the panel is daily.
ASSET_GROWTH_SESSIONS = 252


def _safe_ratio(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    """Elementwise ratio with a non-positive denominator treated as unmeasured.

    A firm with zero or negative book equity does not have a very large
    book-to-price — it has an undefined one, and the arithmetic answer is a
    sign artifact that would rank it at one extreme of the cross-section. The
    same holds for negative total assets, which occur in bad data rather than
    in reality.
    """
    safe = denominator.where(denominator > 0)
    return numerator / safe


def market_cap(panel: CrossSectionPanel) -> pd.DataFrame:
    """Shares outstanding times price.

    The real one, as against the traded-value proxy `evaluation/neutralize.py`
    falls back to. A liquidity proxy and a size measure agree on the direction
    and disagree on the ranking, and the size effect is defined on the second.
    """
    return panel.get("close") * panel.get("shares_outstanding")


@register_cross_sectional_feature(
    "book_to_price", inputs=("close", "total_equity", "shares_outstanding")
)
def _book_to_price(panel: CrossSectionPanel) -> pd.DataFrame:
    """Book equity over market capitalization — the value characteristic (HML).

    Inverted from price-to-book on purpose: B/P is well defined as a firm's
    book value shrinks toward zero, where P/B explodes. Fama and French rank on
    B/P for exactly that reason, and a cross-sectional rank on P/B is dominated
    by whichever handful of firms are closest to zero book.
    """
    return _safe_ratio(panel.get("total_equity"), market_cap(panel))


@register_cross_sectional_feature(
    "earnings_to_price", inputs=("close", "net_income", "shares_outstanding")
)
def _earnings_to_price(panel: CrossSectionPanel) -> pd.DataFrame:
    """Net income over market capitalization.

    Loss-making firms produce a negative E/P, and that is kept rather than
    screened: a negative earnings yield is a real and informative position in
    the cross-section, and dropping those names would leave a value sort
    defined only over profitable firms — which is a quality screen wearing a
    value label.
    """
    return _safe_ratio(panel.get("net_income"), market_cap(panel))


@register_cross_sectional_feature(
    "gross_profitability", inputs=("revenue", "cost_of_goods_sold", "total_assets")
)
def _gross_profitability(panel: CrossSectionPanel) -> pd.DataFrame:
    """(Revenue - COGS) / total assets — Novy-Marx's quality characteristic.

    **Gross** profit rather than net, and over **assets** rather than sales.
    Both choices are the substance of the paper. Net income has been through
    depreciation, interest and tax, each of which is partly an accounting and
    financing decision rather than a statement about the business; gross profit
    is the cleanest accounting measure of what the firm actually earns.
    Dividing by assets asks how productively capital is used, where dividing by
    sales asks about margin — and margin is close to an industry label.

    Novy-Marx reports gross profitability having roughly the power of
    book-to-price, and being *negatively* correlated with it: profitable firms
    are expensive ones. That makes the two complements rather than substitutes,
    which is the argument for carrying both.
    """
    gross_profit = panel.get("revenue") - panel.get("cost_of_goods_sold")
    return _safe_ratio(gross_profit, panel.get("total_assets"))


@register_cross_sectional_feature("asset_growth", inputs=("total_assets",))
def _asset_growth(panel: CrossSectionPanel) -> pd.DataFrame:
    """Year-on-year total-asset growth — the investment characteristic (CMA).

    Signed as *growth*, so a conservative firm scores **low**. Fama-French's
    CMA is long conservative and short aggressive, so a long-only book takes
    the bottom of this sort, not the top. Recorded here because the sign is
    the easiest thing to get backwards and the result would look plausible
    either way.

    Year-on-year rather than quarter-on-quarter: the shorter comparison is
    dominated by working-capital seasonality and measures the calendar.
    """
    assets = panel.get("total_assets")
    return assets / assets.shift(ASSET_GROWTH_SESSIONS) - 1.0


@register_cross_sectional_feature(
    "accruals", inputs=("net_income", "cash_flow_operating", "total_assets")
)
def _accruals(panel: CrossSectionPanel) -> pd.DataFrame:
    """(Net income - operating cash flow) / total assets — Sloan's accrual anomaly.

    The gap between earnings a firm *reports* and cash it actually collected.
    Sloan's finding is that the accrual component of earnings is far less
    persistent than the cash component, and that the market prices earnings
    without distinguishing them — so high-accrual firms subsequently
    underperform.

    Signed so that **high means more accrual**, i.e. worse. A long-only book
    takes the bottom of this sort. Same sign trap as `asset_growth`, same
    reason for stating it.
    """
    accrual = panel.get("net_income") - panel.get("cash_flow_operating")
    return _safe_ratio(accrual, panel.get("total_assets"))


@register_cross_sectional_feature("leverage", inputs=("total_debt", "total_equity"))
def _leverage(panel: CrossSectionPanel) -> pd.DataFrame:
    """Total debt over book equity.

    Carried as a *control* rather than as a signal. Leverage mechanically
    raises equity beta, so a characteristic sort that is silently a leverage
    sort will look like it has found something when it has found `bab` with
    extra steps. Having it in the registry means a run can neutralize against
    it instead of wondering.
    """
    return _safe_ratio(panel.get("total_debt"), panel.get("total_equity"))


#: Every characteristic this module registers, for `--neutralize` and for the
#: report to enumerate without importing the module's private names.
CHARACTERISTICS = (
    "book_to_price",
    "earnings_to_price",
    "gross_profitability",
    "asset_growth",
    "accruals",
    "leverage",
)

#: What each one needs, so a caller can tell before running which are
#: computable from the fundamentals file it actually has. A partial dataset is
#: the normal case, not an error.
CHARACTERISTIC_INPUTS = {
    "book_to_price": ("close", "total_equity", "shares_outstanding"),
    "earnings_to_price": ("close", "net_income", "shares_outstanding"),
    "gross_profitability": ("revenue", "cost_of_goods_sold", "total_assets"),
    "asset_growth": ("total_assets",),
    "accruals": ("net_income", "cash_flow_operating", "total_assets"),
    "leverage": ("total_debt", "total_equity"),
}


def computable_characteristics(available_facts: "list[str] | tuple[str, ...]") -> list:
    """Which characteristics a given set of fact columns supports.

    Args:
        available_facts: Column names present in the fundamentals file.
            `close` is assumed available — this platform always has prices.

    Returns:
        The computable subset, in `CHARACTERISTICS` order. Reported rather than
        raised on, because a fundamentals file covering three fields is the
        normal case and should yield three characteristics rather than an error.
    """
    have = set(available_facts) | {"close"}
    return [
        name for name in CHARACTERISTICS
        if set(CHARACTERISTIC_INPUTS[name]).issubset(have)
    ]

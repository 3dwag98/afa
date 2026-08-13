"""Neutralization and decay: does the signal pick stocks, and for how long.

The central test in this file is the pure-sector one. A signal that is nothing
but a sector bet must neutralize to zero IC — if it does not, the
neutralization is not doing what its name claims and every "neutralized" number
the platform ever prints is decoration. It is written twice over: once with a
signal that has no stock-selection content at all, and once with a signal that
has some, because a neutralizer that returns zero for everything would pass the
first test alone.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.config.loader import load_config
from portfolio_agent.evaluation import (
    SIZE_PROXY_NOTE,
    DecayCurve,
    add_exposures,
    decay_curve,
    decay_from_panel,
    evaluate_neutralized,
    neutralize_panel,
    neutralized_ic,
    residualize,
    rolling_beta,
)
from portfolio_agent.evaluation.neutralize import _rows_for
from portfolio_agent.strategies.base import BaseStrategy
from portfolio_agent.strategies.types import StrategyContext, StrategySignal


@pytest.fixture
def app_config():
    return load_config()


N_SECTORS = 5


def sector_panel(idio_weight: float, n_dates: int = 200, n_symbols: int = 60, seed: int = 0):
    """A panel whose returns are a sector move plus an idiosyncratic move.

    `idio_weight` controls how much of the *score* comes from the idiosyncratic
    part: 0.0 is a pure sector bet with no stock-selection content whatever,
    1.0 knows the idiosyncratic move exactly.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_dates, freq="B")
    sectors = {f"S{i:03d}": f"SEC{i % N_SECTORS}" for i in range(n_symbols)}

    rows = []
    for date in dates:
        sector_return = {f"SEC{k}": rng.normal(0.0, 0.02) for k in range(N_SECTORS)}
        for symbol, sector in sectors.items():
            idio = rng.normal(0.0, 0.01)
            rows.append({
                "date": date,
                "symbol": symbol,
                "forward_return": sector_return[sector] + idio,
                "score": sector_return[sector] + idio_weight * idio,
            })

    panel = pd.DataFrame(rows)
    dummies = pd.get_dummies(panel["symbol"].map(sectors), prefix="sector", dtype=float)
    for column in dummies.columns:
        panel[column] = dummies[column].to_numpy()
    return panel, list(dummies.columns), sectors


# --------------------------------------------------------------------------
# The acceptance test
# --------------------------------------------------------------------------


def test_a_pure_sector_signal_neutralizes_to_zero_ic():
    """The test that proves neutralization does what it claims.

    The score here is the sector's own return, identical for every name in a
    sector. It scores a very high raw IC and makes no stock-selection claim
    whatsoever, so the residual must carry no rankable signal at all.
    """
    panel, exposures, _ = sector_panel(idio_weight=0.0)
    result = neutralized_ic(panel, exposures, horizon=5)

    assert result.raw.mean > 0.5
    assert result.raw.significant
    assert result.neutralized.mean == pytest.approx(0.0, abs=1e-9)
    assert not result.neutralized.significant
    assert result.explained == pytest.approx(1.0)


def test_a_signal_with_real_selection_keeps_most_of_it():
    """Guards the other side: a neutralizer that zeroes everything is useless."""
    panel, exposures, _ = sector_panel(idio_weight=1.0)
    result = neutralized_ic(panel, exposures, horizon=5)

    assert result.neutralized.mean > 0.3
    assert result.neutralized.significant
    assert 0.0 < result.explained < 1.0


def test_raw_and_neutralized_are_reported_together():
    """Replacing one with the other throws away the comparison that informs."""
    panel, exposures, _ = sector_panel(idio_weight=0.5)
    result = neutralized_ic(panel, exposures, horizon=5)

    document = result.to_dict()
    assert "raw_mean_ic" in document and "neutralized_mean_ic" in document
    text = result.render()
    assert "raw" in text and "neutralized" in text
    assert "stock selection" in text


def test_the_two_ic_numbers_are_computed_over_the_same_dates():
    """A neutralized IC over fewer dates is not a comparison.

    Half the dates here are too thin to regress. Whatever is dropped from the
    neutralized side has to be dropped from the raw side too, or the difference
    mixes the exposures' effect with a change of sample.
    """
    panel, exposures, _ = sector_panel(idio_weight=0.5, n_dates=40, n_symbols=30)
    thin_dates = sorted(panel["date"].unique())[:20]
    thin = panel[panel["date"].isin(thin_dates)].groupby("date").head(6)
    mixed = pd.concat([thin, panel[~panel["date"].isin(thin_dates)]], ignore_index=True)

    result = neutralized_ic(mixed, exposures, horizon=5)
    assert result.n_dates_skipped == 20
    assert result.raw.n_dates == result.neutralized.n_dates


# --------------------------------------------------------------------------
# The regression itself
# --------------------------------------------------------------------------


def test_residualize_removes_an_exact_linear_exposure():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    exposures = values.reshape(-1, 1)
    assert residualize(values, exposures) == pytest.approx(np.zeros(5))


def test_the_residual_is_orthogonal_to_every_exposure():
    """The defining property, and the one neutralization's claim rests on.

    If the residual still correlates with an exposure, any IC it retains could
    still be coming from that exposure — which is exactly what neutralizing was
    supposed to rule out.
    """
    rng = np.random.default_rng(4)
    exposures = rng.normal(size=(60, 3))
    values = exposures @ np.array([2.0, -1.0, 0.5]) + rng.normal(size=60) + 7.0

    residual = residualize(values, exposures)
    assert residual.mean() == pytest.approx(0.0, abs=1e-9)
    for column in range(exposures.shape[1]):
        assert np.dot(residual, exposures[:, column]) == pytest.approx(0.0, abs=1e-8)


def test_residualizing_twice_changes_nothing():
    """Projection is idempotent; a second pass finding more to remove is a bug."""
    rng = np.random.default_rng(5)
    exposures = rng.normal(size=(40, 2))
    values = exposures @ np.array([1.0, -2.0]) + rng.normal(size=40)

    once = residualize(values, exposures)
    twice = residualize(once, exposures)
    assert twice == pytest.approx(once, abs=1e-9)


def test_residualize_keeps_the_part_no_exposure_explains():
    """A component orthogonal to the exposures must survive untouched."""
    exposure = np.arange(20.0).reshape(-1, 1)
    independent = residualize(
        np.array([1.0, -1.0] * 10) + np.arange(20.0) * 0.01, exposure
    )
    residual = residualize(exposure.ravel() + independent, exposure)
    assert residual == pytest.approx(independent, abs=1e-9)


def test_residualize_always_removes_the_intercept():
    """Otherwise a drifting level looks like a loading on any non-zero exposure."""
    exposure = np.array([[0.0], [0.0], [0.0], [0.0]])
    residual = residualize(np.array([5.0, 5.0, 5.0, 5.0]), exposure)
    assert residual == pytest.approx(np.zeros(4))


def test_residualize_handles_rank_deficient_dummies():
    """Sector dummies sum to the intercept; a normal-equation solve would fail."""
    dummies = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    residual = residualize(np.array([1.0, 1.0, 3.0, 3.0]), dummies)
    assert residual == pytest.approx(np.zeros(4), abs=1e-9)


def test_a_numerically_exhausted_residual_snaps_to_zero():
    """Float noise from an exact fit is not random — it is a scaled copy.

    Every name in a sector runs identical arithmetic, so all of them get the
    same 1e-17 residual and the vector is still a perfect sector ordering three
    hundred orders of magnitude below anything meaningful. Spearman does not
    care about scale, so without this the "neutralized" IC of a pure sector
    signal comes back at 0.18 instead of 0.
    """
    dummies = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    residual = residualize(np.array([1.0, 1.0, 3.0, 3.0]), dummies)
    assert np.all(residual == 0.0)   # exactly zero, not merely small


def test_a_real_residual_is_not_snapped_away():
    exposure = np.arange(20.0).reshape(-1, 1)
    values = exposure.ravel() + np.array([0.001, -0.001] * 10)
    residual = residualize(values, exposure)
    assert np.abs(residual).max() > 1e-5


def test_dates_too_thin_to_regress_are_skipped_not_fitted():
    """Fitting k exposures across six names produces a residual made of noise."""
    panel, exposures, _ = sector_panel(idio_weight=0.5, n_dates=10, n_symbols=6)
    _, used, skipped = neutralize_panel(panel, exposures, min_names=5)
    assert used == 0
    assert skipped == 10


def test_a_missing_exposure_column_raises():
    panel, _, _ = sector_panel(idio_weight=0.5, n_dates=10)
    with pytest.raises(ValueError, match="exposure column"):
        neutralize_panel(panel, ["not_a_column"])


# --------------------------------------------------------------------------
# Exposure construction
# --------------------------------------------------------------------------


def price_panel(n_dates: int = 400, n_symbols: int = 20, seed: int = 3) -> pd.DataFrame:
    """A panel with prices and volume, as `keep_prices=True` produces."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_dates, freq="B")
    rows = []
    market = rng.normal(0.0003, 0.011, n_dates)
    for i in range(n_symbols):
        beta = 0.5 + i / n_symbols
        returns = beta * market + rng.normal(0.0, 0.008, n_dates)
        close = 100.0 * np.exp(np.cumsum(returns))
        volume = rng.integers(1e5, 1e6, n_dates).astype(float)
        for j, date in enumerate(dates):
            rows.append({
                "date": date, "symbol": f"S{i:02d}", "score": float(rng.normal()),
                "forward_return": float(returns[j]),
                "close": float(close[j]), "volume": float(volume[j]),
            })
    return pd.DataFrame(rows)


def test_rolling_beta_recovers_a_known_beta():
    """A synthetic name with beta 2 should measure near 2, not near 1."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-03", periods=600, freq="B")
    market = rng.normal(0.0, 0.01, 600)
    returns = pd.DataFrame(
        {
            "LOW": 0.5 * market + rng.normal(0.0, 0.001, 600),
            "HIGH": 2.0 * market + rng.normal(0.0, 0.001, 600),
            "MID": 1.0 * market + rng.normal(0.0, 0.001, 600),
        },
        index=dates,
    )
    betas = rolling_beta(returns, window=252)
    final = betas.iloc[-1] * 3.0 / (0.5 + 2.0 + 1.0)   # composite is their mean
    assert final["HIGH"] > final["MID"] > final["LOW"]
    assert betas["HIGH"].iloc[-1] / betas["LOW"].iloc[-1] == pytest.approx(4.0, rel=0.15)


def test_rolling_beta_is_causal():
    """The beta used on date t has to have been estimable on date t."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("2022-01-03", periods=500, freq="B")
    returns = pd.DataFrame(
        {f"S{i}": rng.normal(0.0, 0.01, 500) for i in range(5)}, index=dates
    )
    full = rolling_beta(returns, window=126)
    truncated = rolling_beta(returns.iloc[:400], window=126)
    common = full.index.intersection(truncated.index)
    pd.testing.assert_frame_equal(
        full.loc[common], truncated.loc[common], check_exact=False
    )


def test_exposures_carry_beta_and_the_size_proxy():
    panel, columns, notes = add_exposures(price_panel())
    assert set(columns) == {"beta", "size"}
    assert panel["beta"].notna().any()
    assert panel["size"].notna().any()


def test_the_size_proxy_substitution_is_stated_in_the_output():
    """An acceptance criterion, and the right one: in the output, not the code."""
    panel, columns, notes = add_exposures(price_panel())
    assert any(SIZE_PROXY_NOTE == note for note in notes)
    assert any("free float" in note for note in notes)

    result = neutralized_ic(panel, columns, horizon=5, notes=notes)
    assert "free float" in result.render()
    assert any("proxy" in note for note in result.to_dict()["notes"])


def test_a_missing_sector_map_is_reported_not_passed_over():
    """"Neutralized" quietly meaning "not sector-neutralized" is the failure mode."""
    _, _, notes = add_exposures(price_panel())
    assert any("NOT sector-neutral" in note for note in notes)


def test_a_supplied_sector_map_becomes_dummy_exposures():
    panel = price_panel(n_symbols=12)
    sectors = {f"S{i:02d}": f"SEC{i % 3}" for i in range(12)}
    panel, columns, _ = add_exposures(panel, sector_map=sectors)
    assert {"sector_SEC0", "sector_SEC1", "sector_SEC2"} <= set(columns)


def test_a_panel_without_prices_says_so_rather_than_silently_skipping():
    panel, _, _ = sector_panel(idio_weight=0.5, n_dates=20)
    _, columns, notes = add_exposures(panel)
    assert columns == []
    assert any("keep_prices" in note for note in notes)


@pytest.mark.parametrize(
    "sessions,stride,expected", [(252, 1, 252), (252, 5, 50), (60, 5, 20), (252, 100, 20)]
)
def test_rolling_windows_are_converted_from_sessions_to_panel_rows(
    sessions, stride, expected
):
    """A 252-session beta on a stride of 5 is 50 rows, not 252.

    Left in rows it silently becomes a five-year window — and then never fills,
    because a strided panel does not have 252 rows to fill it with.
    """
    assert _rows_for(sessions, stride) == expected


def test_a_strided_panel_reports_the_converted_window():
    panel = price_panel(n_dates=200)
    strided = panel[panel["date"].isin(sorted(panel["date"].unique())[::5])]
    _, _, notes = add_exposures(strided, stride=5)
    assert any("stride=5" in note and "panel rows" in note for note in notes)


# --------------------------------------------------------------------------
# Decay
# --------------------------------------------------------------------------


def decay_panel(profile: Dict[int, float], n_dates: int = 150, n_symbols: int = 40, seed: int = 5):
    """A panel whose IC at each horizon is dialled in by `profile`."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_dates, freq="B")
    horizons = sorted(profile)

    rows = []
    for date in dates:
        score = rng.normal(0.0, 1.0, n_symbols)
        row_returns = {}
        for horizon in horizons:
            weight = profile[horizon]
            row_returns[horizon] = weight * score + (1 - abs(weight)) * rng.normal(
                0.0, 1.0, n_symbols
            )
        for i in range(n_symbols):
            row = {"date": date, "symbol": f"S{i:02d}", "score": float(score[i])}
            for horizon in horizons:
                row[f"forward_return_{horizon}"] = float(row_returns[horizon][i])
            rows.append(row)
    return pd.DataFrame(rows), horizons


def test_a_decay_curve_recovers_a_dialled_in_profile():
    panel, horizons = decay_panel({1: 0.9, 5: 0.4, 21: 0.02})
    curve = decay_from_panel(panel, horizons, strategy="synthetic")

    values = [point.ic.mean for point in curve.points]
    assert values[0] > values[1] > values[2]
    assert curve.peak_horizon() == 1


def test_a_front_loaded_curve_is_described_as_front_loaded():
    """The reading that says "this needs daily turnover, and may be noise"."""
    panel, horizons = decay_panel({1: 0.9, 5: 0.1, 21: 0.01})
    curve = decay_from_panel(panel, horizons, strategy="fast")
    assert "front-loaded" in curve.shape()


def test_a_flat_curve_is_described_as_slow():
    panel, horizons = decay_panel({1: 0.4, 5: 0.4, 21: 0.4})
    curve = decay_from_panel(panel, horizons, strategy="slow")
    assert "slow" in curve.shape()
    assert curve.half_life() is None


def test_half_life_is_interpolated_between_the_bracketing_horizons():
    panel, horizons = decay_panel({1: 0.8, 5: 0.4, 21: 0.05})
    curve = decay_from_panel(panel, horizons, strategy="s")
    half_life = curve.half_life()
    assert half_life is not None
    assert 1.0 < half_life <= 21.0


def test_a_curve_with_no_positive_ic_says_so():
    panel, horizons = decay_panel({1: -0.5, 5: -0.5})
    curve = decay_from_panel(panel, horizons, strategy="s")
    assert "no positive IC" in curve.shape()
    assert curve.half_life() is None


def test_every_horizon_is_scored_on_the_dates_it_can_support():
    """Restricting to the longest horizon's reach would shorten the whole curve."""
    panel, horizons = decay_panel({1: 0.5, 21: 0.5}, n_dates=60)
    last_dates = sorted(panel["date"].unique())[-10:]
    panel.loc[panel["date"].isin(last_dates), "forward_return_21"] = np.nan

    curve = decay_from_panel(panel, horizons, strategy="s")
    by_horizon = {p.horizon: p for p in curve.points}
    assert by_horizon[1].n_observations > by_horizon[21].n_observations


def test_a_missing_horizon_column_raises_rather_than_leaving_a_hole():
    """A gap in the curve reads as a decay, which is the wrong conclusion."""
    panel, horizons = decay_panel({1: 0.5, 5: 0.5})
    with pytest.raises(ValueError, match="forward_return_10"):
        decay_from_panel(panel, [1, 5, 10], strategy="s")


def test_the_decay_curve_renders_and_serializes():
    panel, horizons = decay_panel({1: 0.6, 5: 0.3, 21: 0.05})
    curve = decay_from_panel(panel, horizons, strategy="s")

    text = curve.render()
    assert "Signal decay" in text
    assert "horizon" in text

    document = curve.to_dict()
    assert document["peak_horizon"] == 1
    assert len(document["points"]) == 3
    assert len(curve.to_frame()) == 3


# --------------------------------------------------------------------------
# End to end against registered strategies
# --------------------------------------------------------------------------


def _ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.02, "low": close * 0.98,
            "close": close, "volume": rng.integers(1e5, 1e6, n).astype(float),
        },
        index=index,
    )


@pytest.fixture
def fake_cache(monkeypatch):
    frames = {f"T{i}": _ohlcv(seed=i) for i in range(10)}
    monkeypatch.setattr(
        "portfolio_agent.src.data_store.load_ticker_data",
        lambda ticker, start_date=None, end_date=None: frames.get(ticker),
        raising=True,
    )
    return frames


@pytest.mark.parametrize("strategy", ["momentum", "low_volatility"])
def test_a_decay_curve_is_produced_for_every_registered_strategy(
    app_config, fake_cache, strategy
):
    curve = decay_curve(
        app_config, strategy, universe=list(fake_cache), horizons=(1, 5, 21),
        stride=40, min_history=260, max_dates=4, use_benchmark=False,
    )
    assert isinstance(curve, DecayCurve)
    assert [point.horizon for point in curve.points] == [1, 5, 21]
    assert all(point.n_observations > 0 for point in curve.points)
    assert strategy in curve.render()


@pytest.mark.parametrize("strategy", ["momentum", "low_volatility"])
def test_raw_and_neutralized_ic_come_together_for_every_strategy(
    app_config, fake_cache, strategy
):
    result = evaluate_neutralized(
        app_config, strategy, universe=list(fake_cache),
        horizon=5, stride=20, min_history=260, max_dates=12, use_benchmark=False,
    )
    text = result.render()
    assert "raw" in text and "neutralized" in text
    # The gap is only meaningful against the exposures that were actually used.
    assert "Neutralized against" in text
    assert any("sector" in note for note in result.notes)


def test_the_decay_curve_scores_once_for_every_horizon(app_config, fake_cache):
    """Six horizons must cost one scoring pass, on identical scores.

    The speed is a bonus. The reason it matters is that two horizons scored in
    separate runs could differ by anything that moved between them.
    """
    calls = {"n": 0}
    from portfolio_agent.strategies.cross_sectional import MomentumStrategy

    original = MomentumStrategy.score_batch

    def counting(self, features_by_symbol, context):
        calls["n"] += 1
        return original(self, features_by_symbol, context)

    MomentumStrategy.score_batch = counting
    try:
        curve = decay_curve(
            app_config, "momentum", universe=list(fake_cache),
            horizons=(1, 2, 3, 5, 10, 21), stride=40, min_history=260,
            max_dates=3, use_benchmark=False,
        )
    finally:
        MomentumStrategy.score_batch = original

    assert len(curve.points) == 6
    # One call per scored date, not one per date per horizon.
    assert calls["n"] == 3

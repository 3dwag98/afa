"""Tests for cross-sectional momentum and low-volatility strategies."""

import pandas as pd
import pytest
import yaml

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.strategies.cross_sectional import MomentumStrategy, LowVolatilityStrategy
from portfolio_agent.strategies.ensemble import EnsembleStrategy
from portfolio_agent.strategies.types import RiskParams, StrategyContext


def _risk_params() -> RiskParams:
    return RiskParams(
        target_prob_profit=0.55,
        min_reward_risk=1.5,
        min_price_inr=20.0,
        portfolio_value_inr=1_000_000.0,
        risk_per_trade_pct=0.01,
        max_single_position_pct=0.10,
        atr_stop_multiplier=1.5,
        atr_target_multiplier=2.0,
    )


def _features(close: float, atr: float = 2.0, **extra) -> pd.DataFrame:
    row = {"close": close, "atr_14": atr, **extra}
    return pd.DataFrame([row])


def _momentum_config(**params) -> StrategyConfig:
    return StrategyConfig(type="momentum", params=params)


def _low_vol_config(**params) -> StrategyConfig:
    return StrategyConfig(type="low_volatility", params=params)


class TestMomentumStrategy:
    def test_defaults(self):
        strategy = MomentumStrategy(_momentum_config())
        assert strategy.name == "momentum"
        assert strategy.requires_full_batch is True
        # realized_vol_60 drives the per-position volatility-targeting scalar
        # rather than the ranking; the rest back the tradability screen.
        assert strategy.required_features() == [
            "close", "mom_9m_skip1m", "atr_14", "realized_vol_60",
            "traded_value_60", "zero_return_fraction_60",
            "circuit_lock_fraction_60", "circuit_locked_today",
            "operator_trap_fraction_60", "operator_trap_today",
        ]

    def test_liquidity_filter_features_drop_out_when_disabled(self):
        strategy = MomentumStrategy(_momentum_config(liquidity_filter=False))
        assert strategy.required_features() == [
            "close", "mom_9m_skip1m", "atr_14", "realized_vol_60"
        ]

    def test_min_universe_default_is_statistically_meaningful(self):
        """A 'top decile' of 5 names is one stock, which is not a ranking."""
        strategy = MomentumStrategy(_momentum_config())
        assert strategy.entry_rules()["min_universe"] >= 30

    def test_default_universe_of_ten_is_rejected(self):
        strategy = MomentumStrategy(_momentum_config())
        features_by_symbol = {
            f"SYM{i}": _features(close=100.0, mom_9m_skip1m=i / 100.0) for i in range(1, 11)
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert all(s.signal == "AVOID" for s in signals.values())
        assert "too small" in signals["SYM10"].rationale

    def test_custom_name_and_percentile(self):
        strategy = MomentumStrategy(_momentum_config(name="my_momentum", top_percentile=0.2))
        assert strategy.name == "my_momentum"
        assert strategy.entry_rules()["top_percentile"] == 0.2

    def test_top_decile_gets_buy_rest_avoid(self):
        # 10 symbols, momentum values 0.01..0.10; top_percentile=0.1 -> only the highest qualifies.
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.1, min_universe=5))
        features_by_symbol = {
            f"SYM{i}": _features(close=100.0, mom_9m_skip1m=i / 100.0) for i in range(1, 11)
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert signals["SYM10"].signal == "BUY"
        assert signals["SYM10"].trigger == "Momentum"
        for i in range(1, 10):
            assert signals[f"SYM{i}"].signal == "AVOID"
            assert signals[f"SYM{i}"].trigger == "None"

    def test_highest_momentum_scores_highest(self):
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.5, min_universe=5))
        features_by_symbol = {
            f"SYM{i}": _features(close=100.0, mom_9m_skip1m=i / 100.0) for i in range(1, 6)
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert signals["SYM5"].score > signals["SYM1"].score

    def test_small_universe_falls_back_to_avoid(self):
        strategy = MomentumStrategy(_momentum_config(min_universe=5))
        features_by_symbol = {
            "SYM1": _features(close=100.0, mom_9m_skip1m=0.05),
            "SYM2": _features(close=100.0, mom_9m_skip1m=0.10),
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert all(s.signal == "AVOID" for s in signals.values())
        assert "too small" in signals["SYM1"].rationale

    def test_below_min_price_is_watch_not_buy(self):
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.2, min_universe=5))
        features_by_symbol = {f"SYM{i}": _features(close=100.0, mom_9m_skip1m=i / 100.0) for i in range(1, 6)}
        features_by_symbol["SYM5"] = _features(close=5.0, mom_9m_skip1m=0.05)  # below min_price_inr=20
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert signals["SYM5"].signal == "WATCH"

    def test_score_single_ticker_delegates_to_score_batch(self):
        strategy = MomentumStrategy(_momentum_config(min_universe=1))
        features = _features(close=100.0, mom_9m_skip1m=0.05)
        context = StrategyContext(risk=_risk_params())

        sig = strategy.score("SOLO", features, context)

        assert sig.symbol == "SOLO"


class TestLowVolatilityStrategy:
    def test_defaults(self):
        strategy = LowVolatilityStrategy(_low_vol_config())
        assert strategy.name == "low_volatility"
        assert strategy.requires_full_batch is True
        assert strategy.required_features() == [
            "close", "realized_vol_60", "atr_14",
            "traded_value_60", "zero_return_fraction_60",
            "circuit_lock_fraction_60", "circuit_locked_today",
            "operator_trap_fraction_60", "operator_trap_today",
        ]
        assert strategy.entry_rules()["min_universe"] >= 30

    def test_regime_filter_off_by_default(self):
        """Low-volatility is the defensive sleeve — it is meant to keep working
        through the drawdowns that stand momentum down."""
        strategy = LowVolatilityStrategy(_low_vol_config())
        assert strategy.entry_rules()["crash_protection"]["regime_filter"] is False

    def test_lowest_volatility_gets_buy(self):
        strategy = LowVolatilityStrategy(_low_vol_config(top_percentile=0.1, min_universe=5))
        # Lower realized_vol_60 is better; SYM1 has the lowest.
        features_by_symbol = {
            f"SYM{i}": _features(close=100.0, realized_vol_60=i * 0.05) for i in range(1, 11)
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert signals["SYM1"].signal == "BUY"
        assert signals["SYM1"].trigger == "LowVolatility"
        assert signals["SYM10"].signal == "AVOID"

    def test_small_universe_falls_back_to_avoid(self):
        strategy = LowVolatilityStrategy(_low_vol_config(min_universe=5))
        features_by_symbol = {
            "SYM1": _features(close=100.0, realized_vol_60=0.10),
            "SYM2": _features(close=100.0, realized_vol_60=0.20),
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert all(s.signal == "AVOID" for s in signals.values())


class TestEnsembleRejectsFullBatchMembers:
    def test_ensemble_raises_for_cross_sectional_member(self, tmp_path):
        uma_path = tmp_path / "bad_uma.yaml"
        uma_path.write_text(yaml.safe_dump({
            "name": "bad_uma",
            "method": "weighted_blend",
            "members": [
                {"type": "momentum", "weight": 1.0, "params": {}},
            ],
        }))
        config = StrategyConfig(type="ensemble", config_path=str(uma_path))

        with pytest.raises(ValueError, match="requires the full eligible universe"):
            EnsembleStrategy(config)


def _universe(n_symbols=40, n=400, start=100, end=200, noise=0.004, atr=2.0, vol=0.20, seed=1):
    """A universe with real price history, so the regime filter has something
    to assess (a one-row fixture leaves it fail-neutral).

    Every symbol shares one market shock series plus a small idiosyncratic
    wobble. That common factor is the point: independent per-symbol noise
    averages out of the equal-weighted composite, so a universe of 40
    independently-noisy stocks looks *calm* at the market level no matter how
    wild each one is — which is exactly the diversification a market-wide
    crash does not offer.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    ramp = np.linspace(start, end, n)
    market_shock = rng.normal(0.0, noise, n)

    universe = {}
    for i in range(1, n_symbols + 1):
        idio = rng.normal(0.0, noise * 0.1, n)
        universe[f"SYM{i:02d}"] = pd.DataFrame({
            "close": ramp * (1.0 + market_shock + idio),
            "atr_14": [atr] * n,
            "mom_9m_skip1m": [i / 100.0] * n,
            "realized_vol_60": [vol] * n,
        })
    return universe


class TestMomentumCrashProtection:
    def test_stands_down_in_the_crash_regime(self):
        """Falling market plus a volatility spike is the state momentum
        crashes in — the top decile becomes WATCH, not BUY."""
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.1))
        crash = _universe(n_symbols=40, n=400, start=200, end=100, noise=0.05)
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(crash, context)

        assert not any(s.signal == "BUY" for s in signals.values())
        top = signals["SYM40"]
        assert top.signal == "WATCH"
        assert top.extra["regime"] == "crash_risk"
        assert top.extra["position_scale"] == 0.0
        assert "crash risk" in top.rationale

    def test_buys_the_top_decile_in_a_calm_uptrend(self):
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.1))
        calm = _universe(n_symbols=40, n=400, start=100, end=200, noise=0.004)
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(calm, context)

        assert signals["SYM40"].signal == "BUY"
        assert signals["SYM40"].extra["regime"] == "risk_on"
        assert signals["SYM40"].extra["position_scale"] > 0

    def test_regime_filter_can_be_disabled(self):
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.1, regime_filter=False))
        crash = _universe(n_symbols=40, n=400, start=200, end=100, noise=0.05)
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(crash, context)

        assert signals["SYM40"].signal == "BUY"
        assert signals["SYM40"].extra["regime"] == "unknown"

    def test_bear_exposure_dampens_instead_of_standing_down(self):
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.1, bear_exposure=0.3))
        crash = _universe(n_symbols=40, n=400, start=200, end=100, noise=0.05)
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(crash, context)

        top = signals["SYM40"]
        assert top.signal == "BUY"
        assert 0.0 < top.extra["position_scale"] <= 0.3

    def test_high_volatility_names_get_smaller_positions(self):
        """Constant volatility scaling: twice the risk budget, half the money."""
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.5, volatility_target=0.20))
        universe = _universe(n_symbols=40, n=400, start=100, end=200, noise=0.004)
        universe["SYM40"]["realized_vol_60"] = 0.40  # 2x the 20% target
        universe["SYM39"]["realized_vol_60"] = 0.10  # calm; scalar clamps at 1.0
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(universe, context)

        assert signals["SYM40"].extra["volatility_scalar"] == pytest.approx(0.5)
        assert signals["SYM39"].extra["volatility_scalar"] == pytest.approx(1.0)
        assert signals["SYM40"].extra["position_scale"] < signals["SYM39"].extra["position_scale"]

    def test_volatility_scaling_can_be_disabled(self):
        strategy = MomentumStrategy(
            _momentum_config(top_percentile=0.5, scale_by_volatility=False)
        )
        universe = _universe(n_symbols=40, n=400, start=100, end=200, noise=0.004)
        universe["SYM40"]["realized_vol_60"] = 0.80
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(universe, context)

        assert signals["SYM40"].extra["volatility_scalar"] == pytest.approx(1.0)

    def test_single_row_fixtures_stay_fail_neutral(self):
        """Too little history to judge the regime must not silently disable the
        strategy — there is no evidence of a panic state to act on."""
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.1, min_universe=5))
        features_by_symbol = {
            f"SYM{i}": _features(close=100.0, mom_9m_skip1m=i / 100.0) for i in range(1, 11)
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert signals["SYM10"].signal == "BUY"
        assert signals["SYM10"].extra["regime"] == "unknown"


class TestCostAwareSignals:
    def test_reward_risk_is_net_of_round_trip_costs(self):
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.1, min_universe=5))
        features_by_symbol = {
            f"SYM{i}": _features(close=100.0, atr=2.0, mom_9m_skip1m=i / 100.0)
            for i in range(1, 11)
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        # Gross would be 2.0 ATR target over 1.5 ATR stop = 1.333.
        assert 0 < signals["SYM10"].reward_risk < 1.333

    def test_trade_that_cannot_pay_for_itself_is_not_a_buy(self):
        """When friction exceeds the whole target move, the trade is a WATCH
        regardless of how well it ranks."""
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.1, min_universe=5))
        risk = _risk_params()
        risk.buy_cost_pct = 0.20
        risk.sell_cost_pct = 0.20
        features_by_symbol = {
            f"SYM{i}": _features(close=100.0, atr=2.0, mom_9m_skip1m=i / 100.0)
            for i in range(1, 11)
        }

        signals = strategy.score_batch(features_by_symbol, StrategyContext(risk=risk))

        assert signals["SYM10"].signal == "WATCH"
        assert signals["SYM10"].reward_risk == 0.0
        assert "net_rr(0.00)>0:FAIL" in signals["SYM10"].rationale


class TestTradabilityScreen:
    """A printed close is only a price you could trade at if the stock was
    actually trading. Screened names leave the ranking entirely."""

    @staticmethod
    def _liquid(**overrides):
        row = {
            "traded_value_60": 50_000_000.0,
            "zero_return_fraction_60": 0.0,
            "circuit_lock_fraction_60": 0.0,
            "circuit_locked_today": 0.0,
            "operator_trap_fraction_60": 0.0,
            "operator_trap_today": 0.0,
        }
        row.update(overrides)
        return row

    def _universe_with(self, bad_symbol_overrides, n_symbols=10):
        universe = {}
        for i in range(1, n_symbols + 1):
            symbol = f"SYM{i}"
            extra = self._liquid(**bad_symbol_overrides) if symbol in ("SYM10",) else self._liquid()
            universe[symbol] = _features(
                close=100.0, mom_9m_skip1m=i / 100.0, realized_vol_60=0.20, **extra
            )
        return universe

    def test_circuit_locked_leader_is_dropped_from_the_ranking(self):
        """The classic operator pump: the top-ranked name is locked at its
        upper circuit, so it is not merely un-buyable — it must not shift
        everyone else's percentile either."""
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.2, min_universe=5))
        universe = self._universe_with({"circuit_locked_today": 1.0})
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(universe, context)

        assert signals["SYM10"].signal == "AVOID"
        assert "no fill available" in signals["SYM10"].rationale
        assert signals["SYM10"].extra["position_scale"] == 0.0
        # SYM9 is now the top of a 9-name cross-section, and scores 100.
        assert signals["SYM9"].signal == "BUY"
        assert signals["SYM9"].score == 100.0

    def test_serially_circuit_locked_names_are_dropped(self):
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.2, min_universe=5))
        universe = self._universe_with({"circuit_lock_fraction_60": 0.5})
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(universe, context)

        assert signals["SYM10"].signal == "AVOID"
        assert "circuit-driven" in signals["SYM10"].rationale

    def test_zombie_stock_never_wins_the_low_volatility_decile(self):
        """The illiquidity illusion: the lowest-variance name is a stock that
        barely trades, which is exactly what the metric would reward."""
        strategy = LowVolatilityStrategy(_low_vol_config(top_percentile=0.2, min_universe=5))
        universe = {
            f"SYM{i}": _features(
                close=100.0, realized_vol_60=i * 0.05, **self._liquid()
            )
            for i in range(1, 11)
        }
        # SYM1 has the lowest variance because nothing trades.
        universe["SYM1"] = _features(
            close=100.0, realized_vol_60=0.01,
            **self._liquid(zero_return_fraction_60=0.8, traded_value_60=10_000.0),
        )
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(universe, context)

        assert signals["SYM1"].signal == "AVOID"
        assert "illiquid" in signals["SYM1"].rationale
        assert signals["SYM2"].signal == "BUY"

    def test_filter_can_be_disabled(self):
        strategy = MomentumStrategy(
            _momentum_config(top_percentile=0.2, min_universe=5, liquidity_filter=False)
        )
        universe = self._universe_with({"circuit_locked_today": 1.0})
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(universe, context)

        assert signals["SYM10"].signal == "BUY"

    def test_missing_screening_data_passes_rather_than_rejects(self):
        """Short caches must not disqualify every name — the screen excludes
        on positive evidence of untradeability, not on absence of evidence."""
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.2, min_universe=5))
        universe = {
            f"SYM{i}": _features(close=100.0, mom_9m_skip1m=i / 100.0)
            for i in range(1, 11)
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(universe, context)

        assert signals["SYM10"].signal == "BUY"

    def test_thresholds_are_configurable(self):
        strategy = MomentumStrategy(
            _momentum_config(top_percentile=0.2, min_universe=5, min_traded_value_inr=0.0)
        )
        universe = self._universe_with({"traded_value_60": 1_000.0})
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(universe, context)

        assert signals["SYM10"].signal == "BUY"


class TestBenchmarkDrivenRegime:
    """Given a real index (the Nifty, cached from the dataset's indices/), the
    crash filter keys off it instead of a composite of today's universe."""

    @staticmethod
    def _index_series(start, end, n=400, noise=0.004, seed=2):
        import numpy as np

        rng = np.random.default_rng(seed)
        ramp = np.linspace(start, end, n)
        return pd.Series(
            ramp * (1.0 + rng.normal(0.0, noise, n)),
            index=pd.bdate_range("2020-01-01", periods=n),
        )

    def test_falling_benchmark_stands_momentum_down_despite_a_rising_universe(self):
        """The universe here is climbing; only the index says the market is in
        a volatile downtrend. A composite would miss that entirely."""
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.1))
        rising_universe = _universe(n_symbols=40, n=400, start=100, end=200, noise=0.004)
        crashing_index = self._index_series(200, 100, noise=0.05)
        context = StrategyContext(risk=_risk_params(), benchmark_close=crashing_index)

        signals = strategy.score_batch(rising_universe, context)

        assert signals["SYM40"].extra["regime"] == "crash_risk"
        assert signals["SYM40"].signal == "WATCH"

    def test_rising_benchmark_keeps_momentum_invested(self):
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.1))
        universe = _universe(n_symbols=40, n=400, start=100, end=200, noise=0.004)
        context = StrategyContext(
            risk=_risk_params(), benchmark_close=self._index_series(100, 200)
        )

        signals = strategy.score_batch(universe, context)

        assert signals["SYM40"].extra["regime"] == "risk_on"
        assert signals["SYM40"].signal == "BUY"

    def test_short_benchmark_history_falls_back_to_the_composite(self):
        """A benchmark too short for the 200-day trend test is not usable, and
        must not disable the filter — the composite still is."""
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.1))
        crashing_universe = _universe(n_symbols=40, n=400, start=200, end=100, noise=0.05)
        context = StrategyContext(
            risk=_risk_params(),
            benchmark_close=self._index_series(100, 200, n=50),
        )

        signals = strategy.score_batch(crashing_universe, context)

        assert signals["SYM40"].extra["regime"] == "crash_risk"

    def test_no_benchmark_uses_the_composite(self):
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.1))
        crashing_universe = _universe(n_symbols=40, n=400, start=200, end=100, noise=0.05)

        signals = strategy.score_batch(crashing_universe, StrategyContext(risk=_risk_params()))

        assert signals["SYM40"].extra["regime"] == "crash_risk"


class TestOperatorTrapScreen:
    """The 2% upper-circuit trap: the stock traded a real intraday range and
    only locked at the close, so the zero-range detector never fires and
    momentum reads the printed move as strength."""

    @staticmethod
    def _tradable(**overrides):
        row = {
            "traded_value_60": 50_000_000.0,
            "zero_return_fraction_60": 0.0,
            "circuit_lock_fraction_60": 0.0,
            "circuit_locked_today": 0.0,
            "operator_trap_fraction_60": 0.0,
            "operator_trap_today": 0.0,
        }
        row.update(overrides)
        return row

    def _universe_with(self, leader_overrides, n_symbols=10):
        universe = {}
        for i in range(1, n_symbols + 1):
            symbol = f"SYM{i}"
            extra = self._tradable(**leader_overrides) if symbol == "SYM10" else self._tradable()
            universe[symbol] = _features(
                close=100.0, mom_9m_skip1m=i / 100.0, realized_vol_60=0.20, **extra
            )
        return universe

    def test_a_stock_locked_in_an_upper_circuit_never_becomes_a_buy(self):
        """The DoD: the momentum leader shut pinned at its upper limit, so the
        BUY it would otherwise generate is a phantom — there is no offer."""
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.2, min_universe=5))
        universe = self._universe_with({"operator_trap_today": 1.0})
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(universe, context)

        assert signals["SYM10"].signal == "AVOID"
        assert "upper circuit" in signals["SYM10"].rationale
        assert signals["SYM10"].extra["position_scale"] == 0.0
        # And it left the cross-section entirely rather than shifting percentiles.
        assert signals["SYM9"].signal == "BUY"
        assert signals["SYM9"].score == 100.0

    def test_a_sustained_operator_footprint_is_dropped(self):
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.2, min_universe=5))
        universe = self._universe_with({"operator_trap_fraction_60": 0.4})
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(universe, context)

        assert signals["SYM10"].signal == "AVOID"
        assert "operator footprint" in signals["SYM10"].rationale

    def test_an_occasional_lock_is_tolerated(self):
        """One lock is a news day. The screen excludes structural
        untradeability, not every stock that ever hit a limit."""
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.2, min_universe=5))
        universe = self._universe_with({"operator_trap_fraction_60": 0.02})
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(universe, context)

        assert signals["SYM10"].signal == "BUY"

    def test_the_screen_is_tunable(self):
        strategy = MomentumStrategy(
            _momentum_config(top_percentile=0.2, min_universe=5, max_operator_trap_fraction=0.5)
        )
        universe = self._universe_with({"operator_trap_fraction_60": 0.4})
        context = StrategyContext(risk=_risk_params())

        assert strategy.score_batch(universe, context)["SYM10"].signal == "BUY"

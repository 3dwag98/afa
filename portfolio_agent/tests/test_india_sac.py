"""Tests for the SAC allocation strategy.

Weighted toward the failure modes rather than the happy path, because the
happy path here is a matrix multiply and the failure modes are the expensive
part: an untrained actor that trades on noise, an allocation weight published
as a probability, and a gross reward:risk fed to a net gate.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="the SAC strategy needs the gpu extra")

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.features.scaling import FeatureScaler
from portfolio_agent.strategies.india_sac import (
    DEFAULT_SAC_FEATURES, IndiaSACStrategy, SACActorNetwork,
)
from portfolio_agent.strategies.registry import get_available_strategies, load_strategy
from portfolio_agent.strategies.types import RiskParams, StrategyContext


def _risk_params(**overrides):
    defaults = dict(
        target_prob_profit=0.55, min_reward_risk=1.2, min_price_inr=20.0,
        portfolio_value_inr=1_000_000.0, risk_per_trade_pct=0.01,
        max_single_position_pct=0.03,
    )
    defaults.update(overrides)
    return RiskParams(**defaults)


def _context(mc=None):
    return StrategyContext(risk=_risk_params(), weights={}, mc_result=mc)


def _features(n=300, close=1500.0, atr=30.0, seed=0):
    """A frame with every required feature finite on the last row."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-02", periods=n)
    frame = pd.DataFrame(index=index)
    for name in DEFAULT_SAC_FEATURES:
        frame[name] = rng.normal(0.0, 1.0, n)
    frame["close"] = close + rng.normal(0.0, 1.0, n)
    frame["atr_14"] = atr
    return frame


def _config(**params):
    return StrategyConfig(type="india_sac", params=params)


def _train_a_checkpoint(tmp_path, bias, features=None, scaler=None, hidden_dim=8):
    """Write a checkpoint whose actor emits a near-constant weight.

    The final bias is set directly so the sigmoid output is predictable; the
    tests care about the plumbing around the network, not about what a real
    policy would learn.
    """
    names = list(features or DEFAULT_SAC_FEATURES)
    model = SACActorNetwork(state_dim=len(names), action_dim=1, hidden_dim=hidden_dim)
    with torch.no_grad():
        model.mean_head.weight.zero_()
        model.mean_head.bias.fill_(bias)

    metadata = {"feature_names": names, "hidden_dim": hidden_dim}
    if scaler is not None:
        metadata["feature_scaler"] = scaler.to_dict()

    path = tmp_path / "india_sac_best.pt"
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)
    return path


class TestRefusesToRunUntrained:
    """The most expensive failure this strategy could have.

    A randomly-initialized sigmoid emits arbitrary values in [0, 1], so a large
    share of any universe clears a 0.60 threshold — the platform would place
    real trades on noise while reporting a healthy allocation score. Loading
    must fail rather than improvise.
    """

    def test_load_fails_when_no_checkpoint_exists(self, tmp_path):
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))

        assert strategy.load() is False

    def test_scoring_raises_rather_than_returning_neutral_signals(self, tmp_path):
        """A blanket except that returned all-HOLD would make a broken model
        indistinguishable from a market with no opportunities, permanently."""
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))

        with pytest.raises(RuntimeError, match="untrained or unloadable"):
            strategy.score_batch({"A.NS": _features()}, _context())

    def test_a_corrupt_checkpoint_also_fails_closed(self, tmp_path):
        (tmp_path / "india_sac_best.pt").write_bytes(b"not a torch checkpoint")
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))

        assert strategy.load() is False


class TestUnitsAreNotConfused:
    """Three fields that would each look plausible while being wrong."""

    def test_the_allocation_weight_is_not_published_as_a_probability(self, tmp_path):
        """`probability_profit` reaches the live Excel report and the SQLite
        recommendation row labelled as a probability of profit. The actor's
        output is an allocation on a different scale entirely."""
        _train_a_checkpoint(tmp_path, bias=2.0)  # sigmoid(2) ~ 0.881
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))

        signal = strategy.score_batch({"A.NS": _features()}, _context())["A.NS"]

        assert signal.extra["sac_allocation_weight"] == pytest.approx(0.8808, abs=1e-3)
        assert signal.component_scores["SAC"] == pytest.approx(0.8808, abs=1e-3)
        # ...and the probability field stays empty when no MC result was supplied.
        assert signal.probability_profit == 0.0
        assert "prob=n/a" in signal.rationale

    def test_the_monte_carlo_supplies_the_probability_when_present(self, tmp_path):
        from portfolio_agent.src.monte_carlo import MonteCarloResult

        _train_a_checkpoint(tmp_path, bias=2.0)
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))
        mc = MonteCarloResult(probability_profit=0.63, expected_return_pct=0.04)

        signal = strategy.score_batch({"A.NS": _features()}, _context(mc))["A.NS"]

        assert signal.probability_profit == pytest.approx(0.63)

    def test_reward_risk_is_net_of_friction_not_gross(self, tmp_path):
        """compliance.min_reward_risk is a *net* gate. A gross ratio fed to it
        overstates every candidate by roughly the round-trip cost stack."""
        _train_a_checkpoint(tmp_path, bias=2.0)
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))

        signal = strategy.score_batch({"A.NS": _features()}, _context())["A.NS"]

        gross = (signal.target_price - signal.entry_price) / (
            signal.entry_price - signal.stop_price
        )
        assert signal.reward_risk < gross

    def test_stops_come_from_atr_multiples_not_fixed_percentages(self, tmp_path):
        """A filled position inherits the signal's own levels, so a hardcoded
        3%/5% would put the exit plan out of step with risk.atr_*_multiplier."""
        _train_a_checkpoint(tmp_path, bias=2.0)
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))
        frame = _features(close=1500.0, atr=30.0)

        signal = strategy.score_batch({"A.NS": frame}, _context())["A.NS"]

        # calculate_stop_target rounds to paise, so the tolerance is the
        # function's own precision rather than floating point's.
        entry = signal.entry_price
        assert signal.stop_price == pytest.approx(entry - 1.5 * 30.0, abs=0.01)
        assert signal.target_price == pytest.approx(entry + 2.0 * 30.0, abs=0.01)


class TestSignalMapping:
    @pytest.mark.parametrize(
        "bias,expected",
        [(3.0, "BUY"), (0.0, "HOLD"), (-3.0, "SELL")],
    )
    def test_thresholds_map_the_weight_to_an_action(self, tmp_path, bias, expected):
        _train_a_checkpoint(tmp_path, bias=bias)
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))

        signal = strategy.score_batch({"A.NS": _features()}, _context())["A.NS"]

        assert signal.signal == expected

    def test_an_invalid_stop_avoids_regardless_of_conviction(self, tmp_path):
        """A stop at or above entry cannot be traded, however much the policy
        wants the name."""
        _train_a_checkpoint(tmp_path, bias=5.0)
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))
        frame = _features(close=10.0, atr=100.0)  # stop would land below zero

        signal = strategy.score_batch({"A.NS": frame}, _context())["A.NS"]

        assert signal.signal == "AVOID"

    def test_thresholds_must_be_ordered(self, tmp_path):
        with pytest.raises(ValueError, match="exit_threshold"):
            IndiaSACStrategy(_config(
                models_dir=str(tmp_path), action_threshold=0.3, exit_threshold=0.8
            ))


class TestEverySymbolGetsASignal:
    """Callers index the result directly, so a missing key is a crash."""

    def test_short_history_returns_avoid_not_a_missing_key(self, tmp_path):
        _train_a_checkpoint(tmp_path, bias=2.0)
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))

        signals = strategy.score_batch(
            {"GOOD.NS": _features(), "SHORT.NS": _features(n=50)}, _context()
        )

        assert set(signals) == {"GOOD.NS", "SHORT.NS"}
        assert signals["SHORT.NS"].signal == "AVOID"
        assert "needs 252 bars" in signals["SHORT.NS"].rationale

    def test_a_nan_in_the_latest_row_is_not_scoreable(self, tmp_path):
        _train_a_checkpoint(tmp_path, bias=2.0)
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))
        frame = _features()
        frame.iloc[-1, frame.columns.get_loc("rsi_14")] = np.nan

        signals = strategy.score_batch({"A.NS": frame}, _context())

        assert signals["A.NS"].signal == "AVOID"

    def test_an_empty_frame_is_not_scoreable(self, tmp_path):
        _train_a_checkpoint(tmp_path, bias=2.0)
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))

        signals = strategy.score_batch({"A.NS": pd.DataFrame()}, _context())

        assert signals["A.NS"].signal == "AVOID"


class TestCheckpointContract:
    def test_the_feature_scaler_travels_with_the_weights(self, tmp_path):
        """Same contract MLStrategy uses. Without it the actor is handed raw
        price levels at inference — `close` near 1500 — after training on
        standardized inputs."""
        scaler = FeatureScaler.fit(np.random.default_rng(0).normal(
            0.0, 1.0, size=(200, len(DEFAULT_SAC_FEATURES))
        ))
        _train_a_checkpoint(tmp_path, bias=2.0, scaler=scaler)
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))

        assert strategy.load() is True
        assert strategy._scaler is not None

    def test_the_feature_order_travels_with_the_weights(self, tmp_path):
        """A state vector assembled in a different order than training used is
        undetectable from the weights and silently scores nonsense."""
        reordered = list(reversed(DEFAULT_SAC_FEATURES))
        _train_a_checkpoint(tmp_path, bias=2.0, features=reordered)
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))

        assert strategy.load() is True
        assert strategy.required_features() == reordered

    def test_a_wider_action_head_is_rejected(self):
        with pytest.raises(ValueError, match="action_dim must be 1"):
            SACActorNetwork(state_dim=4, action_dim=2)


class TestDeterminism:
    def test_two_identical_scoring_rounds_agree(self, tmp_path):
        """SAC trains a stochastic policy; inference must use the mean action.
        Sampling here would make one backtest disagree with itself."""
        _train_a_checkpoint(tmp_path, bias=0.7)
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))
        universe = {f"T{i}.NS": _features(seed=i) for i in range(5)}

        first = strategy.score_batch(universe, _context())
        second = strategy.score_batch(universe, _context())

        assert {s: v.score for s, v in first.items()} == {
            s: v.score for s, v in second.items()
        }

    def test_batched_and_single_scoring_agree(self, tmp_path):
        """score() delegates to score_batch, so a name must not be scored
        differently depending on how many others were asked for."""
        _train_a_checkpoint(tmp_path, bias=0.7)
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))
        universe = {f"T{i}.NS": _features(seed=i) for i in range(4)}

        batched = strategy.score_batch(universe, _context())
        alone = strategy.score("T2.NS", universe["T2.NS"], _context())

        assert alone.score == batched["T2.NS"].score


class TestRegistration:
    def test_the_strategy_is_registered(self):
        assert "india_sac" in get_available_strategies()

    def test_it_loads_through_the_registry(self, tmp_path):
        strategy = load_strategy(_config(models_dir=str(tmp_path)))

        assert isinstance(strategy, IndiaSACStrategy)
        assert strategy.name == "india_sac"

    def test_it_declares_batched_scoring(self, tmp_path):
        strategy = IndiaSACStrategy(_config(models_dir=str(tmp_path)))

        assert strategy.supports_gpu_batch is True
        assert strategy.requires_full_batch is False

    def test_every_declared_feature_exists_in_the_registry(self):
        """A required feature the pipeline cannot build fails every ticker."""
        from portfolio_agent.features.registry import list_features

        assert set(DEFAULT_SAC_FEATURES) <= set(list_features())

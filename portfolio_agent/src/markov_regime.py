"""Markov-switching regime model for the benchmark return series.

What this replaces. `src/regime.py` classifies the market with a deterministic
threshold cascade — distance from a 200-day moving average, 60-day realized
volatility against a 20% target, ADX-14 — and emits a hard string label. The
branch ordering is well reasoned and the fail-neutral behaviour is right, but
as a statistical object it has four defects:

1. **No state probability.** The output is a label, so everything downstream is
   a step function of a noisy continuous input. A benchmark oscillating around
   its 200-day average flips the whole permission map day to day; there is no
   hysteresis and no smoothing.
2. **No persistence model.** Real regimes are strongly persistent — empirical
   daily self-transition probabilities for equity volatility states run 0.95 to
   0.99. A threshold rule's implicit persistence is whatever the autocorrelation
   of its inputs happens to be, which is neither the same thing nor estimable.
3. **The thresholds are free parameters chosen by hand** (200 days, 1.5x, ADX
   20, ±2%) and never fitted or tested. Each is a researcher degree of freedom
   inflating every backtest run through it.
4. **The states are defined, not discovered.** Four hand-named states with
   hand-set boundaries cannot find the structure actually present in the data,
   and model selection over the number of states is not even expressible.

What this provides instead. A K-state Gaussian hidden Markov model,

    r_t = mu_{S_t} + sigma_{S_t} * z_t,    Pr(S_t = j | S_{t-1} = i) = p_ij

fitted by Baum-Welch (EM), with the number of states chosen by BIC rather than
by hand. Every defect above has a direct answer: a filtered probability vector
instead of a label, an estimated transition matrix instead of implicit
persistence, maximum-likelihood parameters instead of hand-set thresholds, and
states discovered from the data instead of named in advance.

**The look-ahead distinction that matters.** Two quantities come out of an HMM
and they are not interchangeable:

- ``filtered_probabilities`` — P(S_t | information up to and including t). This
  is the only one a trading decision may use, and the forward recursion that
  produces it never touches an observation after t.
- ``smoothed_probabilities`` — P(S_t | the entire sample, including the future).
  Strictly better for describing history and strictly unusable for trading. It
  is provided for research plots and is named to make misuse obvious.

Parameter estimation is a separate question: an HMM fitted on the whole sample
has seen the test period even if the filter has not. `fit_gaussian_hmm` is
therefore meant to be fitted on a training window and the filter run forward
from those frozen parameters — see `filtered_probabilities`, which takes a
model and a series rather than doing both at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TRADING_DAYS_PER_YEAR = 252

# Variance floor. Without one, EM can drive a state's variance toward zero by
# wrapping it tightly around a single observation, which sends the likelihood to
# infinity — the classic degenerate solution of a Gaussian mixture.
_MIN_VARIANCE = 1e-12

# Probabilities below this are treated as zero when normalizing, so a
# numerically impossible state cannot produce a 0/0.
_MIN_PROBABILITY = 1e-300

# Conventional names for volatility-ordered states, used when a caller wants
# something readable. Purely cosmetic: the model discovers the states, and
# these are attached afterward in ascending order of fitted volatility.
STATE_NAMES = {
    1: ("SINGLE",),
    2: ("CALM", "TURBULENT"),
    3: ("CALM", "TURBULENT", "CRISIS"),
    4: ("CALM", "NORMAL", "TURBULENT", "CRISIS"),
    5: ("CALM", "NORMAL", "ELEVATED", "TURBULENT", "CRISIS"),
    6: ("CALM", "NORMAL", "ELEVATED", "TURBULENT", "CRISIS", "PANIC"),
}


@dataclass
class GaussianHMM:
    """A fitted K-state Gaussian hidden Markov model.

    States are always ordered by ascending fitted volatility. HMM states are
    exchangeable — relabelling them permutes the parameters without changing
    the likelihood — so without a convention, two runs on the same data can
    produce identical models with the labels swapped, and anything keyed to a
    state index becomes unreproducible.
    """

    means: np.ndarray  # (K,) per-state mean daily return
    variances: np.ndarray  # (K,) per-state daily return variance
    transition_matrix: np.ndarray  # (K, K), row i = distribution of S_t given S_{t-1}=i
    initial_distribution: np.ndarray  # (K,)
    log_likelihood: float
    n_observations: int
    iterations: int
    converged: bool

    @property
    def n_states(self) -> int:
        return int(self.means.size)

    @property
    def state_names(self) -> Tuple[str, ...]:
        """Readable names in ascending volatility order."""
        return STATE_NAMES.get(self.n_states, tuple(f"STATE_{i}" for i in range(self.n_states)))

    @property
    def annualized_volatility(self) -> np.ndarray:
        """Per-state volatility, annualized — the interpretable scale."""
        return np.sqrt(np.maximum(self.variances, 0.0)) * math.sqrt(TRADING_DAYS_PER_YEAR)

    @property
    def annualized_drift(self) -> np.ndarray:
        """Per-state mean return, annualized."""
        return self.means * TRADING_DAYS_PER_YEAR

    @property
    def self_transition_probabilities(self) -> np.ndarray:
        """Diagonal of P — how persistent each state is.

        The headline diagnostic. Equity volatility regimes are strongly
        persistent, so a fitted model whose diagonal sits well below ~0.9 is
        usually describing noise rather than regimes, and should be treated as
        a reason to doubt the fit rather than a finding about the market.
        """
        return np.diag(self.transition_matrix)

    @property
    def expected_durations(self) -> np.ndarray:
        """Expected number of consecutive days in each state, 1/(1-p_ii).

        The most legible statement of what the model believes: "crisis states
        last about three weeks" is a claim a person can agree or disagree with,
        which is more than a threshold cascade ever offered.
        """
        diagonal = np.clip(self.self_transition_probabilities, 0.0, 1.0 - 1e-12)
        return 1.0 / (1.0 - diagonal)

    @property
    def n_parameters(self) -> int:
        """Free parameters: K(K-1) transitions + K means + K variances + (K-1) initial."""
        k = self.n_states
        return k * (k - 1) + 2 * k + (k - 1)

    def bic(self) -> float:
        """Bayesian information criterion; lower is better.

        BIC = -2 logL + p log(n). This is what chooses K, in place of the four
        hand-named states the threshold cascade fixed in advance.
        """
        if self.n_observations <= 0:
            return float("inf")
        return -2.0 * self.log_likelihood + self.n_parameters * math.log(self.n_observations)

    def aic(self) -> float:
        """Akaike information criterion; lower is better."""
        return -2.0 * self.log_likelihood + 2.0 * self.n_parameters

    def stationary_distribution(self) -> np.ndarray:
        """Long-run share of time spent in each state.

        The normalized left eigenvector of P for eigenvalue 1. Useful as a
        prior for the filter's first step, and as a sanity check: a state the
        model says the market occupies 0.1% of the time is probably an
        artifact of a handful of observations.
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        vector = np.real(eigenvectors[:, index])
        vector = np.abs(vector)
        total = vector.sum()
        if total <= 0:
            return np.full(self.n_states, 1.0 / self.n_states)
        return vector / total

    def summary(self) -> Dict[str, object]:
        """Report-ready description of what was fitted."""
        return {
            "n_states": self.n_states,
            "state_names": list(self.state_names),
            "annualized_drift": [float(v) for v in self.annualized_drift],
            "annualized_volatility": [float(v) for v in self.annualized_volatility],
            "self_transition_probabilities": [
                float(v) for v in self.self_transition_probabilities
            ],
            "expected_durations_days": [float(v) for v in self.expected_durations],
            "stationary_distribution": [float(v) for v in self.stationary_distribution()],
            "log_likelihood": float(self.log_likelihood),
            "bic": float(self.bic()),
            "aic": float(self.aic()),
            "n_observations": int(self.n_observations),
            "converged": bool(self.converged),
            "iterations": int(self.iterations),
        }


def _emission_densities(observations: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """(T, K) Gaussian densities b_j(x_t)."""
    variances = np.maximum(variances, _MIN_VARIANCE)
    deviation = observations[:, None] - means[None, :]
    exponent = -0.5 * deviation**2 / variances[None, :]
    return np.exp(exponent) / np.sqrt(2.0 * np.pi * variances[None, :])


def _forward(
    densities: np.ndarray,
    transition_matrix: np.ndarray,
    initial_distribution: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled forward recursion (the Hamilton filter).

    Returns (alpha, scale) where alpha[t] is P(S_t | x_1..x_t) — already
    normalized by the scaling constants — and scale[t] is the one-step
    predictive likelihood, whose logs sum to the sample log-likelihood.

    Scaling rather than log-sum-exp because the recursion normalizes at every
    step anyway: over a few thousand daily observations the unscaled alphas
    underflow to zero within a hundred steps.
    """
    n_periods, n_states = densities.shape
    alpha = np.zeros((n_periods, n_states))
    scale = np.zeros(n_periods)

    weighted = initial_distribution * densities[0]
    scale[0] = weighted.sum()
    alpha[0] = weighted / max(scale[0], _MIN_PROBABILITY)

    for t in range(1, n_periods):
        predicted = alpha[t - 1] @ transition_matrix
        weighted = predicted * densities[t]
        scale[t] = weighted.sum()
        alpha[t] = weighted / max(scale[t], _MIN_PROBABILITY)

    return alpha, scale


def _backward(
    densities: np.ndarray,
    transition_matrix: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Scaled backward recursion, using the forward pass's scaling constants."""
    n_periods, n_states = densities.shape
    beta = np.zeros((n_periods, n_states))
    beta[-1] = 1.0

    for t in range(n_periods - 2, -1, -1):
        beta[t] = transition_matrix @ (densities[t + 1] * beta[t + 1])
        beta[t] /= max(scale[t + 1], _MIN_PROBABILITY)

    return beta


def _initial_parameters(observations: np.ndarray, n_states: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic starting means and variances, seeded by volatility.

    Observations are split into K groups by quantiles of |r|, so the initial
    states already differ in the dimension that actually separates equity
    regimes. This is deterministic on purpose: EM finds a local optimum, so a
    random start would make the fitted regime map — and therefore which sleeves
    may trade — differ between two runs on identical data.
    """
    magnitude = np.abs(observations)
    edges = np.quantile(magnitude, np.linspace(0.0, 1.0, n_states + 1))
    # Nudge the top edge so the largest observation lands inside a bucket.
    edges[-1] = np.inf

    means = np.zeros(n_states)
    variances = np.zeros(n_states)
    for state in range(n_states):
        mask = (magnitude >= edges[state]) & (magnitude < edges[state + 1])
        bucket = observations[mask]
        if bucket.size < 2:
            bucket = observations
        means[state] = float(np.mean(bucket))
        variances[state] = float(max(np.var(bucket), _MIN_VARIANCE))

    # Guarantee distinguishable variances even on near-constant input, or EM
    # starts from K identical states and never separates them.
    if np.allclose(variances, variances[0]):
        base = max(float(np.var(observations)), _MIN_VARIANCE)
        variances = base * np.linspace(0.5, 2.0, n_states)

    return means, variances


def fit_gaussian_hmm(
    returns: Sequence[float],
    n_states: int = 3,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
    min_observations: int = 250,
) -> Optional[GaussianHMM]:
    """Fit a K-state Gaussian HMM by Baum-Welch (EM).

    Fit this on a *training window* and run the filter forward from the frozen
    parameters. Fitting on the whole sample and then reading the filter over
    that same sample is a subtler leak than using smoothed probabilities, but
    it is still a leak: the parameters saw the test period.

    Args:
        returns: Daily returns of the benchmark (or a market proxy).
        n_states: K. Use `select_n_states` rather than guessing.
        max_iterations: EM iteration cap.
        tolerance: Convergence threshold on the log-likelihood improvement.
        min_observations: Refuse to fit below this. A transition matrix has
            K(K-1) free parameters estimated from the handful of transitions
            actually observed, and a regime model fitted to one year of data
            mostly describes that year.

    Returns:
        A fitted GaussianHMM with states ordered by ascending volatility, or
        None when there is too little usable history — in which case callers
        should fall back to the threshold classification rather than trade a
        model they could not fit.
    """
    observations = np.asarray(returns, dtype=float).ravel()
    observations = observations[np.isfinite(observations)]

    n_periods = observations.size
    if n_states < 1 or n_periods < max(min_observations, n_states * 10):
        return None
    if float(np.var(observations)) <= _MIN_VARIANCE:
        return None

    means, variances = _initial_parameters(observations, n_states)
    transition_matrix = np.full((n_states, n_states), 0.05 / max(1, n_states - 1))
    np.fill_diagonal(transition_matrix, 0.95)
    if n_states == 1:
        transition_matrix = np.ones((1, 1))
    initial_distribution = np.full(n_states, 1.0 / n_states)

    previous_log_likelihood = -np.inf
    log_likelihood = -np.inf
    converged = False
    iteration = 0

    for iteration in range(1, int(max_iterations) + 1):
        densities = _emission_densities(observations, means, variances)
        alpha, scale = _forward(densities, transition_matrix, initial_distribution)

        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            return None
        log_likelihood = float(np.sum(np.log(np.maximum(scale, _MIN_PROBABILITY))))

        beta = _backward(densities, transition_matrix, scale)

        # E-step: gamma_t(i) = P(S_t = i | all data), xi_t(i,j) the joint.
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma = gamma / np.maximum(gamma_sum, _MIN_PROBABILITY)

        if n_periods > 1:
            future = densities[1:] * beta[1:]  # (T-1, K)
            xi = (
                alpha[:-1, :, None]
                * transition_matrix[None, :, :]
                * future[:, None, :]
                / np.maximum(scale[1:, None, None], _MIN_PROBABILITY)
            )
            xi_sum = xi.sum(axis=0)
        else:
            xi_sum = np.zeros((n_states, n_states))

        # M-step.
        initial_distribution = gamma[0].copy()
        state_weight = gamma.sum(axis=0)

        if n_periods > 1:
            row_totals = xi_sum.sum(axis=1, keepdims=True)
            transition_matrix = xi_sum / np.maximum(row_totals, _MIN_PROBABILITY)
            # A state the data never visits leaves an all-zero row, which is
            # not a distribution; make it absorbing rather than undefined.
            empty_rows = row_totals.ravel() <= _MIN_PROBABILITY
            if np.any(empty_rows):
                transition_matrix[empty_rows] = 0.0
                transition_matrix[empty_rows, np.where(empty_rows)[0]] = 1.0

        weights = np.maximum(state_weight, _MIN_PROBABILITY)
        means = (gamma * observations[:, None]).sum(axis=0) / weights
        deviation = observations[:, None] - means[None, :]
        variances = (gamma * deviation**2).sum(axis=0) / weights
        variances = np.maximum(variances, _MIN_VARIANCE)

        if log_likelihood - previous_log_likelihood < tolerance and iteration > 1:
            converged = True
            break
        previous_log_likelihood = log_likelihood

    # Identifiability: order by ascending volatility so state indices mean the
    # same thing across runs, and permute every parameter consistently.
    order = np.argsort(variances)
    means = means[order]
    variances = variances[order]
    transition_matrix = transition_matrix[np.ix_(order, order)]
    initial_distribution = initial_distribution[order]

    # Rows can drift off exactly 1 through the permutation and the floors.
    row_totals = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = transition_matrix / np.maximum(row_totals, _MIN_PROBABILITY)
    initial_distribution = initial_distribution / max(
        float(initial_distribution.sum()), _MIN_PROBABILITY
    )

    return GaussianHMM(
        means=means,
        variances=variances,
        transition_matrix=transition_matrix,
        initial_distribution=initial_distribution,
        log_likelihood=log_likelihood,
        n_observations=n_periods,
        iterations=iteration,
        converged=converged,
    )


def filtered_probabilities(
    model: GaussianHMM,
    returns: Sequence[float],
) -> np.ndarray:
    """P(S_t | information up to and including t), shape (T, K).

    **The only state estimate a trading decision may use.** The forward
    recursion at step t reads observations 1..t and nothing else, so running it
    over a series produces, at every row, the probability an observer standing
    at that date would have held. That is what makes it usable as a signal at
    all, and it is worth stating explicitly because the smoothed alternative
    below looks better on every plot and is unusable.

    Run this with parameters fitted on an earlier window; passing a model
    fitted on this same series leaks the parameters, even though the recursion
    itself does not look forward.
    """
    observations = np.asarray(returns, dtype=float).ravel()
    observations = observations[np.isfinite(observations)]
    if observations.size == 0:
        return np.zeros((0, model.n_states))

    densities = _emission_densities(observations, model.means, model.variances)
    alpha, _ = _forward(densities, model.transition_matrix, model.initial_distribution)
    return alpha


def smoothed_probabilities(
    model: GaussianHMM,
    returns: Sequence[float],
) -> np.ndarray:
    """P(S_t | the WHOLE sample), shape (T, K) — research only, never a signal.

    Uses observations after t, so every row encodes information the market had
    not yet revealed. It is the right quantity for describing history ("when
    was the 2020 crisis, in hindsight?") and produces a visibly cleaner state
    sequence than the filter, which is exactly what makes it dangerous: a
    backtest wired to this will look excellent and be worthless.
    """
    observations = np.asarray(returns, dtype=float).ravel()
    observations = observations[np.isfinite(observations)]
    if observations.size == 0:
        return np.zeros((0, model.n_states))

    densities = _emission_densities(observations, model.means, model.variances)
    alpha, scale = _forward(densities, model.transition_matrix, model.initial_distribution)
    beta = _backward(densities, model.transition_matrix, scale)

    gamma = alpha * beta
    totals = gamma.sum(axis=1, keepdims=True)
    return gamma / np.maximum(totals, _MIN_PROBABILITY)


def forecast_state_distribution(
    model: GaussianHMM,
    current_probabilities: Sequence[float],
    horizon_days: int = 1,
) -> np.ndarray:
    """P(S_{t+h}) = pi_t . P^h — where the regime is likely to be, not just where it is.

    A threshold cascade cannot answer this at all: it describes today and has
    no mechanism for tomorrow. An estimated transition matrix does, which is
    what makes a regime model anticipatory rather than merely descriptive.
    """
    probabilities = np.asarray(current_probabilities, dtype=float).ravel()
    if probabilities.size != model.n_states:
        raise ValueError(
            f"expected {model.n_states} state probabilities, got {probabilities.size}"
        )
    total = probabilities.sum()
    if total <= 0:
        return model.stationary_distribution()

    projected = probabilities / total
    for _ in range(max(0, int(horizon_days))):
        projected = projected @ model.transition_matrix
    return projected


def select_n_states(
    returns: Sequence[float],
    candidates: Sequence[int] = (2, 3, 4),
    criterion: str = "bic",
    **fit_kwargs,
) -> Tuple[Optional[GaussianHMM], List[Dict[str, float]]]:
    """Fit each candidate K and return the best model by BIC (or AIC).

    The number of regimes becomes an estimate with a stated criterion rather
    than four states named in advance. BIC is the default because it penalizes
    parameters harder than AIC, and on a K-state HMM the parameter count grows
    quadratically in K — left to AIC, the selection reliably prefers more
    states than the data supports.

    Returns:
        (best_model, scores) where scores lists every candidate's criterion
        value, so the margin between K and K+1 is visible rather than implied.
        best_model is None when no candidate could be fitted.
    """
    if criterion not in ("bic", "aic"):
        raise ValueError(f"criterion must be 'bic' or 'aic', got {criterion!r}")

    scores: List[Dict[str, float]] = []
    best_model: Optional[GaussianHMM] = None
    best_score = float("inf")

    for k in candidates:
        model = fit_gaussian_hmm(returns, n_states=int(k), **fit_kwargs)
        if model is None:
            continue
        score = model.bic() if criterion == "bic" else model.aic()
        scores.append({
            "n_states": float(k),
            "bic": float(model.bic()),
            "aic": float(model.aic()),
            "log_likelihood": float(model.log_likelihood),
            "converged": float(model.converged),
        })
        if score < best_score:
            best_score, best_model = score, model

    return best_model, scores


def sleeve_weights(
    state_probabilities: Sequence[float],
    affinity: Dict[str, Sequence[float]],
) -> Dict[str, float]:
    """Continuous per-sleeve weights from a state probability vector.

        w_sleeve = sum_k pi_k * a_{sleeve,k}

    This is the point of the whole module. The meta-orchestrator currently keys
    a permission map off a hard regime string, so a benchmark oscillating
    around its 200-day average flips which sleeves may trade from one session
    to the next. Weighting by the filtered probability removes that entirely:
    pi_t varies smoothly, so the permission map does too, and the existing
    "mute, don't veto" semantics — a good design — survive unchanged.

    Args:
        state_probabilities: pi_t, the filtered distribution over states.
        affinity: Per-sleeve preference for each state, in the same
            volatility-ascending order the model reports. Values are usually in
            [0, 1]; 1 means "fully enabled in this state", 0 means muted.

    Returns:
        Sleeve name -> weight. A sleeve whose affinity row is the wrong length
        is rejected rather than silently truncated, since a misaligned row
        would enable a strategy in the state it was meant to be muted in.
    """
    probabilities = np.asarray(state_probabilities, dtype=float).ravel()
    total = probabilities.sum()
    if total <= 0:
        return {name: 0.0 for name in affinity}
    probabilities = probabilities / total

    weights: Dict[str, float] = {}
    for name, row in affinity.items():
        values = np.asarray(row, dtype=float).ravel()
        if values.size != probabilities.size:
            raise ValueError(
                f"affinity row for {name!r} has {values.size} entries but the model "
                f"has {probabilities.size} states"
            )
        weights[name] = float(np.dot(probabilities, values))
    return weights


@dataclass
class RegimeState:
    """The model's view of one date, in terms a caller can act on."""

    probabilities: np.ndarray  # filtered P(S_t | F_t), volatility-ascending
    state_names: Tuple[str, ...]
    expected_volatility: float  # probability-weighted annualized volatility
    most_likely_state: int
    confidence: float  # probability of the most likely state
    model_summary: Dict[str, object] = field(default_factory=dict)

    @property
    def most_likely_name(self) -> str:
        return self.state_names[self.most_likely_state]

    def as_dict(self) -> Dict[str, object]:
        return {
            "probabilities": {
                name: float(p) for name, p in zip(self.state_names, self.probabilities)
            },
            "expected_volatility": float(self.expected_volatility),
            "most_likely_state": self.most_likely_name,
            "confidence": float(self.confidence),
        }


def current_regime_state(
    model: GaussianHMM,
    returns: Sequence[float],
) -> Optional[RegimeState]:
    """Filter `returns` through `model` and describe the final date.

    `expected_volatility` is the probability-weighted average of the per-state
    volatilities rather than the volatility of the single most likely state.
    That distinction is the practical value of a probabilistic regime model: a
    60/40 split between calm and crisis is a materially riskier market than a
    confident calm reading, and a hard label reports both as "calm".
    """
    probabilities = filtered_probabilities(model, returns)
    if probabilities.shape[0] == 0:
        return None

    latest = probabilities[-1]
    most_likely = int(np.argmax(latest))
    return RegimeState(
        probabilities=latest,
        state_names=model.state_names,
        expected_volatility=float(np.dot(latest, model.annualized_volatility)),
        most_likely_state=most_likely,
        confidence=float(latest[most_likely]),
        model_summary=model.summary(),
    )


def assess_markov_regime(
    market_close: Optional["object"] = None,
    returns: Optional[Sequence[float]] = None,
    candidates: Sequence[int] = (2, 3, 4),
    train_fraction: float = 0.6,
    min_observations: int = 250,
) -> Optional[RegimeState]:
    """End-to-end regime read from a benchmark price series.

    Fits on the leading `train_fraction` of history and filters the *whole*
    series forward from those frozen parameters. Both halves of that matter:

    - Fitting on the full sample would let the parameters see the period the
      filter is later read over. The forward recursion still would not look
      ahead, but the model describing it would have — a subtler leak than
      using smoothed probabilities, and just as real.
    - Filtering the whole series (not just the test tail) is what makes the
      returned state a proper posterior: the filter needs to have walked
      through the intervening history to know where the market is now.

    Args:
        market_close: Benchmark close prices (a pandas Series, or anything
            with a `pct_change`). Ignored when `returns` is given.
        returns: Daily returns, if already computed.
        candidates: Values of K to select between by BIC.
        train_fraction: Share of the history used to estimate parameters.
        min_observations: Refuse to fit below this many training returns.

    Returns:
        A RegimeState describing the latest date, or None when the history is
        too short to fit — in which case callers should fall back to the
        threshold classification in src/regime.py rather than trade a model
        they could not estimate.
    """
    if returns is None:
        if market_close is None:
            return None
        try:
            series = market_close.pct_change().dropna()
        except AttributeError:
            return None
        observations = np.asarray(series, dtype=float)
    else:
        observations = np.asarray(returns, dtype=float).ravel()

    observations = observations[np.isfinite(observations)]
    if observations.size < min_observations:
        return None

    split = int(observations.size * min(max(train_fraction, 0.1), 1.0))
    training = observations[:split]
    if training.size < min_observations:
        return None

    model, _ = select_n_states(
        training, candidates=candidates, min_observations=min_observations
    )
    if model is None:
        return None

    return current_regime_state(model, observations)

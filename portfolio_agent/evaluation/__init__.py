"""Forecast evaluation: measure skill directly, without simulating a book.

    from portfolio_agent.config.loader import load_config
    from portfolio_agent.evaluation import evaluate_forecast

    result = evaluate_forecast(load_config(), "momentum", universe_size=200)
    print(result.render())

`metrics.py` holds pure functions over a tidy
`(date, symbol, score, forward_return)` panel; `harness.py` produces such a
panel from any registered strategy and reduces it to a `ForecastEvaluation`.
Nothing here constructs a `BacktestEngine`, and nothing here imports PyTorch at
module scope — a rule-based screen can be evaluated on an install that has
neither.
"""

from .costs import (
    CostModel,
    NetSpread,
    cost_notes,
    evaluate_net,
    one_way_turnover,
)
from .decay import DecayCurve, DecayPoint, decay_curve, decay_from_panel
from .harness import (
    ForecastEvaluation,
    FoldEvaluation,
    build_forecast_panel,
    compare_forecasts,
    evaluate_forecast,
    evaluate_panel,
    forward_return,
)
from .neutralize import (
    SIZE_PROXY_NOTE,
    NeutralizationResult,
    add_exposures,
    evaluate_neutralized,
    neutralize_panel,
    neutralized_ic,
    residualize,
    rolling_beta,
)
from .metrics import (
    BucketAnalysis,
    ErrorSummary,
    ICSummary,
    bucket_analysis,
    directional_hit_rate,
    rank_ic,
    rank_ic_series,
    rank_error_summary,
    score_dispersion,
    signal_decay,
    summarize_ic,
)

__all__ = [
    "BucketAnalysis",
    "CostModel",
    "DecayCurve",
    "DecayPoint",
    "NetSpread",
    "NeutralizationResult",
    "SIZE_PROXY_NOTE",
    "add_exposures",
    "cost_notes",
    "decay_curve",
    "decay_from_panel",
    "evaluate_net",
    "evaluate_neutralized",
    "one_way_turnover",
    "neutralize_panel",
    "neutralized_ic",
    "residualize",
    "rolling_beta",
    "ErrorSummary",
    "ForecastEvaluation",
    "FoldEvaluation",
    "ICSummary",
    "bucket_analysis",
    "build_forecast_panel",
    "compare_forecasts",
    "directional_hit_rate",
    "evaluate_forecast",
    "evaluate_panel",
    "forward_return",
    "rank_error_summary",
    "rank_ic",
    "rank_ic_series",
    "score_dispersion",
    "signal_decay",
    "summarize_ic",
]

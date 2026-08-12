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

from .harness import (
    ForecastEvaluation,
    FoldEvaluation,
    build_forecast_panel,
    compare_forecasts,
    evaluate_forecast,
    evaluate_panel,
    forward_return,
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

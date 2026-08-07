"""Outcome tracking module for portfolio agent self-learning loop."""

import random
import uuid
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.models import Recommendation, TradeOutcome
from src.storage import init_db, save_trade_outcome, get_open_trades


def simulate_outcome(
    recommendation: Recommendation,
    seed: int | None = None
) -> TradeOutcome:
    """Simulate a trade outcome for a given recommendation.
    
    Args:
        recommendation: The recommendation to simulate outcome for.
        seed: Optional random seed for reproducibility.
        
    Returns:
        TradeOutcome object with simulated results.
        
    Rules:
        - outcome_source = SIMULATED
        - Use recommendation.entry_price.
        - Simulate random return between -6% and +12%.
        - If return > 0, outcome = WIN, else LOSS.
        - exit_date = entry_date + 20 days.
        - exit_price = entry_price * (1 + return_pct/100).
    """
    if seed is not None:
        random.seed(seed)
    
    # Generate random return between -6% and +12%
    return_pct = random.uniform(-6, 12)
    
    # Determine outcome based on return
    outcome = "WIN" if return_pct > 0 else "LOSS"
    
    # Calculate dates
    entry_date = datetime.now()
    exit_date = entry_date + timedelta(days=20)
    
    # Calculate exit price
    exit_price = recommendation.entry_price * (1 + return_pct / 100)
    
    return TradeOutcome(
        trade_id=str(uuid.uuid4()),
        recommendation_id=recommendation.recommendation_id or "",
        symbol=recommendation.symbol,
        signal_trigger=recommendation.trigger,
        entry_date=entry_date.strftime("%Y-%m-%d"),
        entry_price=recommendation.entry_price,
        exit_date=exit_date.strftime("%Y-%m-%d"),
        exit_price=exit_price,
        outcome=outcome,
        return_pct=return_pct,
        outcome_source="SIMULATED"
    )


def mark_outcome_manual(
    sqlite_path: str,
    recommendation_id: str,
    exit_price: float,
    exit_date: str
) -> TradeOutcome:
    """Manually mark the outcome of a trade.
    
    Args:
        sqlite_path: Path to SQLite database file.
        recommendation_id: ID of the recommendation to mark.
        exit_price: Actual exit price of the trade.
        exit_date: Actual exit date (YYYY-MM-DD format).
        
    Returns:
        TradeOutcome object with the marked result.
        
    Rules:
        - Fetch recommendation from SQLite.
        - Calculate return_pct.
        - outcome = WIN if return_pct > 0 else LOSS.
        - outcome_source = MANUAL.
        - Save to trade_outcomes.
    """
    init_db(sqlite_path)
    
    import sqlite3
    
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Fetch recommendation
        cursor.execute("""
            SELECT * FROM recommendations WHERE recommendation_id = ?
        """, (recommendation_id,))
        
        row = cursor.fetchone()
        
        if row is None:
            raise ValueError(f"Recommendation {recommendation_id} not found")
        
        entry_price = row["entry_price"]
        symbol = row["symbol"]
        trigger = row["trigger"]
        entry_date = row["created_at"][:10]  # Extract YYYY-MM-DD
        
        # Calculate return percentage
        return_pct = ((exit_price - entry_price) / entry_price) * 100
        
        # Determine outcome
        outcome = "WIN" if return_pct > 0 else "LOSS"
        
        # Create trade outcome
        trade_outcome = TradeOutcome(
            trade_id=str(uuid.uuid4()),
            recommendation_id=recommendation_id,
            symbol=symbol,
            signal_trigger=trigger,
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            outcome=outcome,
            return_pct=return_pct,
            outcome_source="MANUAL"
        )
        
        # Save to database
        save_trade_outcome(sqlite_path, trade_outcome)
        
        return trade_outcome


def update_outcomes_from_market(
    sqlite_path: str,
    data: dict[str, pd.DataFrame]
) -> list[TradeOutcome]:
    """Update open trade outcomes using market data.
    
    Args:
        sqlite_path: Path to SQLite database file.
        data: Dictionary mapping ticker symbols to DataFrames with OHLCV data.
              Each DataFrame should have 'close' column and datetime index.
              
    Returns:
        List of updated TradeOutcome objects.
        
    Rules:
        - Find OPEN trade outcomes.
        - If market data contains a close price on or after entry_date + 20 days, mark outcome.
        - Use close price after 20 trading days if available.
        - outcome_source = MARKET.
        - Save updated outcomes.
    """
    init_db(sqlite_path)
    
    # Get all open trades
    open_trades = get_open_trades(sqlite_path)
    updated_outcomes: list[TradeOutcome] = []
    
    for trade in open_trades:
        symbol = trade.symbol
        
        # Check if we have market data for this symbol
        if symbol not in data:
            continue
            
        df = data[symbol].copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Parse entry date
        entry_date = pd.to_datetime(trade.entry_date)
        target_date = entry_date + timedelta(days=20)
        
        # Find the close price on or after target_date
        # We need the first close price at or after 20 days from entry
        mask = df.index >= target_date
        
        if not mask.any():
            # No data available after target date, skip this trade
            continue
        
        # Get the first available close price on or after target_date
        relevant_data = df[mask]
        if len(relevant_data) == 0:
            continue
            
        exit_price = float(relevant_data['close'].iloc[0])
        exit_date = relevant_data.index[0].strftime("%Y-%m-%d")
        
        # Calculate return percentage
        return_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        
        # Determine outcome
        outcome = "WIN" if return_pct > 0 else "LOSS"
        
        # Create updated trade outcome
        updated_outcome = TradeOutcome(
            trade_id=trade.trade_id,
            recommendation_id=trade.recommendation_id,
            symbol=trade.symbol,
            signal_trigger=trade.signal_trigger,
            entry_date=trade.entry_date,
            entry_price=trade.entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            outcome=outcome,
            return_pct=return_pct,
            outcome_source="MARKET"
        )
        
        # Save to database
        save_trade_outcome(sqlite_path, updated_outcome)
        updated_outcomes.append(updated_outcome)
    
    return updated_outcomes

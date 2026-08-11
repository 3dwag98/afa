"""
Backtest Reporting Module for Portfolio Agent.

Exports backtest results into a highly formatted, professional Excel workbook.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import xlsxwriter

logger = logging.getLogger(__name__)


# Expected columns for Trade_Log sheet (16 columns in exact order)
EXPECTED_COLUMNS = [
    'trade_id',
    'ticker',
    'entry_date',
    'entry_price',
    'exit_date',
    'exit_price',
    'quantity',
    'side',
    'signal_trigger',
    'gross_pnl',
    'transaction_costs',
    'taxes',
    'net_pnl',
    'return_pct',
    'holding_days',
    'exit_reason'
]

# Executive_Summary rows: (label, analytics key, unit).
#
# The unit is declared here rather than guessed from the value at write time.
# The old code inferred it with `value / 100 if abs(value) > 1 else value`,
# which silently multiplied any genuine sub-1% figure by 100 (a 0.5% win rate
# was written to Excel as 50%) and formatted the Sharpe ratio as a percentage.
#
# CONTRACT: every 'percent' metric is passed in PERCENT units (18.5 == 18.5%),
# never as a 0-1 decimal. See agents/backtester.py, which builds this dict.
SUMMARY_PERCENT = 'percent'
SUMMARY_RATIO = 'ratio'
SUMMARY_CURRENCY = 'currency'
SUMMARY_COUNT = 'count'

SUMMARY_METRICS = [
    ('CAGR (%)', 'cagr', SUMMARY_PERCENT),
    ('Sharpe Ratio', 'sharpe', SUMMARY_RATIO),
    # A Sharpe ratio without the number of configurations behind it is the
    # maximum of an unrecorded number of draws. PSR is the probability the true
    # Sharpe clears zero given the sample's length, skew and kurtosis; DSR is
    # the same probability measured against the Sharpe the best of N trials
    # would show by luck alone. A DSR below 0.95 means the headline number is
    # not distinguishable from what the search itself would produce on a
    # strategy with no edge. See src/performance_stats.py.
    ('Probabilistic Sharpe', 'probabilistic_sharpe', SUMMARY_RATIO),
    ('Deflated Sharpe', 'deflated_sharpe', SUMMARY_RATIO),
    ('Trials Deflated Against', 'n_trials', SUMMARY_COUNT),
    ('Sortino Ratio', 'sortino', SUMMARY_RATIO),
    # Portfolio-level risk, which per-position sizing cannot see. The multiple
    # is how many times larger the book's volatility actually was than sizing
    # each position independently implied — 1.0 would mean correlation cost
    # nothing, and on a long-only Indian equity book it does not.
    ('Book Volatility (%)', 'book_volatility', SUMMARY_PERCENT),
    ('Correlation Risk Multiple', 'correlation_risk_multiple', SUMMARY_RATIO),
    ('Diversification Ratio', 'diversification_ratio', SUMMARY_RATIO),
    ('Max Drawdown (%)', 'max_drawdown', SUMMARY_PERCENT),
    ('Profit Factor', 'profit_factor', SUMMARY_RATIO),
    ('Probability of Ruin (%)', 'probability_of_ruin', SUMMARY_PERCENT),
    ('Total Return (%)', 'total_return', SUMMARY_PERCENT),
    ('Annualized Volatility (%)', 'volatility', SUMMARY_PERCENT),
    ('Win Rate (%)', 'win_rate', SUMMARY_PERCENT),
    ('Total Trades', 'total_trades', SUMMARY_COUNT),
    ('Final Portfolio Value (₹)', 'final_portfolio_value', SUMMARY_CURRENCY),
    ('Initial Capital (₹)', 'initial_capital', SUMMARY_CURRENCY),
]

SUMMARY_UNITS = {label: unit for label, _, unit in SUMMARY_METRICS}

# Expected columns for Daily_Trade_Log sheet (11 columns in exact order)
EXPECTED_DAILY_COLUMNS = [
    'date',
    'ticker',
    'action',
    'price',
    'quantity',
    'position_value',
    'cash_balance',
    'total_portfolio_value',
    'score',
    'signal',
    'notes'
]


def _normalize_trade_log(trade_log) -> pd.DataFrame:
    """
    Normalize trade log into a flat DataFrame with consistent columns.
    
    Args:
        trade_log: List of trade dicts, or dict of trades.
        
    Returns:
        DataFrame with exactly 16 columns as defined in EXPECTED_COLUMNS.
    """
    # Handle None / empty
    if not trade_log:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    
    # If it's a dict, try to convert to list of its values
    if isinstance(trade_log, dict):
        trade_log = list(trade_log.values())
    
    # Ensure list of dicts
    rows = []
    for t in trade_log:
        if isinstance(t, dict):
            # Flatten nested dicts - skip any nested dict/list values
            row = {}
            for k in EXPECTED_COLUMNS:
                val = t.get(k)
                # Skip nested structures
                if isinstance(val, (dict, list)):
                    val = None
                row[k] = val
            rows.append(row)
        else:
            # Skip malformed entries but log them
            logger.warning(f"Skipping malformed trade entry: {t}")
            continue
    
    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    return df


def _normalize_daily_log(daily_log) -> pd.DataFrame:
    """
    Normalize daily activity log into a flat DataFrame with consistent columns.
    
    Args:
        daily_log: List of daily activity dicts.
        
    Returns:
        DataFrame with exactly 11 columns as defined in EXPECTED_DAILY_COLUMNS.
    """
    # Handle None / empty
    if not daily_log:
        return pd.DataFrame(columns=EXPECTED_DAILY_COLUMNS)
    
    # If it's a dict, try to convert to list of its values
    if isinstance(daily_log, dict):
        daily_log = list(daily_log.values())
    
    # Ensure list of dicts
    rows = []
    for d in daily_log:
        if isinstance(d, dict):
            # Flatten nested dicts - skip any nested dict/list values
            row = {}
            for k in EXPECTED_DAILY_COLUMNS:
                val = d.get(k)
                # Skip nested structures
                if isinstance(val, (dict, list)):
                    val = None
                row[k] = val
            rows.append(row)
        else:
            # Skip malformed entries but log them
            logger.warning(f"Skipping malformed daily activity entry: {d}")
            continue
    
    df = pd.DataFrame(rows, columns=EXPECTED_DAILY_COLUMNS)
    return df


def export_backtest_excel(
    analytics: Dict[str, Any],
    equity_curve: pd.Series,
    trade_log: List[Dict[str, Any]],
    brain_evolution: List[Dict[str, Any]],
    daily_activity_log: List[Dict[str, Any]],
    filepath: str,
    parallel_metrics: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export backtest results to a formatted Excel workbook.

    Args:
        analytics: Dictionary containing key performance metrics including:
            - cagr: Compound Annual Growth Rate
            - sharpe: Sharpe Ratio
            - sortino: Sortino Ratio
            - max_drawdown: Maximum Drawdown (%)
            - profit_factor: Profit Factor
            - probability_of_ruin: Probability of Ruin
            - monte_carlo_results: Dict with 5th, 50th, 95th percentile terminal wealth
        equity_curve: pd.Series with DateTimeIndex containing daily portfolio values.
            May include 'benchmark' column for Nifty 50 comparison.
        trade_log: List of trade dictionaries with columns:
            - entry_date, exit_date, ticker, side, entry_price, exit_price,
            - qty, gross_pnl, stt_taxes, slippage, net_pnl, holding_days, signal_trigger
        brain_evolution: List of dicts showing agent weights over time:
            - trading_day, weights (Trend, Breakout, Volume, MC_Prob)
        daily_activity_log: List of daily activity dicts with 11 keys:
            - date, ticker, action, price, quantity, position_value, cash_balance,
              total_portfolio_value, score, signal, notes
        filepath: Output file path for the Excel workbook.
        parallel_metrics: Optional dict with parallel execution info:
            - execution_time_seconds: Total execution time
            - num_workers: Number of parallel workers used
            - success_rate: Success/failure rate of parallel runs
            - total_tickers: Total tickers processed
            - failed_tickers: Number of tickers that failed
    
    Returns:
        str: Path to the created Excel file.
    
    Raises:
        ValueError: If aggregated data is empty or invalid.
    """
    # =========================================================================
    # DATA INTEGRITY CHECK - Critical validation before writing
    # =========================================================================
    logger.info("Validating backtest data before Excel export...")
    
    # Check equity curve
    if equity_curve is None or len(equity_curve) == 0:
        raise ValueError("Equity curve is empty. Cannot generate report with no portfolio data.")
    
    # Check analytics
    if not analytics:
        logger.warning("Analytics dictionary is empty. Using default values.")
        analytics = {
            'cagr': 0, 'sharpe': 0, 'sortino': 0, 'max_drawdown': 0,
            'profit_factor': 0, 'probability_of_ruin': 0, 'total_return': 0,
            'volatility': 0, 'win_rate': 0, 'total_trades': 0,
            'final_portfolio_value': equity_curve.iloc[-1] if len(equity_curve) > 0 else 0,
            'initial_capital': analytics.get('initial_capital', 1000000)
        }
    
    # Normalize and validate trade log
    trade_df = _normalize_trade_log(trade_log)
    if len(trade_df) == 0:
        logger.warning("Trade log is empty. Report will show no trades.")
    
    # Normalize and validate daily activity log
    daily_df = _normalize_daily_log(daily_activity_log)
    if len(daily_df) == 0:
        logger.warning("Daily activity log is empty. Report will show no daily activities.")
    
    # Validate brain evolution
    if not brain_evolution:
        logger.warning("Brain evolution is empty. Report will show no learning history.")
    
    logger.info(f"Data validation complete: {len(equity_curve)} equity points, "
                f"{len(trade_df)} trades, {len(daily_df)} daily activities")
    
    # Ensure output directory exists
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file to ensure clean write (avoid conflicts with sheet management)
    if output_path.exists():
        try:
            output_path.unlink()
            logger.info(f"Removed existing Excel file: {filepath}")
        except Exception as e:
            logger.warning(f"Could not remove existing file: {e}")

    # Create Excel writer with xlsxwriter engine for full formatting support
    # xlsxwriter provides better support for charts, conditional formatting, and data bars
    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })

        currency_format = workbook.add_format({
            'num_format': '₹#,##0.00',
            'border': 1
        })

        currency_positive = workbook.add_format({
            'num_format': '₹#,##0.00',
            'font_color': 'green',
            'border': 1
        })

        currency_negative = workbook.add_format({
            'num_format': '₹#,##0.00',
            'font_color': 'red',
            'border': 1
        })

        percent_format = workbook.add_format({
            'num_format': '0.00%',
            'border': 1
        })

        percent_positive = workbook.add_format({
            'num_format': '0.00%',
            'font_color': 'green',
            'border': 1
        })

        percent_negative = workbook.add_format({
            'num_format': '0.00%',
            'font_color': 'red',
            'border': 1
        })

        number_format = workbook.add_format({
            'num_format': '#,##0.00',
            'border': 1
        })

        title_format = workbook.add_format({
            'bold': True,
            'font_size': 16,
            'fg_color': '#2F5597',
            'font_color': 'white',
            'align': 'center',
            'valign': 'vcenter'
        })

        # =========================================================================
        # Sheet 1: Executive_Summary
        # =========================================================================
        summary_df = _create_executive_summary_df(analytics)
        summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False, header=False)

        worksheet_summary = writer.sheets['Executive_Summary']
        worksheet_summary.set_column('A:A', 25)
        worksheet_summary.set_column('B:B', 15)

        # Apply title format to first row
        worksheet_summary.write('A1', 'Metric', header_format)
        worksheet_summary.write('B1', 'Value', header_format)

        # Apply formatting to data rows. The unit comes from SUMMARY_METRICS,
        # so each value is scaled exactly once and ratios are never rendered
        # as percentages.
        for row_idx in range(1, len(summary_df) + 1):
            label = summary_df.iloc[row_idx - 1, 0]
            value = summary_df.iloc[row_idx - 1, 1]
            worksheet_summary.write(row_idx, 0, label)

            if not isinstance(value, (int, float)) or isinstance(value, bool):
                worksheet_summary.write(row_idx, 1, str(value))
                continue

            unit = SUMMARY_UNITS.get(label, SUMMARY_RATIO)
            if unit == SUMMARY_PERCENT:
                # Excel percent formats multiply by 100 on display, so a
                # percent-unit input has to be written as a fraction.
                fraction = value / 100.0
                fmt = percent_negative if fraction < 0 else percent_format
                worksheet_summary.write_number(row_idx, 1, fraction, fmt)
            elif unit == SUMMARY_CURRENCY:
                worksheet_summary.write_number(row_idx, 1, value, currency_format)
            elif unit == SUMMARY_COUNT:
                worksheet_summary.write_number(row_idx, 1, value, number_format)
            else:
                worksheet_summary.write_number(row_idx, 1, value, number_format)

        # Conditional formatting for Sharpe and MaxDD
        _apply_conditional_formatting_summary(worksheet_summary, summary_df, workbook)

        # =========================================================================
        # Sheet 2: Equity_Curve
        # =========================================================================
        equity_df = _prepare_equity_curve_df(equity_curve)
        equity_df.to_excel(writer, sheet_name='Equity_Curve', index=True, header=True)

        worksheet_equity = writer.sheets['Equity_Curve']
        worksheet_equity.set_column('A:A', 12)  # Date
        worksheet_equity.set_column('B:E', 18)  # value columns

        # Format header (column A is the Date index, then one per column)
        worksheet_equity.write(0, 0, 'Date', header_format)
        for col_num, col_name in enumerate(equity_df.columns, start=1):
            worksheet_equity.write(0, col_num, col_name, header_format)

        # Column letters are derived from the actual layout: 'Benchmark' is
        # only present for some runs, and hard-coding B/C/D made the equity
        # chart plot the date column as its values whenever it was missing.
        def _column_letter(name: str) -> str:
            return chr(ord('A') + 1 + list(equity_df.columns).index(name))

        last_row = len(equity_df) + 1
        dates_ref = f'=Equity_Curve!$A$2:$A${last_row}'

        # Create chart for Equity Curve
        chart_equity = workbook.add_chart({'type': 'line'})
        chart_equity.set_title({'name': 'Portfolio Value vs Benchmark'})
        chart_equity.set_x_axis({'name': 'Date'})
        chart_equity.set_y_axis({'name': 'Portfolio Value (₹)'})

        # Add portfolio value series
        value_col = _column_letter('Portfolio_Value')
        chart_equity.add_series({
            'name': 'Portfolio Value',
            'categories': dates_ref,
            'values': f'=Equity_Curve!${value_col}$2:${value_col}${last_row}',
            'line': {'color': '#2F5597', 'width': 2}
        })

        # Add benchmark series if available
        if 'Benchmark' in equity_df.columns:
            bench_col = _column_letter('Benchmark')
            chart_equity.add_series({
                'name': 'Nifty 50 Benchmark',
                'categories': dates_ref,
                'values': f'=Equity_Curve!${bench_col}$2:${bench_col}${last_row}',
                'line': {'color': '#ED7D31', 'width': 2, 'dash_type': 'dash'}
            })

        worksheet_equity.insert_chart('G2', chart_equity)

        # Create secondary axis chart for Drawdown
        chart_dd = workbook.add_chart({'type': 'line'})
        chart_dd.set_title({'name': 'Drawdown %'})
        chart_dd.set_y_axis({'name': 'Drawdown %', 'position': 'right'})

        dd_col = _column_letter('Drawdown_%')
        chart_dd.add_series({
            'name': 'Drawdown %',
            'categories': dates_ref,
            'values': f'=Equity_Curve!${dd_col}$2:${dd_col}${last_row}',
            'line': {'color': '#C00000', 'width': 1.5}
        })

        worksheet_equity.insert_chart('G20', chart_dd)

        # =========================================================================
        # Sheet 3: Trade_Log
        # =========================================================================
        trade_df = _normalize_trade_log(trade_log)
        
        # Verify we have exactly 16 columns
        assert trade_df.shape[1] == 16, f"Trade_Log should have 16 columns, got {trade_df.shape[1]}"
        logger.info(f"Trade_Log shape: {trade_df.shape}")
        
        # Set column widths for all 16 columns (defined outside the if block for reuse)
        col_widths = [
            ('A:A', 14),   # trade_id
            ('B:B', 12),   # ticker
            ('C:C', 12),   # entry_date
            ('D:D', 12),   # entry_price
            ('E:E', 12),   # exit_date
            ('F:F', 12),   # exit_price
            ('G:G', 10),   # quantity
            ('H:H', 8),    # side
            ('I:I', 15),   # signal_trigger
            ('J:J', 14),   # gross_pnl
            ('K:K', 14),   # transaction_costs
            ('L:L', 12),   # taxes
            ('M:M', 14),   # net_pnl
            ('N:N', 12),   # return_pct
            ('O:O', 12),   # holding_days
            ('P:P', 15),   # exit_reason
        ]
        
        if len(trade_df) > 0:
            trade_df.to_excel(writer, sheet_name='Trade_Log', index=False, header=True)

            worksheet_trades = writer.sheets['Trade_Log']
            
            for col_range, width in col_widths:
                worksheet_trades.set_column(col_range, width)

            # Get column indices for formatting
            entry_price_col = trade_df.columns.get_loc('entry_price')
            exit_price_col = trade_df.columns.get_loc('exit_price')
            gross_pnl_col = trade_df.columns.get_loc('gross_pnl')
            transaction_costs_col = trade_df.columns.get_loc('transaction_costs')
            taxes_col = trade_df.columns.get_loc('taxes')
            net_pnl_col = trade_df.columns.get_loc('net_pnl')
            return_pct_col = trade_df.columns.get_loc('return_pct')

            # Apply formatting to each row.
            #
            # Every write targets the column looked up above, never a hardcoded
            # index. The hardcoded version wrote exit_price into position 4 —
            # which is exit_DATE — so the exported Trade_Log showed a price
            # where every trade's exit date should have been.
            for row_idx in range(1, len(trade_df) + 1):
                row = trade_df.iloc[row_idx - 1]

                for col in (entry_price_col, exit_price_col,
                            transaction_costs_col, taxes_col):
                    value = row.iloc[col]
                    if pd.notna(value):
                        worksheet_trades.write_number(row_idx, col, value, currency_format)

                for col in (gross_pnl_col, net_pnl_col):
                    value = row.iloc[col]
                    if pd.notna(value):
                        fmt = currency_positive if value >= 0 else currency_negative
                        worksheet_trades.write_number(row_idx, col, value, fmt)

                # BacktestEngine writes return_pct in PERCENT units (5.0 ==
                # +5%), so it is always divided by 100 to get the fraction
                # Excel's percent format expects. The old `if abs(val) > 1`
                # guard left every trade between -1% and +1% scaled up 100x.
                val_return = row.iloc[return_pct_col]
                if pd.notna(val_return):
                    return_val = val_return / 100.0
                    fmt = percent_positive if return_val >= 0 else percent_negative
                    worksheet_trades.write_number(row_idx, return_pct_col, return_val, fmt)

            # Apply conditional formatting for the Net PnL column
            worksheet_trades.conditional_format(
                1, net_pnl_col, len(trade_df), net_pnl_col,
                {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': currency_positive}
            )
            worksheet_trades.conditional_format(
                1, net_pnl_col, len(trade_df), net_pnl_col,
                {'type': 'cell', 'criteria': '<', 'value': 0, 'format': currency_negative}
            )
            
            # Freeze the header row
            worksheet_trades.freeze_panes(1, 0)
        else:
            # Create empty sheet with headers if no trades
            empty_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
            empty_df.to_excel(writer, sheet_name='Trade_Log', index=False)
            
            worksheet_trades = writer.sheets['Trade_Log']
            # Still set column widths and freeze header for empty sheet
            for col_range, width in col_widths:
                worksheet_trades.set_column(col_range, width)
            worksheet_trades.freeze_panes(1, 0)

        # =========================================================================
        # Sheet 4: Daily_Trade_Log
        # =========================================================================
        daily_df = _normalize_daily_log(daily_activity_log if 'daily_activity_log' in locals() else [])
        
        # Verify we have exactly 11 columns
        assert daily_df.shape[1] == 11, f"Daily_Trade_Log should have 11 columns, got {daily_df.shape[1]}"
        logger.info(f"Daily_Trade_Log shape: {daily_df.shape}")
        
        # Set column widths for all 11 columns (defined outside the if block for reuse)
        col_widths_daily = [
            ('A:A', 12),   # date
            ('B:B', 15),   # ticker
            ('C:C', 14),   # action
            ('D:D', 12),   # price
            ('E:E', 10),   # quantity
            ('F:F', 14),   # position_value
            ('G:G', 14),   # cash_balance
            ('H:H', 16),   # total_portfolio_value
            ('I:I', 10),   # score
            ('J:J', 10),   # signal
            ('K:K', 30),   # notes
        ]
        
        if len(daily_df) > 0:
            daily_df.to_excel(writer, sheet_name='Daily_Trade_Log', index=False, header=True)

            worksheet_daily = writer.sheets['Daily_Trade_Log']
            
            for col_range, width in col_widths_daily:
                worksheet_daily.set_column(col_range, width)

            # Get column indices for formatting
            price_col = daily_df.columns.get_loc('price')
            position_value_col = daily_df.columns.get_loc('position_value')
            cash_balance_col = daily_df.columns.get_loc('cash_balance')
            total_portfolio_value_col = daily_df.columns.get_loc('total_portfolio_value')
            action_col = daily_df.columns.get_loc('action')

            # Apply number formatting to numeric columns (written at the
            # looked-up column index, not a hardcoded one)
            numeric_cols = (price_col, position_value_col, cash_balance_col,
                            total_portfolio_value_col)
            for row_idx in range(1, len(daily_df) + 1):
                row = daily_df.iloc[row_idx - 1]
                for col in numeric_cols:
                    value = row.iloc[col]
                    if pd.notna(value):
                        worksheet_daily.write_number(row_idx, col, value, number_format)

            # Conditional formatting on the action column. The sheet is written
            # with index=False, so the DataFrame position IS the worksheet
            # column — the old `action_col + 1` painted the price column
            # instead, leaving the actions themselves unformatted.
            for match, fmt_spec in (
                ('BUY', {'font_color': 'green', 'bold': True}),
                ('SELL', {'font_color': 'red', 'bold': True}),
                ('STOP_LOSS', {'bg_color': '#FFC7CE', 'font_color': '#9C0006'}),
                ('TARGET', {'bg_color': '#C6EFCE', 'font_color': '#006100'}),
            ):
                worksheet_daily.conditional_format(
                    1, action_col, len(daily_df), action_col,
                    {'type': 'text', 'criteria': 'containing', 'value': match,
                     'format': workbook.add_format(fmt_spec)}
                )
            
            # Freeze the header row
            worksheet_daily.freeze_panes(1, 0)
            
            # Add autofilter on header row
            worksheet_daily.autofilter(0, 0, len(daily_df), len(daily_df.columns) - 1)
        else:
            # Create empty sheet with headers if no activity
            empty_df = pd.DataFrame(columns=EXPECTED_DAILY_COLUMNS)
            empty_df.to_excel(writer, sheet_name='Daily_Trade_Log', index=False)
            
            worksheet_daily = writer.sheets['Daily_Trade_Log']
            # Still set column widths and freeze header for empty sheet
            for col_range, width in col_widths_daily:
                worksheet_daily.set_column(col_range, width)
            worksheet_daily.freeze_panes(1, 0)
            worksheet_daily.autofilter(0, 0, 0, len(empty_df.columns) - 1)

        # =========================================================================
        # Sheet 5: Monthly_Heatmap
        # =========================================================================
        monthly_df = _create_monthly_heatmap_df(equity_curve)
        monthly_df.to_excel(writer, sheet_name='Monthly_Heatmap', index=True, header=True)

        worksheet_heatmap = writer.sheets['Monthly_Heatmap']
        worksheet_heatmap.set_column('A:A', 8)   # Year
        for col in range(1, 13):  # Jan-Dec
            worksheet_heatmap.set_column(col, col, 10)

        # Apply color scale conditional formatting
        worksheet_heatmap.conditional_format(
            1, 1, len(monthly_df), 12,
            {
                'type': '3_color_scale',
                'min_color': '#FF0000',      # Red for negative
                'mid_color': '#FFFF00',      # Yellow for neutral
                'max_color': '#00B050'       # Green for positive
            }
        )

        # Format as percentages. _create_monthly_heatmap_df returns monthly
        # returns in PERCENT units, so every cell is divided by 100 — the old
        # magnitude guess turned a quiet +0.4% month into +40%.
        for row_idx in range(1, len(monthly_df) + 1):
            for col_idx in range(1, 13):
                val = monthly_df.iloc[row_idx - 1, col_idx - 1] if col_idx - 1 < len(monthly_df.columns) else None
                if val is not None and pd.notna(val):
                    worksheet_heatmap.write_number(row_idx, col_idx, val / 100.0, percent_format)

        # =========================================================================
        # Sheet 5: Brain_Evolution
        # =========================================================================
        brain_df = _prepare_brain_evolution_df(brain_evolution)
        brain_df.to_excel(writer, sheet_name='Brain_Evolution', index=False, header=True)

        worksheet_brain = writer.sheets['Brain_Evolution']
        worksheet_brain.set_column('A:A', 12)  # Trading Day
        worksheet_brain.set_column('B:B', 10)  # Date
        worksheet_brain.set_column('C:C', 10)  # Trend
        worksheet_brain.set_column('D:D', 12)  # Breakout
        worksheet_brain.set_column('E:E', 10)  # Volume
        worksheet_brain.set_column('F:F', 12)  # MC_Prob

        # Create chart for brain evolution (only when there is data to plot —
        # xlsxwriter refuses to save a workbook containing a seriesless chart)
        colors = ['#4472C4', '#ED7D31', '#70AD47', '#264478']
        weight_cols = ['Trend', 'Breakout', 'Volume', 'MC_Prob']
        plottable = [c for c in weight_cols if c in brain_df.columns]

        if len(brain_df) > 0 and plottable:
            chart_brain = workbook.add_chart({'type': 'line'})
            chart_brain.set_title({'name': 'Agent Weights Evolution Over Time'})
            chart_brain.set_x_axis({'name': 'Trading Day'})
            chart_brain.set_y_axis({'name': 'Weight (%)'})

            for idx, col in enumerate(weight_cols):
                if col not in brain_df.columns:
                    continue
                letter = chr(ord("A") + list(brain_df.columns).index(col))
                chart_brain.add_series({
                    'name': col,
                    'categories': f'=Brain_Evolution!$A$2:$A${len(brain_df) + 1}',
                    'values': f'=Brain_Evolution!${letter}$2:${letter}${len(brain_df) + 1}',
                    'line': {'color': colors[idx], 'width': 2}
                })

            worksheet_brain.insert_chart('H2', chart_brain)

        # =========================================================================
        # Sheet 6: Monte_Carlo_Simulations
        # =========================================================================
        mc_df = _create_monte_carlo_df(analytics)
        mc_df.to_excel(writer, sheet_name='Monte_Carlo_Simulations', index=False, header=True)

        worksheet_mc = writer.sheets['Monte_Carlo_Simulations']
        worksheet_mc.set_column('A:A', 20)
        worksheet_mc.set_column('B:B', 18)

        # Apply currency formatting to terminal wealth values
        for row_idx in range(1, len(mc_df) + 1):
            val = mc_df.iloc[row_idx - 1, 1]
            if pd.notna(val):
                worksheet_mc.write_number(row_idx, 1, val, currency_format)

        # =========================================================================
        # Sheet 7: Parallel_Execution_Summary (only if parallel_metrics provided)
        # =========================================================================
        if parallel_metrics:
            _create_parallel_execution_sheet(workbook, writer, parallel_metrics, header_format, number_format)

    logger.info(f"Backtest report exported to: {filepath}")
    return filepath


def _create_parallel_execution_sheet(
    workbook,
    writer: pd.ExcelWriter,
    parallel_metrics: Dict[str, Any],
    header_format,
    number_format
) -> None:
    """
    Create Parallel Execution Summary sheet showing execution metrics.
    
    Args:
        workbook: xlsxwriter workbook object.
        writer: pandas ExcelWriter object.
        parallel_metrics: Dictionary with parallel execution info.
        header_format: Format for header cells.
        number_format: Format for numeric cells.
    """
    worksheet = workbook.add_worksheet('Parallel_Execution_Summary')
    
    # Define additional formats
    success_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    warning_format = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
    error_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    
    # Header row
    worksheet.write('A1', 'Parallel Execution Summary', header_format)
    worksheet.merge_range('A1:B1', 'Parallel Execution Summary', header_format)
    
    # Metrics data
    metrics_data = [
        ('Execution Time (seconds)', parallel_metrics.get('execution_time_seconds', 0)),
        ('Number of Workers', parallel_metrics.get('num_workers', 0)),
        ('Success Rate (%)', parallel_metrics.get('success_rate', 0)),
        ('Total Tickers Processed', parallel_metrics.get('total_tickers', 0)),
        ('Failed Tickers', parallel_metrics.get('failed_tickers', 0)),
        ('Worker Type', parallel_metrics.get('worker_type', 'N/A')),
    ]
    
    # Write metrics
    for row_idx, (metric_name, value) in enumerate(metrics_data, start=1):
        worksheet.write(row_idx, 0, metric_name)
        
        if isinstance(value, (int, float)):
            if metric_name == 'Success Rate (%)':
                # Apply color based on success rate
                if value >= 95:
                    fmt = success_format
                elif value >= 80:
                    fmt = warning_format
                else:
                    fmt = error_format
                worksheet.write_number(row_idx, 1, value, fmt)
            elif metric_name == 'Failed Tickers' and value > 0:
                worksheet.write_number(row_idx, 1, value, error_format)
            else:
                worksheet.write_number(row_idx, 1, value, number_format)
        else:
            worksheet.write(row_idx, 1, str(value))
    
    # Auto-fit columns
    worksheet.set_column('A:A', 30)
    worksheet.set_column('B:B', 15)
    
    # Add worker breakdown if available
    if 'worker_details' in parallel_metrics:
        details = parallel_metrics['worker_details']
        start_row = len(metrics_data) + 3
        
        worksheet.write(start_row, 0, 'Worker Details', header_format)
        
        if isinstance(details, dict):
            for idx, (key, value) in enumerate(details.items()):
                worksheet.write(start_row + 1 + idx, 0, f'  {key}')
                worksheet.write(start_row + 1 + idx, 1, value)


def _create_executive_summary_df(analytics: Dict[str, Any]) -> pd.DataFrame:
    """Create Executive Summary DataFrame (see SUMMARY_METRICS for units)."""
    metrics = [(label, analytics.get(key, 0)) for label, key, _ in SUMMARY_METRICS]
    return pd.DataFrame(metrics, columns=['Metric', 'Value'])


def _apply_conditional_formatting_summary(worksheet, df: pd.DataFrame, workbook) -> None:
    """Apply conditional formatting for Sharpe and MaxDD."""
    green_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    red_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

    # Find row indices for Sharpe and MaxDD
    sharpe_row = None
    maxdd_row = None

    for idx, row in df.iterrows():
        if 'Sharpe' in row['Metric']:
            sharpe_row = idx + 1  # +1 because Excel is 1-indexed and we have header
        if 'Drawdown' in row['Metric'] or 'MaxDD' in row['Metric']:
            maxdd_row = idx + 1

    # Apply conditional formatting for Sharpe > 1 (Green)
    if sharpe_row is not None:
        worksheet.conditional_format(
            sharpe_row, 1, sharpe_row, 1,
            {'type': 'cell', 'criteria': '>', 'value': 1, 'format': green_format}
        )

    # Apply conditional formatting for MaxDD > 20% (Red)
    if maxdd_row is not None:
        worksheet.conditional_format(
            maxdd_row, 1, maxdd_row, 1,
            {'type': 'cell', 'criteria': '<', 'value': -0.20, 'format': red_format}
        )


def _prepare_equity_curve_df(equity_curve: pd.Series) -> pd.DataFrame:
    """Prepare Equity Curve DataFrame for export.

    Indexed by date (rather than carrying Date as a column alongside a
    meaningless RangeIndex), so writing it with index=True produces
    Date | Portfolio_Value | [Benchmark] | Drawdown_% and the sheet's charts
    line up with the columns they are supposed to plot.
    """
    df = pd.DataFrame(index=pd.Index(equity_curve.index, name='Date'))

    # Portfolio Value
    df['Portfolio_Value'] = equity_curve.values

    # Check if benchmark data is included
    if hasattr(equity_curve, 'attrs') and 'benchmark' in equity_curve.attrs:
        df['Benchmark'] = equity_curve.attrs['benchmark']
    elif isinstance(equity_curve, pd.DataFrame):
        if 'benchmark' in equity_curve.columns:
            df['Benchmark'] = equity_curve['benchmark'].values

    # Calculate Drawdown %
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    df['Drawdown_%'] = drawdown.values

    return df


# NOTE: an unused _prepare_trade_log_df() lived here, producing a *second*,
# display-name trade-log schema ('Entry Date', 'Qty', 'STT/Taxes', ...) that
# nothing exported. It was deleted: _normalize_trade_log() + EXPECTED_COLUMNS
# is the single Trade_Log schema, and the stale duplicate was what tests were
# still asserting against.


def _create_monthly_heatmap_df(equity_curve: pd.Series) -> pd.DataFrame:
    """Create Monthly Heatmap DataFrame showing returns by month and year."""
    # Get monthly returns
    monthly_returns = equity_curve.resample('ME').last().pct_change() * 100

    # Create pivot table: Years as rows, Months as columns
    years = monthly_returns.index.year.unique()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    heatmap_data = {}
    for i, month in enumerate(months, 1):
        month_data = []
        for year in years:
            mask = (monthly_returns.index.year == year) & (monthly_returns.index.month == i)
            if mask.any():
                month_data.append(monthly_returns[mask].iloc[0])
            else:
                month_data.append(np.nan)
        heatmap_data[month] = month_data

    df = pd.DataFrame(heatmap_data, index=years)
    df.index.name = 'Year'

    return df


BRAIN_COLUMNS = ['Trading Day', 'Date', 'Trend', 'Breakout', 'Volume', 'MC_Prob']


def _prepare_brain_evolution_df(brain_evolution: List[Dict[str, Any]]) -> pd.DataFrame:
    """Prepare Brain Evolution DataFrame for export.

    Always returns the full column set, even with no snapshots — an empty,
    column-less frame produced a chart with no series, and xlsxwriter refuses
    to save a workbook containing one, so the whole export died at close().
    """
    if not brain_evolution:
        return pd.DataFrame(columns=BRAIN_COLUMNS)

    records = []
    for entry in brain_evolution:
        record = {
            'Trading_Day': entry.get('trading_day', 0),
            'Date': entry.get('date', ''),
            'Trend': entry.get('weights', {}).get('Trend', 0),
            'Breakout': entry.get('weights', {}).get('Breakout', 0),
            'Volume': entry.get('weights', {}).get('Volume', 0),
            'MC_Prob': entry.get('weights', {}).get('MC_Prob', 0)
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Rename columns for display
    column_mapping = {
        'Trading_Day': 'Trading Day',
        'Date': 'Date',
        'Trend': 'Trend',
        'Breakout': 'Breakout',
        'Volume': 'Volume',
        'MC_Prob': 'MC_Prob'
    }

    df = df.rename(columns=column_mapping)
    return df


def _create_monte_carlo_df(analytics: Dict[str, Any]) -> pd.DataFrame:
    """Create Monte Carlo Simulations DataFrame."""
    mc_results = analytics.get('monte_carlo_results', {})

    # Default values if not provided
    p5 = mc_results.get('percentile_5', mc_results.get('p5', 0))
    p50 = mc_results.get('percentile_50', mc_results.get('p50', 0))
    p95 = mc_results.get('percentile_95', mc_results.get('p95', 0))

    metrics = [
        ('5th Percentile Terminal Wealth', p5),
        ('50th Percentile (Median) Terminal Wealth', p50),
        ('95th Percentile Terminal Wealth', p95)
    ]

    return pd.DataFrame(metrics, columns=['Metric', 'Terminal Wealth (₹)'])

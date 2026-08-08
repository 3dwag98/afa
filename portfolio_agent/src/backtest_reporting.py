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
    filepath: str
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

    Returns:
        str: Path to the created Excel file.
    """
    # Ensure output directory exists
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create Excel writer with xlsxwriter engine
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

        # Apply formatting to data rows
        for row_idx in range(1, len(summary_df) + 1):
            worksheet_summary.write(row_idx, 0, summary_df.iloc[row_idx - 1, 0])
            value = summary_df.iloc[row_idx - 1, 1]

            # Determine format based on metric type
            if isinstance(value, (int, float)):
                if 'Sharpe' in summary_df.iloc[row_idx - 1, 0]:
                    fmt = percent_positive if value > 0 else percent_negative
                    worksheet_summary.write_number(row_idx, 1, value, fmt)
                elif 'Drawdown' in summary_df.iloc[row_idx - 1, 0] or '%' in summary_df.iloc[row_idx - 1, 0]:
                    fmt = percent_negative if value < -0.20 else percent_format
                    worksheet_summary.write_number(row_idx, 1, value / 100 if abs(value) > 1 else value, fmt)
                elif 'Ruin' in summary_df.iloc[row_idx - 1, 0] or 'Probability' in summary_df.iloc[row_idx - 1, 0]:
                    worksheet_summary.write_number(row_idx, 1, value, percent_format)
                elif 'CAGR' in summary_df.iloc[row_idx - 1, 0] or 'Return' in summary_df.iloc[row_idx - 1, 0]:
                    worksheet_summary.write_number(row_idx, 1, value / 100 if abs(value) > 1 else value, percent_format)
                else:
                    worksheet_summary.write_number(row_idx, 1, value, number_format)
            else:
                worksheet_summary.write(row_idx, 1, str(value))

        # Conditional formatting for Sharpe and MaxDD
        _apply_conditional_formatting_summary(worksheet_summary, summary_df, workbook)

        # =========================================================================
        # Sheet 2: Equity_Curve
        # =========================================================================
        equity_df = _prepare_equity_curve_df(equity_curve)
        equity_df.to_excel(writer, sheet_name='Equity_Curve', index=True, header=True)

        worksheet_equity = writer.sheets['Equity_Curve']
        worksheet_equity.set_column('A:A', 12)  # Date
        worksheet_equity.set_column('B:B', 18)  # Portfolio Value
        worksheet_equity.set_column('C:C', 18)  # Benchmark (if exists)
        worksheet_equity.set_column('D:D', 12)  # Drawdown %

        # Format header
        for col_num, col_name in enumerate(equity_df.columns):
            worksheet_equity.write(0, col_num + 1, col_name, header_format)
        worksheet_equity.write(0, 0, 'Date', header_format)

        # Create chart for Equity Curve
        chart_equity = workbook.add_chart({'type': 'line'})
        chart_equity.set_title({'name': 'Portfolio Value vs Benchmark'})
        chart_equity.set_x_axis({'name': 'Date'})
        chart_equity.set_y_axis({'name': 'Portfolio Value (₹)'})

        # Add portfolio value series
        chart_equity.add_series({
            'name': 'Portfolio Value',
            'categories': f'=Equity_Curve!$A$2:$A${len(equity_df) + 1}',
            'values': f'=Equity_Curve!$B$2:$B${len(equity_df) + 1}',
            'line': {'color': '#2F5597', 'width': 2}
        })

        # Add benchmark series if available
        if 'Benchmark' in equity_df.columns:
            chart_equity.add_series({
                'name': 'Nifty 50 Benchmark',
                'categories': f'=Equity_Curve!$A$2:$A${len(equity_df) + 1}',
                'values': f'=Equity_Curve!$C$2:$C${len(equity_df) + 1}',
                'line': {'color': '#ED7D31', 'width': 2, 'dash_type': 'dash'}
            })

        worksheet_equity.insert_chart('E2', chart_equity)

        # Create secondary axis chart for Drawdown
        chart_dd = workbook.add_chart({'type': 'line'})
        chart_dd.set_title({'name': 'Drawdown %'})
        chart_dd.set_y_axis({'name': 'Drawdown %', 'position': 'right'})

        chart_dd.add_series({
            'name': 'Drawdown %',
            'categories': f'=Equity_Curve!$A$2:$A${len(equity_df) + 1}',
            'values': f'=Equity_Curve!$D$2:$D${len(equity_df) + 1}',
            'line': {'color': '#C00000', 'width': 1.5}
        })

        worksheet_equity.insert_chart('E20', chart_dd)

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

            # Apply formatting to each row
            for row_idx in range(1, len(trade_df) + 1):
                # Entry Price (column D)
                val_entry = trade_df.iloc[row_idx - 1, entry_price_col]
                if pd.notna(val_entry):
                    worksheet_trades.write_number(row_idx, 3, val_entry, currency_format)

                # Exit Price (column E)
                val_exit = trade_df.iloc[row_idx - 1, exit_price_col]
                if pd.notna(val_exit):
                    worksheet_trades.write_number(row_idx, 4, val_exit, currency_format)

                # Gross PnL (column J)
                val_gross = trade_df.iloc[row_idx - 1, gross_pnl_col]
                if pd.notna(val_gross):
                    fmt = currency_positive if val_gross >= 0 else currency_negative
                    worksheet_trades.write_number(row_idx, 9, val_gross, fmt)

                # Transaction Costs (column K)
                val_txn = trade_df.iloc[row_idx - 1, transaction_costs_col]
                if pd.notna(val_txn):
                    worksheet_trades.write_number(row_idx, 10, val_txn, currency_format)

                # Taxes (column L)
                val_taxes = trade_df.iloc[row_idx - 1, taxes_col]
                if pd.notna(val_taxes):
                    worksheet_trades.write_number(row_idx, 11, val_taxes, currency_format)

                # Net PnL (column M)
                val_net = trade_df.iloc[row_idx - 1, net_pnl_col]
                if pd.notna(val_net):
                    fmt = currency_positive if val_net >= 0 else currency_negative
                    worksheet_trades.write_number(row_idx, 12, val_net, fmt)

                # Return Pct (column N) - stored as decimal (e.g., 0.05 for 5%)
                val_return = trade_df.iloc[row_idx - 1, return_pct_col]
                if pd.notna(val_return):
                    # Ensure value is stored as decimal
                    return_val = val_return / 100 if abs(val_return) > 1 else val_return
                    fmt = percent_positive if return_val >= 0 else percent_negative
                    worksheet_trades.write_number(row_idx, 13, return_val, fmt)

            # Apply conditional formatting for Net PnL column (M)
            worksheet_trades.conditional_format(
                1, 12, len(trade_df), 12,
                {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': currency_positive}
            )
            worksheet_trades.conditional_format(
                1, 12, len(trade_df), 12,
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

            # Apply number formatting to numeric columns
            for row_idx in range(1, len(daily_df) + 1):
                # Price (column D)
                val_price = daily_df.iloc[row_idx - 1, price_col]
                if pd.notna(val_price):
                    worksheet_daily.write_number(row_idx, 3, val_price, number_format)

                # Position Value (column F)
                val_pos = daily_df.iloc[row_idx - 1, position_value_col]
                if pd.notna(val_pos):
                    worksheet_daily.write_number(row_idx, 5, val_pos, number_format)

                # Cash Balance (column G)
                val_cash = daily_df.iloc[row_idx - 1, cash_balance_col]
                if pd.notna(val_cash):
                    worksheet_daily.write_number(row_idx, 6, val_cash, number_format)

                # Total Portfolio Value (column H)
                val_total = daily_df.iloc[row_idx - 1, total_portfolio_value_col]
                if pd.notna(val_total):
                    worksheet_daily.write_number(row_idx, 7, val_total, number_format)

            # Conditional formatting on action column
            # "BUY" green font
            worksheet_daily.conditional_format(
                1, action_col + 1, len(daily_df), action_col + 1,
                {'type': 'text', 'criteria': 'containing', 'value': 'BUY', 
                 'format': workbook.add_format({'font_color': 'green', 'bold': True})}
            )
            # "SELL" red font
            worksheet_daily.conditional_format(
                1, action_col + 1, len(daily_df), action_col + 1,
                {'type': 'text', 'criteria': 'containing', 'value': 'SELL', 
                 'format': workbook.add_format({'font_color': 'red', 'bold': True})}
            )
            # "STOP_LOSS_HIT" red fill
            worksheet_daily.conditional_format(
                1, action_col + 1, len(daily_df), action_col + 1,
                {'type': 'text', 'criteria': 'containing', 'value': 'STOP_LOSS', 
                 'format': workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})}
            )
            # "TARGET_HIT" green fill
            worksheet_daily.conditional_format(
                1, action_col + 1, len(daily_df), action_col + 1,
                {'type': 'text', 'criteria': 'containing', 'value': 'TARGET', 
                 'format': workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})}
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

        # Format as percentages
        for row_idx in range(1, len(monthly_df) + 1):
            for col_idx in range(1, 13):
                val = monthly_df.iloc[row_idx - 1, col_idx - 1] if col_idx - 1 < len(monthly_df.columns) else None
                if val is not None and pd.notna(val):
                    worksheet_heatmap.write_number(row_idx, col_idx, val / 100 if abs(val) > 1 else val, percent_format)

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

        # Create chart for brain evolution
        chart_brain = workbook.add_chart({'type': 'line'})
        chart_brain.set_title({'name': 'Agent Weights Evolution Over Time'})
        chart_brain.set_x_axis({'name': 'Trading Day'})
        chart_brain.set_y_axis({'name': 'Weight (%)'})

        colors = ['#4472C4', '#ED7D31', '#70AD47', '#264478']
        weight_cols = ['Trend', 'Breakout', 'Volume', 'MC_Prob']

        for idx, col in enumerate(weight_cols):
            if col in brain_df.columns:
                chart_brain.add_series({
                    'name': col,
                    'categories': f'=Brain_Evolution!$A$2:$A${len(brain_df) + 1}',
                    'values': f'=Brain_Evolution!${chr(ord("C") + idx)}$2:${chr(ord("C") + idx)}${len(brain_df) + 1}',
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

    logger.info(f"Backtest report exported to: {filepath}")
    return filepath


def _create_executive_summary_df(analytics: Dict[str, Any]) -> pd.DataFrame:
    """Create Executive Summary DataFrame."""
    metrics = [
        ('CAGR', analytics.get('cagr', 0)),
        ('Sharpe Ratio', analytics.get('sharpe', 0)),
        ('Sortino Ratio', analytics.get('sortino', 0)),
        ('Max Drawdown (%)', analytics.get('max_drawdown', 0)),
        ('Profit Factor', analytics.get('profit_factor', 0)),
        ('Probability of Ruin (%)', analytics.get('probability_of_ruin', 0)),
        ('Total Return (%)', analytics.get('total_return', 0)),
        ('Annualized Volatility (%)', analytics.get('volatility', 0)),
        ('Win Rate (%)', analytics.get('win_rate', 0)),
        ('Total Trades', analytics.get('total_trades', 0)),
        ('Final Portfolio Value (₹)', analytics.get('final_portfolio_value', 0)),
        ('Initial Capital (₹)', analytics.get('initial_capital', 0))
    ]

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
    """Prepare Equity Curve DataFrame for export."""
    df = pd.DataFrame()
    df['Date'] = equity_curve.index

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


def _prepare_trade_log_df(trade_log: List[Dict[str, Any]]) -> pd.DataFrame:
    """Prepare Trade Log DataFrame for export."""
    if not trade_log:
        return pd.DataFrame()

    required_columns = [
        'entry_date', 'exit_date', 'ticker', 'side', 'entry_price', 'exit_price',
        'qty', 'gross_pnl', 'stt_taxes', 'slippage', 'net_pnl', 'holding_days', 'signal_trigger'
    ]

    # Normalize trade log entries
    normalized_trades = []
    for trade in trade_log:
        normalized = {}
        for col in required_columns:
            # Handle different naming conventions
            val = trade.get(col, trade.get(col.replace('_', ' '), ''))
            normalized[col] = val if val != '' else None
        normalized_trades.append(normalized)

    df = pd.DataFrame(normalized_trades)

    # Rename columns for display
    column_mapping = {
        'entry_date': 'Entry Date',
        'exit_date': 'Exit Date',
        'ticker': 'Ticker',
        'side': 'Side',
        'entry_price': 'Entry Price',
        'exit_price': 'Exit Price',
        'qty': 'Qty',
        'gross_pnl': 'Gross PnL',
        'stt_taxes': 'STT/Taxes',
        'slippage': 'Slippage',
        'net_pnl': 'Net PnL',
        'holding_days': 'Holding Days',
        'signal_trigger': 'Signal Trigger'
    }

    df = df.rename(columns=column_mapping)
    return df


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


def _prepare_brain_evolution_df(brain_evolution: List[Dict[str, Any]]) -> pd.DataFrame:
    """Prepare Brain Evolution DataFrame for export."""
    if not brain_evolution:
        return pd.DataFrame()

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

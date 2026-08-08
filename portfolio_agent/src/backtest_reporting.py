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


def export_backtest_excel(
    analytics: Dict[str, Any],
    equity_curve: pd.Series,
    trade_log: List[Dict[str, Any]],
    brain_evolution: List[Dict[str, Any]],
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
        trade_df = _prepare_trade_log_df(trade_log)
        if len(trade_df) > 0:
            trade_df.to_excel(writer, sheet_name='Trade_Log', index=False, header=True)

            worksheet_trades = writer.sheets['Trade_Log']
            worksheet_trades.set_column('A:A', 12)  # Entry Date
            worksheet_trades.set_column('B:B', 12)  # Exit Date
            worksheet_trades.set_column('C:C', 10)  # Ticker
            worksheet_trades.set_column('D:D', 8)   # Side
            worksheet_trades.set_column('E:E', 12)  # Entry Price
            worksheet_trades.set_column('F:F', 12)  # Exit Price
            worksheet_trades.set_column('G:G', 8)   # Qty
            worksheet_trades.set_column('H:H', 14)  # Gross PnL
            worksheet_trades.set_column('I:I', 14)  # STT/Taxes
            worksheet_trades.set_column('J:J', 12)  # Slippage
            worksheet_trades.set_column('K:K', 14)  # Net PnL
            worksheet_trades.set_column('L:L', 10)  # Holding Days
            worksheet_trades.set_column('M:M', 15)  # Signal Trigger

            # Get column indices for price and PnL columns (after renaming)
            entry_price_col = trade_df.columns.get_loc('Entry Price')
            exit_price_col = trade_df.columns.get_loc('Exit Price')
            gross_pnl_col = trade_df.columns.get_loc('Gross PnL')
            net_pnl_col = trade_df.columns.get_loc('Net PnL')

            # Apply currency formatting to price and PnL columns
            for row_idx in range(1, len(trade_df) + 1):
                # Entry Price (column E)
                val_entry = trade_df.iloc[row_idx - 1, entry_price_col]
                if pd.notna(val_entry):
                    worksheet_trades.write_number(row_idx, 4, val_entry, currency_format)

                # Exit Price (column F)
                val_exit = trade_df.iloc[row_idx - 1, exit_price_col]
                if pd.notna(val_exit):
                    worksheet_trades.write_number(row_idx, 5, val_exit, currency_format)

                # Gross PnL (column H)
                val_gross = trade_df.iloc[row_idx - 1, gross_pnl_col]
                if pd.notna(val_gross):
                    fmt = currency_positive if val_gross >= 0 else currency_negative
                    worksheet_trades.write_number(row_idx, 7, val_gross, fmt)

                # Net PnL (column K)
                val_net = trade_df.iloc[row_idx - 1, net_pnl_col]
                if pd.notna(val_net):
                    fmt = currency_positive if val_net >= 0 else currency_negative
                    worksheet_trades.write_number(row_idx, 10, val_net, fmt)

            # Apply conditional formatting for Net PnL column (K)
            worksheet_trades.conditional_format(
                1, 10, len(trade_df), 10,
                {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': currency_positive}
            )
            worksheet_trades.conditional_format(
                1, 10, len(trade_df), 10,
                {'type': 'cell', 'criteria': '<', 'value': 0, 'format': currency_negative}
            )
        else:
            # Create empty sheet with headers if no trades
            empty_df = pd.DataFrame(columns=[
                'Entry Date', 'Exit Date', 'Ticker', 'Side', 'Entry Price', 'Exit Price',
                'Qty', 'Gross PnL', 'STT/Taxes', 'Slippage', 'Net PnL', 'Holding Days', 'Signal Trigger'
            ])
            empty_df.to_excel(writer, sheet_name='Trade_Log', index=False)

        # =========================================================================
        # Sheet 4: Monthly_Heatmap
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

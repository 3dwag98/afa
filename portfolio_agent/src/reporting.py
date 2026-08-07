"""Excel reporting module."""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.config import AppConfig
from src.models import Recommendation, IndicatorSnapshot, AgentBrain
from src.monte_carlo import MonteCarloResult


def export_excel_report(
    config: AppConfig,
    brain: AgentBrain,
    recommendations: list[Recommendation],
    indicators: list[IndicatorSnapshot],
    mc_results: list[MonteCarloResult],
    run_id: str
) -> str:
    """Generate a professional Excel workbook using xlsxwriter.

    Args:
        config: Application configuration.
        brain: Agent brain state with weights and history.
        recommendations: List of trade recommendations.
        indicators: List of indicator snapshots.
        mc_results: List of Monte Carlo simulation results.
        run_id: Unique run identifier.

    Returns:
        Path to created Excel file.
    """
    output_path = config.excel_output
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1,
            'align': 'center'
        })

        currency_format = workbook.add_format({
            'num_format': '₹#,##0'
        })

        percent_format = workbook.add_format({
            'num_format': '0.0%'
        })

        decimal_format = workbook.add_format({
            'num_format': '0.0000'
        })

        buy_format = workbook.add_format({
            'bg_color': '#C6EFCE',  # Green
            'border': 1
        })

        avoid_format = workbook.add_format({
            'bg_color': '#FFC7CE',  # Red
            'border': 1
        })

        watch_format = workbook.add_format({
            'bg_color': '#FFEB9C',  # Yellow
            'border': 1
        })

        # === Sheet 1: Summary ===
        _create_summary_sheet(
            writer, workbook, header_format, config, 
            recommendations, run_id, brain
        )

        # === Sheet 2: Live_Recommendations ===
        _create_recommendations_sheet(
            writer, workbook, header_format, currency_format,
            percent_format, recommendations, buy_format, avoid_format, watch_format
        )

        # === Sheet 3: Indicators ===
        _create_indicators_sheet(
            writer, workbook, header_format, indicators
        )

        # === Sheet 4: Monte_Carlo ===
        _create_monte_carlo_sheet(
            writer, workbook, header_format, percent_format, mc_results
        )

        # === Sheet 5: Agent_Brain_Weights ===
        _create_brain_weights_sheet(
            writer, workbook, header_format, brain
        )

        # === Sheet 6: Learning_History ===
        _create_learning_history_sheet(
            writer, workbook, header_format, brain
        )

        # === Sheet 7: Trade_Memory ===
        _create_trade_memory_sheet(
            writer, workbook, header_format, brain
        )

    return output_path


def _create_summary_sheet(
    writer: pd.ExcelWriter,
    workbook: Any,
    header_format: Any,
    config: AppConfig,
    recommendations: list[Recommendation],
    run_id: str,
    brain: AgentBrain
) -> None:
    """Create Summary sheet."""
    worksheet = workbook.add_worksheet('Summary')
    
    # Count signals
    num_recommendations = len(recommendations)
    num_buy = sum(1 for r in recommendations if r.signal == 'BUY')
    num_watch = sum(1 for r in recommendations if r.signal == 'WATCH')
    num_avoid = sum(1 for r in recommendations if r.signal == 'AVOID')
    
    # Build summary data
    summary_data = [
        ['Run ID', run_id],
        ['Generated at', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Portfolio value', f'₹{config.portfolio_value_inr:,.0f}'],
        ['Paper trading mode', config.paper_trading_mode],
        ['Number of recommendations', num_recommendations],
        ['Number of BUY signals', num_buy],
        ['Number of WATCH signals', num_watch],
        ['Number of AVOID signals', num_avoid],
        ['', ''],
        ['Brain weights summary', ''],
    ]
    
    # Add brain weights
    for factor, weight in brain.weights.items():
        summary_data.append([f'  {factor}', f'{weight}%'])
    
    # Write data
    for row_idx, row_data in enumerate(summary_data):
        for col_idx, cell_value in enumerate(row_data):
            if row_idx == 0:
                worksheet.write(row_idx, col_idx, cell_value, header_format)
            else:
                worksheet.write(row_idx, col_idx, cell_value)
    
    # Auto-fit columns
    worksheet.set_column(0, 0, 25)
    worksheet.set_column(1, 1, 20)


def _create_recommendations_sheet(
    writer: pd.ExcelWriter,
    workbook: Any,
    header_format: Any,
    currency_format: Any,
    percent_format: Any,
    recommendations: list[Recommendation],
    buy_format: Any,
    avoid_format: Any,
    watch_format: Any
) -> None:
    """Create Live_Recommendations sheet."""
    worksheet = workbook.add_worksheet('Live_Recommendations')
    
    # Column headers
    columns = [
        'Symbol', 'Signal', 'Score', 'Trigger', 'MC Prob Profit',
        'Entry', 'Stop', 'Target', 'Reward/Risk', 'Qty',
        'Investment INR', 'Max Loss INR', 'Compliance', 'Rationale'
    ]
    
    # Write headers
    for col_idx, col_name in enumerate(columns):
        worksheet.write(0, col_idx, col_name, header_format)
    
    # Currency column indices (0-based)
    currency_cols = [5, 6, 7, 10, 11]  # Entry, Stop, Target, Investment INR, Max Loss INR
    prob_col = 4  # MC Prob Profit
    
    # Write data rows
    for row_idx, rec in enumerate(recommendations, start=1):
        row_data = [
            rec.symbol,
            rec.signal,
            rec.score,
            rec.trigger,
            rec.mc_probability_profit,
            rec.entry_price,
            rec.stop_price,
            rec.target_price,
            rec.reward_risk,
            rec.quantity,
            rec.investment_inr,
            rec.max_loss_inr,
            rec.compliance_status,
            rec.rationale
        ]
        
        # Determine signal format
        signal_upper = rec.signal.upper() if rec.signal else ''
        if signal_upper == 'BUY':
            cell_format = buy_format
        elif signal_upper == 'AVOID':
            cell_format = avoid_format
        elif signal_upper == 'WATCH':
            cell_format = watch_format
        else:
            cell_format = None
        
        for col_idx, value in enumerate(row_data):
            if col_idx in currency_cols:
                worksheet.write_number(row_idx, col_idx, value, currency_format)
            elif col_idx == prob_col:
                worksheet.write_number(row_idx, col_idx, value, percent_format)
            else:
                worksheet.write(row_idx, col_idx, value, cell_format)
    
    # Auto-fit columns
    for i in range(len(columns)):
        worksheet.set_column(i, i, 15)
    worksheet.set_column(13, 13, 40)  # Rationale column wider


def _create_indicators_sheet(
    writer: pd.ExcelWriter,
    workbook: Any,
    header_format: Any,
    indicators: list[IndicatorSnapshot]
) -> None:
    """Create Indicators sheet."""
    worksheet = workbook.add_worksheet('Indicators')
    
    # Column headers
    columns = [
        'Symbol', 'Date', 'Close', 'SMA20', 'SMA50', 'SMA200',
        'Donchian Upper 20', 'Prev Donchian Upper', 'Volume Ratio', 'ATR14'
    ]
    
    # Write headers
    for col_idx, col_name in enumerate(columns):
        worksheet.write(0, col_idx, col_name, header_format)
    
    # Write data rows
    for row_idx, ind in enumerate(indicators, start=1):
        row_data = [
            ind.symbol,
            ind.date.strftime('%Y-%m-%d') if hasattr(ind, 'date') and ind.date else '',
            getattr(ind, 'close', None),
            ind.sma20,
            ind.sma50,
            ind.sma200,
            ind.donchian_upper_20,
            ind.prev_donchian_upper_20,
            ind.volume_ratio,
            ind.atr14
        ]
        
        for col_idx, value in enumerate(row_data):
            worksheet.write(row_idx, col_idx, value)
    
    # Auto-fit columns
    for i in range(len(columns)):
        worksheet.set_column(i, i, 15)


def _create_monte_carlo_sheet(
    writer: pd.ExcelWriter,
    workbook: Any,
    header_format: Any,
    percent_format: Any,
    mc_results: list[MonteCarloResult]
) -> None:
    """Create Monte_Carlo sheet."""
    worksheet = workbook.add_worksheet('Monte_Carlo')
    
    # Column headers
    columns = [
        'Symbol', 'Horizon Days', 'Simulations', 'Probability Profit',
        'Expected Return %', 'VaR 95 %', 'CVaR 95 %'
    ]
    
    # Write headers
    for col_idx, col_name in enumerate(columns):
        worksheet.write(0, col_idx, col_name, header_format)
    
    # Write data rows
    for row_idx, mc in enumerate(mc_results, start=1):
        row_data = [
            getattr(mc, 'symbol', ''),
            mc.horizon_days,
            mc.simulations_count,
            mc.probability_profit,
            mc.expected_return_pct,
            mc.var_95,
            mc.cvar_95
        ]
        
        for col_idx, value in enumerate(row_data):
            if col_idx >= 3:  # Probability Profit and beyond are percentages
                worksheet.write_number(row_idx, col_idx, value, percent_format)
            else:
                worksheet.write(row_idx, col_idx, value)
    
    # Auto-fit columns
    for i in range(len(columns)):
        worksheet.set_column(i, i, 18)


def _create_brain_weights_sheet(
    writer: pd.ExcelWriter,
    workbook: Any,
    header_format: Any,
    brain: AgentBrain
) -> None:
    """Create Agent_Brain_Weights sheet with data bars."""
    worksheet = workbook.add_worksheet('Agent_Brain_Weights')
    
    # Column headers
    columns = ['Signal Factor', 'Learned Weight %']
    
    # Write headers
    for col_idx, col_name in enumerate(columns):
        worksheet.write(0, col_idx, col_name, header_format)
    
    # Write data rows
    row_idx = 1
    for factor, weight in brain.weights.items():
        worksheet.write(row_idx, 0, factor)
        worksheet.write_number(row_idx, 1, weight)
        row_idx += 1
    
    # Add data bars using conditional formatting
    if brain.weights:
        max_weight = max(brain.weights.values())
        min_weight = min(brain.weights.values())
        
        # Apply data bar conditional formatting to weight column
        worksheet.conditional_format(
            1, 1, len(brain.weights), 1,
            {
                'type': 'data_bar',
                'min_type': 'min',
                'max_type': 'max',
                'bar_color': '#638EC6',
                'bar_border_color': '#638EC6',
                'bar_solid': True,
            }
        )
    
    # Auto-fit columns
    worksheet.set_column(0, 0, 20)
    worksheet.set_column(1, 1, 15)


def _create_learning_history_sheet(
    writer: pd.ExcelWriter,
    workbook: Any,
    header_format: Any,
    brain: AgentBrain
) -> None:
    """Create Learning_History sheet."""
    worksheet = workbook.add_worksheet('Learning_History')
    
    # Column header
    columns = ['Log Entry']
    
    # Write headers
    for col_idx, col_name in enumerate(columns):
        worksheet.write(0, col_idx, col_name, header_format)
    
    # Write data rows from learning_log
    for row_idx, log_entry in enumerate(brain.learning_log, start=1):
        if isinstance(log_entry, dict):
            entry_str = str(log_entry)
        else:
            entry_str = str(log_entry)
        worksheet.write(row_idx, 0, entry_str)
    
    # Auto-fit columns
    worksheet.set_column(0, 0, 80)


def _create_trade_memory_sheet(
    writer: pd.ExcelWriter,
    workbook: Any,
    header_format: Any,
    brain: AgentBrain
) -> None:
    """Create Trade_Memory sheet from trade_history."""
    worksheet = workbook.add_worksheet('Trade_Memory')
    
    # Column headers from trade_history structure
    columns = [
        'trade_id', 'symbol', 'signal_trigger', 'entry_date', 'entry_price',
        'exit_date', 'exit_price', 'outcome', 'return_pct', 'outcome_source'
    ]
    
    # Write headers
    for col_idx, col_name in enumerate(columns):
        worksheet.write(0, col_idx, col_name, header_format)
    
    # Write data rows from trade_history
    for row_idx, trade in enumerate(brain.trade_history, start=1):
        row_data = [
            trade.get('trade_id', ''),
            trade.get('symbol', ''),
            trade.get('signal_trigger', ''),
            trade.get('entry_date', ''),
            trade.get('entry_price', ''),
            trade.get('exit_date', ''),
            trade.get('exit_price', ''),
            trade.get('outcome', ''),
            trade.get('return_pct', ''),
            trade.get('outcome_source', '')
        ]
        
        for col_idx, value in enumerate(row_data):
            worksheet.write(row_idx, col_idx, value)
    
    # Auto-fit columns
    for i in range(len(columns)):
        worksheet.set_column(i, i, 15)


# Legacy function kept for backward compatibility
def create_excel_report(recommendations: List[Dict[str, Any]],
                        simulations: Dict[str, Dict[str, Any]],
                        portfolio_summary: Dict[str, Any],
                        compliance_results: List[Dict[str, Any]],
                        learning_stats: Dict[str, Any],
                        output_path: str) -> str:
    """Create comprehensive Excel report (legacy function).

    Args:
        recommendations: List of recommendation dictionaries.
        simulations: Monte Carlo simulation results by ticker.
        portfolio_summary: Portfolio summary metrics.
        compliance_results: Compliance check results.
        learning_stats: Agent learning statistics.
        output_path: Path for output Excel file.

    Returns:
        Path to created file.
    """
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Create formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })

        currency_format = workbook.add_format({
            'num_format': '₹#,##0.00'
        })

        percent_format = workbook.add_format({
            'num_format': '0.00%'
        })

        decimal_format = workbook.add_format({
            'num_format': '0.0000'
        })

        # Sheet 1: Recommendations
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
            _format_sheet(writer.sheets['Recommendations'], header_format)

        # Sheet 2: Monte Carlo Simulations
        sim_data = []
        for ticker, results in simulations.items():
            sim_data.append({
                'Ticker': ticker,
                'Mean Return': results.get('mean_return', 0),
                'Std Dev': results.get('std_return', 0),
                'P5': results.get('percentile_5', 0),
                'P95': results.get('percentile_95', 0),
                'Prob Profit': results.get('probability_profit', 0),
                'Simulations': results.get('simulations_count', 0),
                'Horizon Days': results.get('horizon_days', 0)
            })

        if sim_data:
            sim_df = pd.DataFrame(sim_data)
            sim_df.to_excel(writer, sheet_name='Monte_Carlo', index=False)
            _format_sheet(writer.sheets['Monte_Carlo'], header_format)

        # Sheet 3: Portfolio Summary
        summary_data = [{
            'Metric': k,
            'Value': v
        } for k, v in portfolio_summary.items()]

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Portfolio_Summary', index=False)
        _format_sheet(writer.sheets['Portfolio_Summary'], header_format)

        # Sheet 4: Compliance Results
        if compliance_results:
            comp_data = []
            for result in compliance_results:
                comp_data.append({
                    'Ticker': result.get('ticker', ''),
                    'Passed': result.get('passed', False),
                    'Checks': str(result.get('checks', {})),
                    'Warnings': '; '.join(result.get('warnings', []))
                })

            comp_df = pd.DataFrame(comp_data)
            comp_df.to_excel(writer, sheet_name='Compliance', index=False)
            _format_sheet(writer.sheets['Compliance'], header_format)

        # Sheet 5: Learning Stats
        stats_data = [{
            'Metric': k,
            'Value': v
        } for k, v in learning_stats.items()]

        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Learning_Stats', index=False)
        _format_sheet(writer.sheets['Learning_Stats'], header_format)

        # Sheet 6: Metadata
        metadata = [{
            'Key': 'Generated At',
            'Value': datetime.now().isoformat()
        }, {
            'Key': 'Report Type',
            'Value': 'Portfolio Optimization'
        }, {
            'Key': 'Mode',
            'Value': 'Paper Trading / Decision Support'
        }]

        meta_df = pd.DataFrame(metadata)
        meta_df.to_excel(writer, sheet_name='Metadata', index=False)
        _format_sheet(writer.sheets['Metadata'], header_format)

    return output_path


def _format_sheet(worksheet, header_format) -> None:
    """Apply formatting to worksheet.

    Args:
        worksheet: XlsxWriter worksheet object.
        header_format: Format for header row.
    """
    # Auto-fit columns - xlsxwriter uses set_column with width
    # Just apply header format to first row
    worksheet.set_row(0, None, header_format)
    
    # Set a reasonable default column width
    worksheet.set_column(0, 10, 15)


def create_position_tracking_sheet(df: pd.DataFrame, sheet_name: str,
                                    writer: pd.ExcelWriter) -> None:
    """Add position tracking sheet to existing writer.

    Args:
        df: DataFrame with position data.
        sheet_name: Name for the sheet.
        writer: ExcelWriter object.
    """
    if not df.empty:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def format_currency_cell(worksheet, row: int, col: int, value: float) -> None:
    """Write a formatted currency cell.

    Args:
        worksheet: Worksheet object.
        row: Row index.
        col: Column index.
        value: Currency value.
    """
    worksheet.write_number(row, col, value, {'num_format': '₹#,##0.00'})


def format_percent_cell(worksheet, row: int, col: int, value: float) -> None:
    """Write a formatted percentage cell.

    Args:
        worksheet: Worksheet object.
        row: Row index.
        col: Column index.
        value: Percentage value.
    """
    worksheet.write_number(row, col, value, {'num_format': '0.00%'})

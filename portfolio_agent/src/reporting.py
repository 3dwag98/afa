"""Excel reporting module."""

import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional


def create_excel_report(recommendations: List[Dict[str, Any]],
                        simulations: Dict[str, Dict[str, Any]],
                        portfolio_summary: Dict[str, Any],
                        compliance_results: List[Dict[str, Any]],
                        learning_stats: Dict[str, Any],
                        output_path: str) -> str:
    """Create comprehensive Excel report.

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
    from pathlib import Path
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

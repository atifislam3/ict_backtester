import argparse
import logging
import json
from pathlib import Path
import dataclasses
from tabulate import tabulate

from src.logger import setup_logger
from src.config_manager import ConfigManager
from src.data_loader import DataLoader
from src.backtest_engine import BacktestEngine
from src.statistics import calculate_statistics
from src.visualizer import generate_visual_report

def print_summary_table(stats: dict):
    print("\n" + "="*40)
    print("        BACKTEST SUMMARY TABLE")
    print("="*40)
    
    table = [
        ["Total Trades", stats.get('total_trades', 0)],
        ["Win Rate", f"{stats.get('win_rate_pct', 0.0):.2f}%"],
        ["Loss Rate", f"{stats.get('loss_rate_pct', 0.0):.2f}%"],
        ["Expectancy (R)", f"{stats.get('expectancy_r', 0.0):.2f}"],
        ["Profit Factor", f"{stats.get('profit_factor', 0.0):.2f}"],
        ["Max Drawdown (%)", f"{stats.get('max_drawdown_pct', 0.0):.2f}%"],
        ["Sharpe Ratio (Ann.)", f"{stats.get('annualized_sharpe_ratio', 0.0):.2f}"]
    ]
    
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
    print("="*40 + "\n")

def main():
    parser = argparse.ArgumentParser(description="ICT Backtester Pro CLI")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(name="ict_main", console_output=True)
    logger.info(f"Starting ICT Backtester Pro with config: {args.config}")

    # 1. Load config using the new config manager
    config_manager = ConfigManager(args.config)
    config = config_manager.load_config()
    logger.info(f"Configuration loaded for instrument: {config.instrument}")

    # 2. Fetch Data
    logger.info("Fetching and preparing data...")
    loader = DataLoader()
    df = loader.get_data(
        instrument=config.instrument,
        timeframe=config.timeframe,
        start_date=config.date_range.start,
        end_date=config.date_range.end
    )
    
    if df.empty:
        logger.error("No data available for backtesting. Exiting.")
        return

    logger.info(f"Data loaded successfully. Total candles: {len(df)}")

    # 3. Run Backtest
    logger.info("Running Event-Driven Backtest...")
    # Convert ConfigSchema model to dict for backward compatibility with BacktestEngine
    config_dict = config.model_dump()
    engine = BacktestEngine(df, config_dict)
    trades_df = engine.run()
    
    if trades_df.empty:
        logger.warning("No trades were taken during the backtest.")
        return
        
    logger.info(f"Backtest completed. Total trades taken: {len(trades_df)}")

    # 4. Calculate Statistics
    logger.info("Calculating statistics...")
    stats = calculate_statistics(
        trades_df, 
        initial_capital=100000.0, 
        risk_percent=config.risk_settings.risk_per_trade_pct
    )

    # 5. Output Summary to Terminal
    print_summary_table(stats)

    # 6. Save JSON Summary
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"stats_summary_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Saved stats summary to {json_path}")

    # 7. Generate Visual Report
    logger.info("Generating interactive HTML report...")
    report_path = generate_visual_report(trades_df, stats, output_dir="reports")
    logger.info(f"Visual report successfully generated at: {report_path}")

if __name__ == "__main__":
    main()

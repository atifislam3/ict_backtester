import logging
from src.config import load_config
from src.data_loader import load_data
from src.backtest_engine import BacktestEngine
from src.statistics import calculate_statistics
from src.visualizer import generate_visual_report
import dataclasses

# Set up simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Load Configurations
    config = load_config()
    logger.info(f"Loaded configuration for instrument: {config.instrument}")
    
    # 2. Fetch/Load Historical Data
    df = load_data(
        instrument=config.instrument,
        timeframe=config.timeframe,
        start_date=config.start_date,
        end_date=config.end_date
    )
    
    if df.empty:
        logger.error("No data returned. Exiting.")
        return
        
    logger.info(f"Loaded {len(df)} candles for backtesting.")

    # 3. Run the Backtest Engine
    config_dict = dataclasses.asdict(config)
    engine = BacktestEngine(df, config_dict)
    trades_df = engine.run()
    
    if trades_df.empty:
        logger.warning("No trades were executed during the backtest.")
        return
        
    # 4. Calculate Statistics
    stats = calculate_statistics(trades_df, initial_capital=100000.0, risk_percent=1.0)
    
    logger.info("=== Backtest Statistics ===")
    logger.info(f"Total Trades: {stats['total_trades']}")
    logger.info(f"Win Rate: {stats['win_rate_pct']:.2f}%")
    logger.info(f"Expectancy (R): {stats['expectancy_r']:.2f}")
    logger.info(f"Profit Factor: {stats['profit_factor']:.2f}")
    
    # 5. Generate Interactive Visual Report
    report_path = generate_visual_report(trades_df, stats)
    logger.info(f"Visual report successfully generated at: {report_path}")

if __name__ == "__main__":
    main()

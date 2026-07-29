"""
Data loader module for fetching, processing, and caching historical data.
"""

import os
import subprocess
import time
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import pytz
import yfinance as yf

# Setup module-level logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_dukascopy_data(
    instrument: str, 
    start_date: str, 
    end_date: str, 
    raw_dir: str = "data/raw",
    retries: int = 3,
    delay: int = 5
) -> str:
    """
    Downloads 1-minute historical data from Dukascopy using the dukascopy-node CLI.
    
    Args:
        instrument (str): The instrument to download (e.g., 'xauusd').
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        raw_dir (str): Directory to save raw data.
        retries (int): Number of retries on failure.
        delay (int): Delay in seconds between retries.
        
    Returns:
        str: The path to the downloaded CSV file.
        
    Raises:
        RuntimeError: If download fails after maximum retries.
        FileNotFoundError: If the downloaded file cannot be found.
    """
    os.makedirs(raw_dir, exist_ok=True)
    instrument_lower = instrument.lower()
    
    # We use `npx dukascopy-node` as a reliable way to get free tick/1m data
    cmd = [
        "npx", "dukascopy-node", 
        "-i", instrument_lower, 
        "-start", start_date, 
        "-end", end_date, 
        "-t", "m1", 
        "-f", "csv", 
        "-dir", raw_dir
    ]
    
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading data for {instrument} (Attempt {attempt}/{retries})...")
            # Using shell=True for Windows npx command compatibility if necessary, 
            # but list of args with shell=False is generally safer if npx.cmd is found in PATH.
            # Using shell=True for npx on Windows typically works best to resolve the binary.
            is_windows = os.name == 'nt'
            subprocess.run(cmd, check=True, capture_output=True, text=True, shell=is_windows)
            logger.info("Download completed successfully.")
            
            # Find the downloaded file
            downloaded_files = list(Path(raw_dir).glob(f"*{instrument_lower}*.csv"))
            if not downloaded_files:
                raise FileNotFoundError("Command succeeded but no CSV file was found in output directory.")
                
            # Return the most recently modified file matching the instrument
            latest_file = max(downloaded_files, key=os.path.getmtime)
            return str(latest_file)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Download command failed: {e.stderr}")
            if attempt < retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("Maximum retries reached. Download failed.")
                raise RuntimeError(f"Failed to download data after {retries} attempts.") from e
        except FileNotFoundError as e:
            logger.error("npx command not found. Please ensure Node.js is installed.")
            raise RuntimeError("npx not found") from e
            
    return ""

def validate_data_gaps(df: pd.DataFrame, timeframe: str) -> None:
    """
    Validates data for missing candles and reports them.
    
    Args:
        df (pd.DataFrame): The DataFrame with a DatetimeIndex.
        timeframe (str): The expected timeframe string (e.g., '1H').
    """
    # Map typical backtesting timeframes to pandas Timedelta strings
    tf_map = {'1H': '1h', '4H': '4h', '1D': '1D', '15M': '15min', '5M': '5min', '1M': '1min'}
    pd_tf = tf_map.get(timeframe.upper(), timeframe.lower())
    
    expected_diff = pd.Timedelta(pd_tf).total_seconds()
    
    # Simple validation: Check if there are gaps in the expected business day frequency
    # We ignore weekend gaps (Saturday and Sunday)
    business_hours_idx = df.index[df.index.dayofweek < 5]
    if len(business_hours_idx) > 1:
        time_diffs = business_hours_idx.to_series().diff().dt.total_seconds()
        
        # A gap is any difference strictly greater than the expected timeframe duration
        # (plus a small epsilon to avoid floating point precision issues)
        gaps = time_diffs[time_diffs > (expected_diff + 1.0)]
        
        if not gaps.empty:
            logger.warning(f"Validation Error: Detected {len(gaps)} missing/gap candles in active trading periods!")
            for time_idx, gap_duration in gaps.head(5).items():
                hours_missing = gap_duration / 3600
                logger.warning(f"Gap detected ending at {time_idx}: ~{hours_missing:.1f} hours missing.")
            if len(gaps) > 5:
                logger.warning(f"... (and {len(gaps) - 5} more gaps omitted)")
        else:
            logger.info("Data validation passed. No significant intra-week gaps detected.")

def process_and_cache_data(
    csv_path: str, 
    instrument: str,
    timeframe: str,
    processed_dir: str = "data/processed"
) -> pd.DataFrame:
    """
    Loads raw CSV, converts to EST, resamples, validates, and caches to Parquet.
    
    Args:
        csv_path (str): Path to the raw CSV file.
        instrument (str): The instrument name.
        timeframe (str): Target timeframe for resampling (e.g., '1H', '4H').
        processed_dir (str): Directory to save processed Parquet files.
        
    Returns:
        pd.DataFrame: The processed dataframe.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    logger.info(f"Loading raw data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Normalize column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Identify timestamp column
    time_col = 'timestamp' if 'timestamp' in df.columns else 'time' if 'time' in df.columns else df.columns[0]
    
    # Parse dates (assume UTC from Dukascopy)
    if pd.api.types.is_numeric_dtype(df[time_col]):
        df['datetime'] = pd.to_datetime(df[time_col], unit='ms', utc=True)
    else:
        df['datetime'] = pd.to_datetime(df[time_col], utc=True)
        
    df.set_index('datetime', inplace=True)
    df.drop(columns=[time_col], inplace=True, errors='ignore')
    
    # Convert timezone to EST/America/New_York (ICT uses NY time)
    logger.info("Converting timestamps to America/New_York timezone.")
    df = df.tz_convert('America/New_York')
    
    # Resample
    logger.info(f"Resampling data to {timeframe} timeframe.")
    tf_map = {'1H': '1h', '4H': '4h', '1D': '1D', '15M': '15min'}
    pd_tf = tf_map.get(timeframe.upper(), timeframe.lower())
    
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }
    if 'volume' in df.columns:
        ohlc_dict['volume'] = 'sum'
        
    df_resampled = df.resample(pd_tf).agg(ohlc_dict)
    df_resampled.dropna(inplace=True)
    
    # Validate gaps
    validate_data_gaps(df_resampled, timeframe)
    
    # Save to Parquet
    cache_path = Path(processed_dir) / f"{instrument}_{timeframe}.parquet"
    df_resampled.to_parquet(cache_path)
    logger.info(f"Processed data cached to {cache_path}")
    
    return df_resampled

def load_data(
    instrument: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    force_download: bool = False
) -> pd.DataFrame:
    """
    Loads data using yfinance instead of dukascopy to remove Node.js dependency.
    """
    cache_path = Path(processed_dir) / f"{instrument}_{timeframe}.parquet"
    os.makedirs(processed_dir, exist_ok=True)
    
    start_dt = pd.to_datetime(start_date).tz_localize('America/New_York')
    end_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59).tz_localize('America/New_York')
    
    if not force_download and cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)
        mask = (df.index >= start_dt) & (df.index <= end_dt)
        return df.loc[mask]
        
    logger.info("Fetching new data from yfinance...")
    
    # Map instrument to yfinance ticker
    ticker = instrument.upper()
    if ticker == 'XAUUSD':
        ticker = 'GC=F'
    elif ticker == 'EURUSD':
        ticker = 'EURUSD=X'
        
    # Map timeframe
    tf_map = {'1M': '1m', '5M': '5m', '15M': '15m', '1H': '1h', '4H': '4h', '1D': '1d'}
    yf_tf = tf_map.get(timeframe.upper(), '1h')
    
    try:
        # yfinance historical data
        ticker_obj = yf.Ticker(ticker)
        # Note: 1h data is only available for 730 days max in yfinance
        df = ticker_obj.history(start=start_date, end=end_date, interval=yf_tf)
        
        if df.empty:
            logger.error(f"yfinance returned no data for {ticker} from {start_date} to {end_date}.")
            return pd.DataFrame()
            
        # Standardize columns
        df.columns = [col.lower() for col in df.columns]
        df.index.name = 'datetime'
        
        # Ensure correct timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
        else:
            df.index = df.index.tz_convert('America/New_York')
            
        # Cache to parquet
        df.to_parquet(cache_path)
        logger.info(f"Processed data cached to {cache_path}")
        
        mask = (df.index >= start_dt) & (df.index <= end_dt)
        return df.loc[mask]
        
    except Exception as e:
        logger.error(f"Failed to fetch data from yfinance: {e}")
        return pd.DataFrame()

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import pytz
import yfinance as yf
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

class DataQualityError(Exception):
    """Exception raised for data quality issues like gaps or missing sessions."""
    pass

def exponential_backoff_retry(max_retries: int = 3, base_delay: float = 1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise
                    delay = base_delay * (2 ** (retries - 1))
                    logger.warning(f"Error: {e}. Retrying in {delay}s... (Attempt {retries}/{max_retries})")
                    time.sleep(delay)
        return wrapper
    return decorator

class DataLoader:
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @exponential_backoff_retry(max_retries=3, base_delay=2.0)
    def fetch_yfinance_data(self, ticker: str, start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
        logger.info(f"Downloading {ticker} from {start_date} to {end_date} (interval: {interval})")
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False, multi_level_index=False)
        if df.empty:
            raise ValueError(f"No data returned for {ticker} from {start_date} to {end_date}.")
        
        # yfinance multi-index column handling (if using yf.download with single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            # Safe drop if 'Price' level exists, else drop first
            if 'Ticker' in df.columns.names:
                df.columns = df.columns.droplevel('Ticker')
            else:
                df.columns = df.columns.droplevel(1)
            
        return df

    def _tag_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tags each candle with its session based on America/New_York time."""
        # Using typical ICT times in NY Time (EST/EDT):
        # Asian: 19:00 - 00:00, 00:00 - 03:00 (or roughly 19:00 - 03:00)
        # London: 03:00 - 11:00
        # NY: 08:00 - 17:00
        # Overlap: 08:00 - 11:00
        
        def determine_session(hour: int) -> str:
            if 8 <= hour < 11:
                return "Overlap"
            elif 3 <= hour < 8:
                return "London"
            elif 11 <= hour < 17:
                return "NY"
            else:
                return "Asian"
                
        dt_index = pd.DatetimeIndex(df.index)
        df['session'] = dt_index.hour.map(determine_session)
        return df

    def _validate_data(self, df: pd.DataFrame, timeframe: str) -> None:
        if df.empty:
            raise DataQualityError("DataFrame is empty after processing.")
            
        tf_map = {'1H': '1h', '4H': '4h', '1D': '1D'}
        pd_tf = tf_map.get(timeframe.upper(), timeframe.lower())
        expected_diff = pd.Timedelta(pd_tf).total_seconds()
        
        # Check gaps > 1.5x expected interval
        # Exclude weekends
        dt_idx = pd.DatetimeIndex(df.index)
        business_idx = dt_idx[dt_idx.dayofweek < 5]
        if len(business_idx) > 1:
            diffs = business_idx.to_series().diff().dt.total_seconds()
            gaps = diffs[diffs > (expected_diff * 1.5)]
            
            if not gaps.empty:
                error_msg = f"Detected {len(gaps)} missing/gap candles in active trading periods! First gap at {gaps.index[0]}"
                if len(gaps) > 1000:
                    logger.warning(f"Extreme data gaps detected: {len(gaps)}")
                else:
                    logger.warning(error_msg)
                
        # Check for missing sessions (if intraday)
        if timeframe.upper() in ('1H', '15M', '5M'):
            sessions_present = set(df['session'].unique())
            expected = {'Asian', 'London', 'Overlap', 'NY'}
            missing = expected - sessions_present
            if missing:
                logger.warning(f"Missing sessions in data: {missing}")

    def get_data(self, instrument: str, timeframe: str, start_date: str, end_date: str, force_download: bool = False) -> pd.DataFrame:
        cache_path = self.processed_dir / f"{instrument}_{timeframe}_{start_date}_{end_date}.parquet"
        
        if not force_download and cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            table = pq.read_table(cache_path)
            metadata = table.schema.metadata
            if metadata and b'source' in metadata:
                logger.debug(f"Cache metadata: {metadata}")
            return table.to_pandas()
            
        # Map instrument to yfinance
        ticker = instrument.upper()
        if ticker in ('XAUUSD', 'XAUUSD=X'):
            ticker = 'GC=F'
        elif ticker == 'EURUSD':
            ticker = 'EURUSD=X'
            
        # Fetch base data (fetch 1h data for resampling)
        base_interval = '1h'
        if timeframe.upper() in ('1D', 'D'):
            base_interval = '1d'
            
        df = self.fetch_yfinance_data(ticker, start_date, end_date, interval=base_interval)
        
        # Standardize
        df.columns = [c.lower() for c in df.columns]
        df.index.name = 'datetime'
        
        # Timezone conversion (yfinance returns timezone-aware data if available, or naive)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
        else:
            df.index = df.index.tz_convert('America/New_York')
            
        # Resample if needed
        if timeframe.upper() not in ('1H', '1D'):
            logger.info(f"Resampling to {timeframe}")
            tf_map = {'4H': '4h'}
            pd_tf = tf_map.get(timeframe.upper(), timeframe.lower())
            
            agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
            if 'volume' in df.columns:
                agg_dict['volume'] = 'sum'
                
            df = df.resample(pd_tf).agg(agg_dict).dropna()
            
        df = self._tag_sessions(df)
        self._validate_data(df, timeframe)
        
        # Save to Parquet with metadata
        table = pa.Table.from_pandas(df)
        custom_metadata = {
            b'source': b'yfinance',
            b'downloaded_at': str(datetime.now()).encode('utf-8'),
            b'row_count': str(len(df)).encode('utf-8')
        }
        # Merge with existing pandas metadata
        existing_meta = table.schema.metadata or {}
        merged_meta = {**existing_meta, **custom_metadata}
        table = table.replace_schema_metadata(merged_meta)
        
        pq.write_table(table, cache_path)
        logger.info(f"Cached processed data to {cache_path}")
        
        return df

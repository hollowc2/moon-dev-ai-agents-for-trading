"""
🌙 Billy Bitcoin's Time Utilities
Centralized time handling functions
"""

from datetime import datetime, timedelta
import pytz
import pandas as pd

def get_utc_now():
    """Get current UTC time"""
    return datetime.now(pytz.UTC)

def get_unix_timestamp(dt=None):
    """
    Convert datetime to Unix timestamp
    If no datetime provided, uses current UTC time
    """
    if dt is None:
        dt = get_utc_now()
    return int(dt.timestamp())

def get_unix_timestamp_range(days_back=1):
    """
    Get start and end Unix timestamps for a date range
    Returns: (start_unix, end_unix)
    """
    end_time = get_utc_now()
    start_time = end_time - timedelta(days=days_back)
    
    return (
        get_unix_timestamp(start_time),
        get_unix_timestamp(end_time)
    )

def unix_to_datetime(unix_timestamp):
    """Convert Unix timestamp to datetime object"""
    try:
        # Handle both string and integer timestamps
        if isinstance(unix_timestamp, str):
            unix_timestamp = int(unix_timestamp)
        return datetime.fromtimestamp(unix_timestamp, pytz.UTC)
    except Exception as e:
        print(f"❌ Error converting timestamp {unix_timestamp}: {str(e)}")
        return None

def format_timestamp(dt, format='%Y-%m-%d %H:%M:%S'):
    """Format datetime object to string"""
    try:
        if isinstance(dt, (int, str)):
            dt = unix_to_datetime(dt)
        return dt.strftime(format)
    except Exception as e:
        print(f"❌ Error formatting datetime {dt}: {str(e)}")
        return None

def convert_df_timestamps(df, timestamp_column='start'):
    """Convert DataFrame timestamp column from Unix to datetime"""
    try:
        df = df.copy()
        df[timestamp_column] = pd.to_datetime(df[timestamp_column].astype(int), unit='s', utc=True)
        return df
    except Exception as e:
        print(f"❌ Error converting DataFrame timestamps: {str(e)}")
        return df

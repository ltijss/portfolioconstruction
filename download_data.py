"""
Market Close Data Fetcher with Exchange Calendar
Downloads close prices for specified tickers aligned to exchange trading days.
Missing data is forward-filled with the previous close price.
"""

import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
import os
from datetime import timedelta


def get_last_trading_day(exchange='XAMS', today=None):
    """Returns last business day <= today for given exchange as 'YYYY-MM-DD'."""
    if today is None:
        today = pd.Timestamp.now().date()

    cal = mcal.get_calendar(exchange)  # Dynamic: 'XAMS', 'NYSE', etc.
    recent_days = cal.valid_days(
        start_date=(pd.Timestamp(today) - timedelta(days=5)).date(),
        end_date=(pd.Timestamp(today) - timedelta(days=1)).date()  # Yesterday max
    )
    return recent_days[-1].strftime('%Y-%m-%d')

def get_market_close_data(start_date, exchange, tickers, end_date=None, save_to_csv=False):
    """
    Download and align close prices to exchange trading days.

    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD' (e.g., '2024-01-01')
    end_date : str
        End date in format 'YYYY-MM-DD' (e.g., '2024-12-31')
    exchange : str
        Exchange calendar code (e.g., 'XAMS', 'XLON', 'XPAR', 'XFRA')
        Common options:
        - 'XAMS': Amsterdam/Euronext
        - 'XLON': London Stock Exchange
        - 'XPAR': Euronext Paris
        - 'XFRA': Frankfurt/Deutsche Börse
        - 'XMIL': Milan
    tickers : list
        List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
    save_to_csv : bool, optional
        If True, saves output to CSV in 'data' folder (default: False)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with dates as rows and tickers as columns containing close prices

    Example:
    --------
    df = get_market_close_data(
        start_date='2024-01-01',
        end_date='2024-12-31',
        exchange='XAMS',
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        save_to_csv=True
    )
    """

    print("=" * 60)
    print("MARKET CLOSE DATA FETCHER")
    print("=" * 60)
    print(f"Start Date: {start_date}")
    if end_date is None:
        end_date = get_last_trading_day(exchange=exchange)
        print(f"Auto-set end_date to last {exchange} trading day: {end_date}")
    else:
        print(f"User input End Date: {end_date}")
    print(f"Exchange: {exchange}")
    print(f"Tickers: {', '.join(tickers)}")
    print("-" * 60)

    # Step 1: Get trading days from exchange calendar
    print(f"\n[1/4] Getting trading days for {exchange}...")
    try:
        calendar = mcal.get_calendar(exchange)
        schedule = calendar.schedule(start_date=start_date, end_date=end_date)
        trading_days = schedule.index
        print(f"   ✓ Found {len(trading_days)} trading days")
    except Exception as e:
        print(f"   ✗ Error getting calendar: {e}")
        print(f"   Available exchanges: {mcal.get_calendar_names()}")
        return None

    # Step 2: Download close prices for all tickers
    print(f"\n[2/4] Downloading close prices for {len(tickers)} tickers...")
    close_data = {}

    for ticker in tickers:
        try:
            # Download data with some buffer before start_date to ensure forward-fill works
            buffer_start = pd.to_datetime(start_date) - pd.Timedelta(days=30)
            data = yf.download(ticker, start=buffer_start, end=end_date, auto_adjust=True, progress=False)

            if not data.empty and 'Close' in data.columns:
                close_data[ticker] = data['Close']
                print(f"   ✓ {ticker}: {len(data)} data points")
            else:
                print(f"   ✗ {ticker}: No data available")
                close_data[ticker] = pd.Series(dtype=float)
        except Exception as e:
            print(f"   ✗ {ticker}: Error - {e}")
            close_data[ticker] = pd.Series(dtype=float)

    # Step 3: Create DataFrame with all tickers
    print(f"\n[3/4] Aligning data to trading days...")
    df_list = list(close_data.values())
    df = pd.concat(df_list, axis=1, keys=close_data.keys())
    df.columns = close_data.keys()

    # Reindex to trading days only
    df = df.reindex(trading_days)

    # Forward fill missing values (use previous close price)
    df = df.ffill()

    # Trim to valid data range
    print(f"\n   Trimming to valid data range...")
    original_length = len(df)

    # Find minimum date: first date where all tickers have data
    has_all_data = df.notna().all(axis=1)
    if has_all_data.any():
        min_valid_date = df[has_all_data].index[0]

        # Find which tickers are responsible for the minimum date
        # (tickers that had no data before this date)
        min_responsible = []
        for ticker in df.columns:
            first_valid = df[ticker].first_valid_index()
            if first_valid == min_valid_date:
                min_responsible.append(ticker)

        print(f"   ✓ Minimum date: {min_valid_date.date()}")
        if min_responsible:
            print(f"      Limited by: {', '.join(min_responsible)}")
    else:
        print(f"   ✗ Error: No date found where all tickers have data")
        return None

    # Find maximum date: last date where all tickers have data
    if has_all_data.any():
        max_valid_date = df[has_all_data].index[-1]

        # Find which tickers are responsible for the maximum date
        # (tickers that had no data after this date)
        max_responsible = []
        for ticker in df.columns:
            last_valid = df[ticker].last_valid_index()
            if last_valid == max_valid_date:
                max_responsible.append(ticker)

        print(f"   ✓ Maximum date: {max_valid_date.date()}")
        if max_responsible:
            print(f"      Limited by: {', '.join(max_responsible)}")
    else:
        print(f"   ✗ Error: No date found where all tickers have data")
        return None

    # Trim the dataframe
    df = df.loc[min_valid_date:max_valid_date]

    print(f"   ✓ Original range: {original_length} days ({trading_days[0].date()} to {trading_days[-1].date()})")
    print(f"   ✓ Trimmed range: {len(df)} days ({min_valid_date.date()} to {max_valid_date.date()})")

    # Count remaining missing values after trimming
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"   ⚠ Warning: {missing_count} values still missing after trimming")
    else:
        print(f"   ✓ All values present (no missing data)")

    # Step 4: Optionally save to CSV
    if save_to_csv:
        print(f"\n[4/4] Saving to CSV...")
        os.makedirs("data", exist_ok=True)
        filename = f"data/close_prices_{exchange}_{start_date}_to_{end_date}.csv"
        df.to_csv(filename)
        print(f"   ✓ Saved to: {filename}")
    else:
        print(f"\n[4/4] Skipping CSV save (save_to_csv=False)")

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print("\nData shape:", df.shape)
    print("=" * 60)

    return df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
#if __name__ == "__main__":
    # Example 1: Basic usage
    # result = get_market_close_data(
    #     start_date='2024-01-01',
    #     end_date='2024-12-31',
    #     exchange='XAMS',  # Amsterdam exchange
    #     tickers=['AAPL', 'MSFT', 'GOOGL','^GSPC','^990100-USD-STRD'],
    #     save_to_csv=True
    # )


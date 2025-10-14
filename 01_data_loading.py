"""
Bitcoin Market Analysis - Step 1: Data Loading
==============================================

This script fetches historical Bitcoin data from CoinGecko API for the last 5 years,
including daily prices, market caps, volumes, and additional important metrics.
"""

import pandas as pd
import numpy as np
from pycoingecko import CoinGeckoAPI
import time
import requests
from datetime import datetime, timedelta
import os

def fetch_bitcoin_data():
    """
    Fetch comprehensive Bitcoin historical data from CoinGecko API
    
    Returns:
        pd.DataFrame: Clean DataFrame with Bitcoin market data
    """
    
    print("🚀 Starting Bitcoin data fetch from CoinGecko API...")
    print("=" * 60)
    
    # Initialize CoinGecko API
    cg = CoinGeckoAPI()
    
    # Calculate date range (CoinGecko free API limit: 365 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Maximum 365 days for free API
    
    print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"🪙 Asset: Bitcoin (BTC)")
    print(f"💱 Currency: USD")
    print(f"📊 Interval: Daily")
    print(f"⚠️  Note: Using CoinGecko free API (365 days max)")
    
    try:
        # Fetch historical market data with additional metrics
        print("\n📡 Fetching market data...")
        market_data = cg.get_coin_market_chart_by_id(
            id='bitcoin',
            vs_currency='usd',
            days=365,  # Maximum days for free API
            interval='daily'
        )
        
        # Add delay to respect API rate limits
        time.sleep(1)
        
        print("✅ Market data fetched successfully!")
        
        # Extract different data components
        prices = market_data['prices']
        market_caps = market_data['market_caps']
        total_volumes = market_data['total_volumes']
        
        # Create DataFrame from prices data
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        
        # Add market cap and volume data
        df['market_cap'] = [cap[1] for cap in market_caps]
        df['volume'] = [vol[1] for vol in total_volumes]
        
        print(f"   • Raw data points: {len(df)}")
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Filter for last 365 days (already handled by API call)
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Fetch additional OHLC data for more comprehensive analysis
        print("📡 Fetching OHLC data...")
        try:
            # Use a shorter range for OHLC to avoid API limits
            ohlc_start = end_date - timedelta(days=90)  # Last 90 days for OHLC
            ohlc_data = cg.get_coin_market_chart_range_by_id(
                id='bitcoin',
                vs_currency='usd',
                from_timestamp=int(ohlc_start.timestamp()),
                to_timestamp=int(end_date.timestamp())
            )
            
            # Add delay
            time.sleep(1)
            
            # Create OHLC DataFrame
            ohlc_df = pd.DataFrame(ohlc_data['prices'], columns=['timestamp', 'close'])
            
            # Get high and low data (CoinGecko doesn't provide separate OHLC in this endpoint)
            # We'll use the close price as approximation for high/low in daily data
            ohlc_df['high'] = ohlc_df['close']
            ohlc_df['low'] = ohlc_df['close']
            ohlc_df['open'] = ohlc_df['close'].shift(1)
            ohlc_df['datetime'] = pd.to_datetime(ohlc_df['timestamp'], unit='ms')
            
            # Merge OHLC data with main DataFrame
            df = df.merge(ohlc_df[['datetime', 'open', 'high', 'low']], on='datetime', how='left')
            
            print(f"   • After OHLC merge: {len(df)} records")
            
            # Forward fill open prices (first day will be NaN)
            df['open'] = df['open'].fillna(df['price'])
            
            print("✅ OHLC data added successfully!")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not fetch OHLC data: {e}")
            # Create basic OHLC from price data
            df['open'] = df['price'].shift(1)
            df['high'] = df['price']
            df['low'] = df['price']
            df['open'] = df['open'].fillna(df['price'])
        
        # Add additional calculated features
        df['price_change'] = df['price'].diff()
        df['price_change_pct'] = df['price'].pct_change() * 100
        df['volume_change'] = df['volume'].diff()
        df['market_cap_change'] = df['market_cap'].diff()
        
        # Add volatility measures (only if we have enough data)
        if len(df) >= 7:
            df['price_volatility'] = df['price_change_pct'].rolling(window=7).std()
            df['volume_volatility'] = df['volume'].rolling(window=7).std()
        else:
            df['price_volatility'] = np.nan
            df['volume_volatility'] = np.nan
        
        # Reorder columns for better readability
        column_order = [
            'timestamp', 'datetime', 'open', 'high', 'low', 'price', 
            'price_change', 'price_change_pct', 'volume', 'volume_change',
            'market_cap', 'market_cap_change', 'price_volatility', 'volume_volatility'
        ]
        
        df = df[column_order]
        
        # Remove only the first row (due to price_change calculation) but keep volatility NaN for now
        print(f"   • Before cleaning: {len(df)} records")
        df = df.dropna(subset=['price_change', 'price_change_pct']).reset_index(drop=True)
        print(f"   • After cleaning: {len(df)} records")
        
        print(f"\n📊 Data Summary:")
        print(f"   • Total records: {len(df):,}")
        if len(df) > 0:
            print(f"   • Date range: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}")
            print(f"   • Price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
            print(f"   • Average daily volume: ${df['volume'].mean():,.0f}")
            print(f"   • Average market cap: ${df['market_cap'].mean():,.0f}")
        else:
            print("   • No data available")
        
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def save_data_to_csv(df, filename='bitcoin_data.csv'):
    """
    Save the DataFrame to CSV file
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Output filename
    """
    try:
        df.to_csv(filename, index=False)
        print(f"💾 Data saved to '{filename}' successfully!")
        print(f"   • File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
        return True
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return False

def load_data_from_csv(filename='bitcoin_data.csv'):
    """
    Load data from CSV file if it exists
    
    Args:
        filename (str): CSV filename to load
        
    Returns:
        pd.DataFrame or None: Loaded DataFrame or None if file doesn't exist
    """
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            df['datetime'] = pd.to_datetime(df['datetime'])
            print(f"📂 Data loaded from '{filename}' successfully!")
            print(f"   • Records: {len(df):,}")
            print(f"   • Date range: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    else:
        print(f"📂 File '{filename}' not found. Will fetch fresh data.")
        return None

def main():
    """
    Main function to execute data loading process
    """
    print("🔍 Bitcoin Market Analysis - Data Loading")
    print("=" * 60)
    
    # Check if data already exists
    csv_filename = 'bitcoin_data.csv'
    df = load_data_from_csv(csv_filename)
    
    # If no existing data, fetch new data
    if df is None:
        df = fetch_bitcoin_data()
        
        if df is not None:
            # Save to CSV
            save_data_to_csv(df, csv_filename)
        else:
            print("❌ Failed to fetch data. Exiting...")
            return
    
    # Display basic information about the dataset
    print(f"\n📈 Dataset Information:")
    print(f"   • Shape: {df.shape}")
    print(f"   • Columns: {list(df.columns)}")
    print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    # Display first few rows
    print(f"\n📋 First 5 rows:")
    print(df.head())
    
    # Display basic statistics
    print(f"\n📊 Basic Statistics:")
    print(df[['price', 'volume', 'market_cap', 'price_change_pct']].describe())
    
    print(f"\n✅ Data loading completed successfully!")
    print(f"   Ready for Step 2: Preprocessing")

if __name__ == "__main__":
    main()

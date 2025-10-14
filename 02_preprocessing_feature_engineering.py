"""
Bitcoin Market Analysis - Step 2 & 3: Preprocessing and Feature Engineering
=========================================================================

This script handles data preprocessing and creates comprehensive technical indicators
including EMA, RSI, Bollinger Bands, and other features for machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_bitcoin_data(filename='bitcoin_data.csv'):
    """
    Load Bitcoin data from CSV file
    
    Args:
        filename (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded Bitcoin data
    """
    print("📂 Loading Bitcoin data...")
    df = pd.read_csv(filename)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    print(f"✅ Data loaded: {len(df)} records from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    return df

def preprocess_data(df):
    """
    Step 2: Data Preprocessing and Cleaning
    
    Args:
        df (pd.DataFrame): Raw Bitcoin data
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed data
    """
    print("\n🧹 Step 2: Data Preprocessing")
    print("=" * 50)
    
    # Check for missing values
    print("📊 Missing values analysis:")
    missing_values = df.isnull().sum()
    for col, missing in missing_values.items():
        if missing > 0:
            print(f"   • {col}: {missing} missing values")
    
    # Handle missing values strategically
    if missing_values.sum() > 0:
        print("🔧 Handling missing values...")
        
        # For OHLC data, use price as approximation if missing
        if df['high'].isnull().any():
            print("   • Using price as high/low approximation")
            df['high'] = df['high'].fillna(df['price'])
            df['low'] = df['low'].fillna(df['price'])
        
        # Handle completely missing OHLC columns
        if df['high'].isnull().all():
            print("   • Creating OHLC from price data")
            df['high'] = df['price'] * 1.01  # Assume 1% higher than close
            df['low'] = df['price'] * 0.99   # Assume 1% lower than close
            df['open'] = df['price'].shift(1).fillna(df['price'])  # Previous day's close as open
        
        # For volatility columns, fill with 0 if they're all NaN or mostly NaN
        volatility_cols = ['price_volatility', 'volume_volatility']
        for col in volatility_cols:
            if col in df.columns:
                if df[col].isnull().all():
                    df[col] = 0
                    print(f"   • Filled {col} with 0 (all NaN)")
                elif df[col].isnull().any():
                    df[col] = df[col].fillna(0)
                    print(f"   • Filled {col} missing values with 0")
        
        # Forward fill remaining missing values
        df = df.fillna(method='ffill')
        print("✅ Missing values handled")
    else:
        print("✅ No missing values found")
    
    # Check for any remaining missing values
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"⚠️  Warning: {remaining_missing} missing values remain")
        # Fill remaining with mean for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        print("✅ Filled remaining missing values with mean")
    
    # Data quality checks
    print("\n📈 Data Quality Checks:")
    print(f"   • Total records: {len(df):,}")
    
    if len(df) == 0:
        print("❌ Error: No data remaining after preprocessing!")
        return None
    
    print(f"   • Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"   • Price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
    print(f"   • Volume range: ${df['volume'].min():,.0f} - ${df['volume'].max():,.0f}")
    
    # Check for any unrealistic values
    negative_prices = (df['price'] <= 0).sum()
    negative_volumes = (df['volume'] < 0).sum()
    
    if negative_prices > 0:
        print(f"⚠️  Warning: {negative_prices} records with non-positive prices")
    if negative_volumes > 0:
        print(f"⚠️  Warning: {negative_volumes} records with negative volumes")
    
    print("✅ Data preprocessing completed")
    return df

def calculate_technical_indicators(df):
    """
    Step 3: Feature Engineering - Technical Indicators
    
    Args:
        df (pd.DataFrame): Preprocessed Bitcoin data
        
    Returns:
        pd.DataFrame: Data with technical indicators
    """
    print("\n🔧 Step 3: Feature Engineering - Technical Indicators")
    print("=" * 60)
    
    # Create a copy to avoid modifying original data
    df_features = df.copy()
    
    print("📊 Calculating technical indicators...")
    
    # 1. Moving Averages
    print("   • Simple Moving Averages (SMA)")
    df_features['sma_20'] = df_features['price'].rolling(window=20).mean()
    df_features['sma_50'] = df_features['price'].rolling(window=50).mean()
    
    # 2. Exponential Moving Averages (EMA)
    print("   • Exponential Moving Averages (EMA)")
    df_features['ema_12'] = df_features['price'].ewm(span=12).mean()
    df_features['ema_26'] = df_features['price'].ewm(span=26).mean()
    
    # 3. MACD (Moving Average Convergence Divergence)
    print("   • MACD Indicator")
    df_features['macd'] = df_features['ema_12'] - df_features['ema_26']
    df_features['macd_signal'] = df_features['macd'].ewm(span=9).mean()
    df_features['macd_histogram'] = df_features['macd'] - df_features['macd_signal']
    
    # 4. Relative Strength Index (RSI) - 14 days
    print("   • Relative Strength Index (RSI - 14 days)")
    delta = df_features['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_features['rsi'] = 100 - (100 / (1 + rs))
    
    # 5. Bollinger Bands (20-day SMA with 2 standard deviations)
    print("   • Bollinger Bands (20-day SMA, 2 std dev)")
    df_features['bb_middle'] = df_features['price'].rolling(window=20).mean()
    bb_std = df_features['price'].rolling(window=20).std()
    df_features['bb_upper'] = df_features['bb_middle'] + (bb_std * 2)
    df_features['bb_lower'] = df_features['bb_middle'] - (bb_std * 2)
    df_features['bb_width'] = df_features['bb_upper'] - df_features['bb_lower']
    df_features['bb_position'] = (df_features['price'] - df_features['bb_lower']) / (df_features['bb_upper'] - df_features['bb_lower'])
    
    # 6. Price-based features
    print("   • Price-based features")
    df_features['price_change'] = df_features['price'].diff()
    df_features['price_change_pct'] = df_features['price'].pct_change() * 100
    df_features['price_high_low_ratio'] = df_features['high'] / df_features['low']
    df_features['price_close_open_ratio'] = df_features['price'] / df_features['open']
    
    # 7. Volume-based features
    print("   • Volume-based features")
    df_features['volume_sma_20'] = df_features['volume'].rolling(window=20).mean()
    df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma_20']
    df_features['volume_change'] = df_features['volume'].diff()
    df_features['volume_change_pct'] = df_features['volume'].pct_change() * 100
    
    # 8. Volatility features
    print("   • Volatility features")
    df_features['price_volatility_7'] = df_features['price_change_pct'].rolling(window=7).std()
    df_features['price_volatility_14'] = df_features['price_change_pct'].rolling(window=14).std()
    df_features['volume_volatility_7'] = df_features['volume'].rolling(window=7).std()
    
    # 9. Market cap features
    print("   • Market cap features")
    df_features['market_cap_change'] = df_features['market_cap'].diff()
    df_features['market_cap_change_pct'] = df_features['market_cap'].pct_change() * 100
    
    # 10. Trend indicators
    print("   • Trend indicators")
    df_features['price_vs_sma20'] = df_features['price'] / df_features['sma_20'] - 1
    df_features['price_vs_sma50'] = df_features['price'] / df_features['sma_50'] - 1
    df_features['sma20_vs_sma50'] = df_features['sma_20'] / df_features['sma_50'] - 1
    df_features['ema12_vs_ema26'] = df_features['ema_12'] / df_features['ema_26'] - 1
    
    # 11. Technical indicator signals
    print("   • Technical indicator signals")
    df_features['rsi_oversold'] = (df_features['rsi'] < 30).astype(int)
    df_features['rsi_overbought'] = (df_features['rsi'] > 70).astype(int)
    df_features['price_above_bb_upper'] = (df_features['price'] > df_features['bb_upper']).astype(int)
    df_features['price_below_bb_lower'] = (df_features['price'] < df_features['bb_lower']).astype(int)
    df_features['macd_bullish'] = (df_features['macd'] > df_features['macd_signal']).astype(int)
    
    # 12. Lagged features (for time series analysis)
    print("   • Lagged features")
    for lag in [1, 2, 3, 5, 7]:
        df_features[f'price_lag_{lag}'] = df_features['price'].shift(lag)
        df_features[f'volume_lag_{lag}'] = df_features['volume'].shift(lag)
        df_features[f'price_change_lag_{lag}'] = df_features['price_change_pct'].shift(lag)
    
    # 13. Rolling statistics
    print("   • Rolling statistics")
    for window in [5, 10, 20]:
        df_features[f'price_mean_{window}'] = df_features['price'].rolling(window=window).mean()
        df_features[f'price_std_{window}'] = df_features['price'].rolling(window=window).std()
        df_features[f'volume_mean_{window}'] = df_features['volume'].rolling(window=window).mean()
    
    print(f"✅ Technical indicators calculated. Total features: {len(df_features.columns)}")
    
    # Display feature summary
    print(f"\n📊 Feature Summary:")
    print(f"   • Original features: {len(df.columns)}")
    print(f"   • New features added: {len(df_features.columns) - len(df.columns)}")
    print(f"   • Total features: {len(df_features.columns)}")
    
    return df_features

def prepare_ml_data(df_features):
    """
    Prepare data for machine learning by scaling and creating target variables
    
    Args:
        df_features (pd.DataFrame): Data with technical indicators
        
    Returns:
        tuple: (X_scaled, y_classification, feature_names, scaler)
    """
    print("\n🤖 Preparing Data for Machine Learning")
    print("=" * 50)
    
    # Select numerical features for ML (exclude datetime and target variables)
    exclude_columns = ['timestamp', 'open', 'high', 'low', 'price', 'volume', 'market_cap']
    
    # Get feature columns (exclude the specified columns)
    feature_columns = [col for col in df_features.columns if col not in exclude_columns]
    
    # Remove any remaining non-numeric columns
    numeric_columns = df_features[feature_columns].select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in feature_columns if col in numeric_columns]
    
    print(f"📊 Selected {len(feature_columns)} features for machine learning")
    
    # Create feature matrix
    X = df_features[feature_columns].copy()
    
    # Handle any remaining missing values
    X = X.fillna(X.mean())
    
    # Create target variable for classification (next day price direction)
    print("🎯 Creating target variables...")
    df_features['next_day_price'] = df_features['price'].shift(-1)
    df_features['price_direction'] = (df_features['next_day_price'] > df_features['price']).astype(int)
    y_classification = df_features['price_direction'].dropna()
    
    # Align X and y (remove last row from X since we can't predict it)
    X = X.iloc[:-1]
    
    print(f"   • Classification target: {len(y_classification)} samples")
    print(f"   • Positive class (price up): {y_classification.sum()} ({y_classification.mean()*100:.1f}%)")
    print(f"   • Negative class (price down): {(1-y_classification).sum()} ({(1-y_classification.mean())*100:.1f}%)")
    
    # Scale features using MinMaxScaler
    print("📏 Scaling features using MinMaxScaler...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
    
    print("✅ Data preparation completed")
    print(f"   • Final feature matrix shape: {X_scaled.shape}")
    print(f"   • Feature range after scaling: {X_scaled.min().min():.3f} - {X_scaled.max().max():.3f}")
    
    return X_scaled, y_classification, feature_columns, scaler

def save_processed_data(df_features, X_scaled, y_classification, feature_columns, filename='bitcoin_processed_data.csv'):
    """
    Save processed data and feature information
    
    Args:
        df_features (pd.DataFrame): Data with all features
        X_scaled (pd.DataFrame): Scaled feature matrix
        y_classification (pd.Series): Target variable
        feature_columns (list): List of feature names
        filename (str): Output filename
    """
    print(f"\n💾 Saving processed data to '{filename}'...")
    
    # Save the complete dataset with features
    df_features.to_csv(filename)
    
    # Save feature matrix and target
    X_scaled.to_csv('bitcoin_features_scaled.csv')
    y_classification.to_csv('bitcoin_target_classification.csv', header=['price_direction'])
    
    # Save feature information
    feature_info = pd.DataFrame({
        'feature_name': feature_columns,
        'feature_type': ['technical' if any(x in col.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'bb']) 
                        else 'price' if 'price' in col.lower()
                        else 'volume' if 'volume' in col.lower()
                        else 'volatility' if 'volatility' in col.lower()
                        else 'lag' if 'lag' in col.lower()
                        else 'signal' if any(x in col.lower() for x in ['oversold', 'overbought', 'bullish'])
                        else 'other' for col in feature_columns]
    })
    feature_info.to_csv('bitcoin_feature_info.csv', index=False)
    
    print(f"✅ Processed data saved successfully!")
    print(f"   • Main dataset: {filename}")
    print(f"   • Scaled features: bitcoin_features_scaled.csv")
    print(f"   • Target variable: bitcoin_target_classification.csv")
    print(f"   • Feature info: bitcoin_feature_info.csv")

def main():
    """
    Main function to execute preprocessing and feature engineering
    """
    print("🔍 Bitcoin Market Analysis - Preprocessing & Feature Engineering")
    print("=" * 70)
    
    # Load data
    df = load_bitcoin_data('bitcoin_data.csv')
    
    # Step 2: Preprocessing
    df_clean = preprocess_data(df)
    
    # Step 3: Feature Engineering
    df_features = calculate_technical_indicators(df_clean)
    
    # Prepare data for ML
    X_scaled, y_classification, feature_columns, scaler = prepare_ml_data(df_features)
    
    # Save processed data
    save_processed_data(df_features, X_scaled, y_classification, feature_columns)
    
    # Display final summary
    print(f"\n🎉 Preprocessing and Feature Engineering Completed!")
    print(f"=" * 60)
    print(f"📊 Final Dataset Summary:")
    print(f"   • Total records: {len(df_features):,}")
    print(f"   • Total features: {len(df_features.columns)}")
    print(f"   • Features for ML: {len(feature_columns)}")
    print(f"   • Date range: {df_features.index.min().strftime('%Y-%m-%d')} to {df_features.index.max().strftime('%Y-%m-%d')}")
    
    print(f"\n📈 Technical Indicators Created:")
    print(f"   • Moving Averages: SMA 20/50, EMA 12/26")
    print(f"   • RSI: 14-day period")
    print(f"   • Bollinger Bands: 20-day SMA, 2 std dev")
    print(f"   • MACD: 12/26/9 configuration")
    print(f"   • Additional features: {len(feature_columns) - 8} custom indicators")
    
    print(f"\n✅ Ready for Step 4: Data Mining Algorithms!")
    
    return df_features, X_scaled, y_classification, feature_columns, scaler

if __name__ == "__main__":
    df_features, X_scaled, y_classification, feature_columns, scaler = main()

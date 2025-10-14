# ₿ Bitcoin Market Analysis - Data Mining Project

A comprehensive data mining project that analyzes Bitcoin market data using four core algorithms: **Classification**, **Clustering**, **Anomaly Detection**, and **Association Rules Mining**.

![Bitcoin Dashboard](https://img.shields.io/badge/Status-Complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Data Mining](https://img.shields.io/badge/Data%20Mining-4%20Algorithms-orange)

## 🎯 Project Overview

This project demonstrates a complete data mining pipeline for Bitcoin market analysis, from raw data collection to interactive visualization. It implements multiple machine learning algorithms to uncover patterns, predict price movements, identify market regimes, detect anomalies, and discover trading rules.

## 🚀 Features

### 📊 Data Mining Algorithms
- **🎯 Classification**: Random Forest for price direction prediction
- **🎯 Clustering**: K-Means for market regime identification  
- **🚨 Anomaly Detection**: Isolation Forest for unusual market events
- **🔗 Association Rules**: Apriori for pattern discovery

### 📈 Technical Analysis
- **40+ Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Real-time Data**: CoinGecko API integration
- **Interactive Visualizations**: Plotly-powered charts
- **Web Dashboard**: Streamlit interface

### 🔧 Analysis Capabilities
- **Price Prediction**: Next-day direction forecasting
- **Market Regimes**: Bull/Bear/Volatile market identification
- **Anomaly Detection**: Unusual trading events
- **Pattern Mining**: IF-THEN trading rules

## 📁 Project Structure

```
bitcoin-data-mining/
├── 01_data_loading.py              # Data collection from CoinGecko API
├── 02_preprocessing_feature_engineering.py  # Feature engineering & preprocessing
├── 03_data_mining_algorithms.py    # 4 core data mining algorithms
├── 04_visualization.py             # Comprehensive visualizations
├── streamlit_dashboard.py          # Interactive web dashboard
├── run_dashboard.py               # Dashboard launcher script
├── requirements.txt               # Python dependencies
├── bitcoin_data.csv              # Raw Bitcoin data
├── bitcoin_processed_data.csv    # Processed data with features
├── bitcoin_feature_importance.csv # Feature importance results
├── bitcoin_cluster_labels.csv    # Market regime clusters
├── bitcoin_anomaly_labels.csv    # Anomaly detection results
└── bitcoin_association_rules.csv # Discovered trading patterns
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bitcoin-data-mining.git
   cd bitcoin-data-mining
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete analysis pipeline**
   ```bash
   # Step 1: Data Collection
   python 01_data_loading.py
   
   # Step 2: Feature Engineering
   python 02_preprocessing_feature_engineering.py
   
   # Step 3: Data Mining Algorithms
   python 03_data_mining_algorithms.py
   
   # Step 4: Visualization
   python 04_visualization.py
   ```

## 🌐 Interactive Dashboard

Launch the Streamlit dashboard for interactive analysis:

```bash
python run_dashboard.py
```

The dashboard will open at `http://localhost:8501` with:
- 📊 **Overview**: Market summary and key metrics
- 📈 **Price Analysis**: Technical indicators and signals
- 🎯 **Clustering**: Market regime visualization
- 🚨 **Anomaly Detection**: Unusual events identification
- 🔍 **Feature Importance**: ML feature rankings
- 🤖 **Machine Learning**: Live model training

## 📊 Data Mining Results

### Classification Performance
- **Algorithm**: Random Forest with Grid Search
- **Accuracy**: 55-65% (above random 50%)
- **Top Features**: Price volatility, RSI, volume ratios
- **Cross-validation**: 5-fold CV for robust evaluation

### Market Regimes Identified
- **Bull Market**: Positive price momentum
- **Bear Market**: Negative price momentum  
- **High Volatility**: Elevated price swings
- **Overbought/Oversold**: RSI-based conditions
- **Stable Market**: Low volatility periods

### Anomaly Detection
- **Detection Rate**: ~5% of trading days
- **Event Types**: Crashes, pumps, volume spikes
- **Algorithm**: Isolation Forest (unsupervised)

### Association Rules
- **Pattern Discovery**: IF-THEN market relationships
- **Confidence Levels**: 30-70% for strong rules
- **Examples**: "IF high volume AND oversold RSI THEN price increase"

## 🔧 Technical Details

### Data Sources
- **CoinGecko API**: Real-time Bitcoin market data
- **Time Period**: Last 365 days (API limitation)
- **Frequency**: Daily OHLCV data
- **Features**: Price, volume, market cap, technical indicators

### Technical Indicators
- **Moving Averages**: SMA 20/50, EMA 12/26
- **Momentum**: RSI (14-day), MACD (12/26/9)
- **Volatility**: Bollinger Bands, price volatility
- **Volume**: Volume ratios, volume moving averages
- **Trend**: Price vs moving average ratios

### Machine Learning Pipeline
1. **Data Preprocessing**: Missing value handling, scaling
2. **Feature Engineering**: 40+ technical indicators
3. **Model Selection**: Grid search optimization
4. **Evaluation**: Cross-validation, multiple metrics
5. **Interpretation**: Feature importance, rule extraction

## 📈 Usage Examples

### Running Individual Components
```python
# Load and analyze data
from 01_data_loading import fetch_bitcoin_data
df = fetch_bitcoin_data()

# Preprocess and engineer features  
from 02_preprocessing_feature_engineering import calculate_technical_indicators
df_features = calculate_technical_indicators(df)

# Run data mining algorithms
from 03_data_mining_algorithms import classification_random_forest
model, accuracy = classification_random_forest(X_scaled, y_classification)
```

### Custom Analysis
```python
# Add custom technical indicators
df['custom_indicator'] = df['price'].rolling(10).mean() / df['price'].rolling(30).mean()

# Run clustering with custom parameters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(features)
```

## 🎯 Key Findings

1. **Market Efficiency**: Bitcoin shows some predictable patterns despite being a volatile asset
2. **Regime Persistence**: Market regimes typically last 3-14 days
3. **Volume-Price Relationship**: High volume often precedes significant price movements
4. **Technical Indicators**: RSI and Bollinger Bands are most predictive
5. **Anomaly Patterns**: Unusual events often cluster around major news/events

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CoinGecko**: For providing free Bitcoin market data API
- **Scikit-learn**: For comprehensive machine learning algorithms
- **Streamlit**: For the interactive web dashboard framework
- **Plotly**: For beautiful interactive visualizations

## 📞 Contact

- **Project Link**:((https://github.com/YogitaPatil12095/Bitcoin-Data))
- **Issues**:https://github.com/YogitaPatil12095/Bitcoin-Data

---

⭐ **Star this repository if you found it helpful!**

🔗 **Share with others who might benefit from Bitcoin market analysis**

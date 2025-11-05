import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import warnings
import re
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bitcoin Market Analysis Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #f7931a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f7931a;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
    }
    .tip-box {
        background-color: #fff9e6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #f0ad4e;
        margin: 10px 0;
    }
    .help-text {
        font-size: 0.85em;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all Bitcoin analysis data"""
    try:
        # Load main dataset
        df_features = pd.read_csv('bitcoin_processed_data.csv')
        df_features['datetime'] = pd.to_datetime(df_features['datetime'])
        df_features.set_index('datetime', inplace=True)
        
        # Load results
        feature_importance = pd.read_csv('bitcoin_feature_importance.csv')
        cluster_labels = pd.read_csv('bitcoin_cluster_labels.csv')
        anomaly_labels = pd.read_csv('bitcoin_anomaly_labels.csv')
        
        # Load association rules if available
        try:
            association_rules = pd.read_csv('bitcoin_association_rules.csv')
        except:
            association_rules = pd.DataFrame()
        
        return df_features, feature_importance, cluster_labels, anomaly_labels, association_rules
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

@st.cache_data
def prepare_ml_data(df_features):
    """Prepare data for machine learning"""
    # Select numerical features for ML
    exclude_columns = ['timestamp', 'open', 'high', 'low', 'price', 'volume', 'market_cap']
    feature_columns = [col for col in df_features.columns if col not in exclude_columns]
    numeric_columns = df_features[feature_columns].select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in feature_columns if col in numeric_columns]
    
    X = df_features[feature_columns].fillna(df_features[feature_columns].mean())
    
   # Create target variable
    df_features['next_day_price'] = df_features['price'].shift(-1)
    df_features['price_direction'] = (df_features['next_day_price'] > df_features['price']).astype(int)
    
    # Align X and y (Both must be N-1 rows)
    X = X.iloc[:-1]
    y_classification = df_features['price_direction'].iloc[:-1]
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
    
    return X_scaled, y_classification, feature_columns, scaler

def create_price_chart(df_features):
    """Create interactive price chart with technical indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Bitcoin Price with Technical Indicators', 
                       'Volume', 'RSI', 'MACD'),
        row_width=[0.4, 0.1, 0.1, 0.1]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df_features.index,
            open=df_features['open'],
            high=df_features['high'],
            low=df_features['low'],
            close=df_features['price'],
            name="BTC/USD"
        ),
        row=1, col=1
    )
    
    # Moving averages
    if 'sma_20' in df_features.columns:
        fig.add_trace(
            go.Scatter(
                x=df_features.index,
                y=df_features['sma_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
    
    if 'sma_50' in df_features.columns:
        fig.add_trace(
            go.Scatter(
                x=df_features.index,
                y=df_features['sma_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'bb_upper' in df_features.columns:
        fig.add_trace(
            go.Scatter(
                x=df_features.index,
                y=df_features['bb_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_features.index,
                y=df_features['bb_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=df_features.index,
            y=df_features['volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    # RSI
    if 'rsi' in df_features.columns:
        fig.add_trace(
            go.Scatter(
                x=df_features.index,
                y=df_features['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    if 'macd' in df_features.columns:
        fig.add_trace(
            go.Scatter(
                x=df_features.index,
                y=df_features['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=4, col=1
        )
        
        if 'macd_signal' in df_features.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_features.index,
                    y=df_features['macd_signal'],
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color='red', width=2)
                ),
                row=4, col=1
            )
    
    fig.update_layout(
        title='Bitcoin Market Analysis - Technical Indicators',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig

def create_clustering_chart(df_features, cluster_labels):
    """Create clustering visualization"""
    # Prepare data
    cluster_data = df_features.copy()
    min_length = min(len(cluster_data), len(cluster_labels))
    cluster_data = cluster_data.iloc[:min_length]
    cluster_data['cluster'] = cluster_labels['cluster'].iloc[:min_length].values
    
    fig = go.Figure()
    
    # Add scatter points for each cluster
    for cluster_id in cluster_data['cluster'].unique():
        cluster_subset = cluster_data[cluster_data['cluster'] == cluster_id]
        fig.add_trace(
            go.Scatter(
                x=cluster_subset['price'],
                y=cluster_subset['volume'],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(size=8, opacity=0.7),
                text=[f'Price: ${price:,.2f}<br>Volume: ${vol:,.0f}<br>Date: {date}' 
                      for price, vol, date in zip(cluster_subset['price'], 
                                                 cluster_subset['volume'], 
                                                 cluster_subset.index)],
                hovertemplate='%{text}<extra></extra>'
            )
        )
    
    fig.update_layout(
        title='Market Regime Clustering - Price vs Volume',
        xaxis_title='Price (USD)',
        yaxis_title='Volume',
        height=600
    )
    
    return fig

def create_anomaly_chart(df_features, anomaly_labels):
    """Create anomaly detection visualization"""
    # Prepare data
    anomaly_data = df_features.copy()
    min_length = min(len(anomaly_data), len(anomaly_labels))
    anomaly_data = anomaly_data.iloc[:min_length]
    anomaly_data['anomaly'] = anomaly_labels['anomaly'].iloc[:min_length].values
    
    fig = go.Figure()
    
    # Normal data
    normal_data = anomaly_data[anomaly_data['anomaly'] == 0]
    fig.add_trace(
        go.Scatter(
            x=normal_data.index,
            y=normal_data['price'],
            mode='lines',
            name='Normal',
            line=dict(color='blue', width=1),
            opacity=0.7
        )
    )
    
    # Anomaly data
    anomaly_dates = anomaly_data[anomaly_data['anomaly'] == 1]
    fig.add_trace(
        go.Scatter(
            x=anomaly_dates.index,
            y=anomaly_dates['price'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10),
            text=[f'Anomaly detected<br>Price: ${price:,.2f}<br>Date: {date}' 
                  for price, date in zip(anomaly_dates['price'], anomaly_dates.index)],
            hovertemplate='%{text}<extra></extra>'
        )
    )
    
    fig.update_layout(
        title='Bitcoin Price with Detected Anomalies',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=600
    )
    
    return fig

def create_feature_importance_chart(feature_importance, top_n=20):
    """Create feature importance visualization"""
    top_features = feature_importance.head(top_n)
    
    fig = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Most Important Features',
        xaxis_title='Feature Importance',
        yaxis_title='Features',
        height=600
    )
    
    return fig

def search_and_filter_data(df, search_query, date_range=None, column_filters=None):
    """Search and filter data based on query and filters"""
    filtered_df = df.copy()
    
    # Apply date range filter if provided
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        if isinstance(filtered_df.index, pd.DatetimeIndex):
            filtered_df = filtered_df[(filtered_df.index.date >= start_date) & 
                                     (filtered_df.index.date <= end_date)]
    
    # Apply column filters
    if column_filters:
        for column, filter_type, value in column_filters:
            if column in filtered_df.columns:
                if filter_type == 'greater_than' and value:
                    filtered_df = filtered_df[filtered_df[column] > float(value)]
                elif filter_type == 'less_than' and value:
                    filtered_df = filtered_df[filtered_df[column] < float(value)]
                elif filter_type == 'equals' and value:
                    filtered_df = filtered_df[filtered_df[column] == float(value)]
    
    return filtered_df

def explain_indicator(indicator_name):
    """Provide simple explanations for technical indicators"""
    explanations = {
        'price': 'The current trading price of Bitcoin in US dollars',
        'volume': 'The total amount of Bitcoin traded in a given time period',
        'rsi': 'Relative Strength Index (0-100). Below 30 = Oversold (buying opportunity), Above 70 = Overbought (selling opportunity)',
        'macd': 'Moving Average Convergence Divergence - Shows if price trend is strengthening or weakening',
        'sma_20': '20-day Simple Moving Average - Average price over last 20 days',
        'sma_50': '50-day Simple Moving Average - Average price over last 50 days',
        'bb_upper': 'Bollinger Band Upper - Price above this suggests overbought conditions',
        'bb_lower': 'Bollinger Band Lower - Price below this suggests oversold conditions',
        'price_volatility': 'How much the price is fluctuating - Higher means more unstable',
        'cluster': 'Market regime - Similar market conditions grouped together',
        'anomaly': 'Unusual market behavior that stands out from normal patterns'
    }
    return explanations.get(indicator_name.lower(), 'A market indicator that helps understand Bitcoin price movements')

def get_friendly_summary(df, anomaly_labels=None):
    """Create a friendly summary for non-technical users"""
    current_price = df['price'].iloc[-1]
    avg_price = df['price'].mean()
    
    # Determine market status
    if current_price > avg_price * 1.1:
        market_status = "Price is well above average"
        color = "success"
    elif current_price < avg_price * 0.9:
        market_status = "Price is below average"
        color = "warning"
    else:
        market_status = "Neutral - Price is near average"
        color = "info"
    
    # RSI status
    if 'rsi' in df.columns:
        current_rsi = df['rsi'].iloc[-1]
        if current_rsi > 70:
            rsi_status = "Overbought (Consider selling)"
        elif current_rsi < 30:
            rsi_status = "Oversold (Consider buying)"
        else:
            rsi_status = f"Normal ({current_rsi:.1f})"
    else:
        rsi_status = "N/A"
    
    # Anomaly count
    if anomaly_labels is not None:
        anomaly_count = anomaly_labels['anomaly'].sum()
        anomaly_status = f"{anomaly_count} unusual events detected"
    else:
        anomaly_status = "ℹ️ No anomaly data available"
    
    return {
        'market_status': market_status,
        'color': color,
        'rsi_status': rsi_status,
        'anomaly_status': anomaly_status,
        'current_price': current_price,
        'avg_price': avg_price
    }

def answer_question(question, df, cluster_labels=None, anomaly_labels=None):
    """Answer questions about the Bitcoin data"""
    question_lower = question.lower()
    
    # Price questions
    if 'highest' in question_lower and 'price' in question_lower:
        max_price = df['price'].max()
        max_date = df[df['price'] == max_price].index[0]
        return f"The highest price was ${max_price:,.2f} on {max_date.strftime('%Y-%m-%d')}"
    
    if 'lowest' in question_lower and 'price' in question_lower:
        min_price = df['price'].min()
        min_date = df[df['price'] == min_price].index[0]
        return f"The lowest price was ${min_price:,.2f} on {min_date.strftime('%Y-%m-%d')}"
    
    if 'current' in question_lower and 'price' in question_lower:
        current_price = df['price'].iloc[-1]
        return f"Current Bitcoin price is ${current_price:,.2f}"
    
    if 'average' in question_lower and 'price' in question_lower:
        avg_price = df['price'].mean()
        return f"Average price: ${avg_price:,.2f}"
    
    # Volume questions
    if 'volume' in question_lower and 'highest' in question_lower:
        max_vol = df['volume'].max()
        max_date = df[df['volume'] == max_vol].index[0]
        return f"Highest volume was ${max_vol:,.0f} on {max_date.strftime('%Y-%m-%d')}"
    
    if 'volume' in question_lower and 'average' in question_lower:
        avg_vol = df['volume'].mean()
        return f"📊 Average volume: ${avg_vol:,.0f}"
    
    # RSI questions
    if 'rsi' in question_lower and 'overbought' in question_lower:
        overbought = df[df['rsi'] > 70]
        if len(overbought) > 0:
            return f"⚠️ RSI was overbought (>70) on {len(overbought)} days"
        return "✅ No overbought conditions detected in the current data"
    
    if 'rsi' in question_lower and 'oversold' in question_lower:
        oversold = df[df['rsi'] < 30]
        if len(oversold) > 0:
            return f"🛒 RSI was oversold (<30) on {len(oversold)} days"
        return "✅ No oversold conditions detected in the current data"
    
    # Anomaly questions
    if 'anomaly' in question_lower or 'unusual' in question_lower:
        if anomaly_labels is not None:
            anomaly_count = anomaly_labels['anomaly'].sum()
            return f"Detected {anomaly_count} anomalies in the dataset"
        return "ℹ️ Anomaly detection has not been run yet"
    
    # Trend questions
    if 'trend' in question_lower or 'direction' in question_lower:
        recent_change = ((df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]) * 100
        if recent_change > 0:
            return f"Upward trend: Price increased by {recent_change:.2f}% in this period"
        else:
            return f"Downward trend: Price decreased by {abs(recent_change):.2f}% in this period"
    
    # Volatility questions
    if 'volatility' in question_lower or 'volatile' in question_lower:
        volatilities = df['price_volatility_7'].dropna()
        if len(volatilities) > 0:
            avg_vol = volatilities.mean()
            return f"📊 Average 7-day price volatility: {avg_vol:.4f}"
        return "ℹ️ Volatility data not available for the selected period"
    
    # General statistics
    if 'statistics' in question_lower or 'stats' in question_lower:
        stats_text = f"""
        Dataset Statistics:
        - Total data points: {len(df):,}
        - Price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}
        - Average price: ${df['price'].mean():,.2f}
        - Average volume: ${df['volume'].mean():,.0f}
        - Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}
        """
        return stats_text
    
    # Default response
    return "I can answer questions about: prices, volume, RSI, anomalies, trends, volatility, and statistics. Try asking 'What was the highest price?' or 'Show me anomaly statistics'"

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">₿ Bitcoin Market Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading Bitcoin analysis data...'):
        df_features, feature_importance, cluster_labels, anomaly_labels, association_rules = load_data()
    
    if df_features is None:
        st.error("Failed to load data. Please ensure all CSV files are available.")
        return
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    st.sidebar.caption("Use these controls to filter and explore your data")
    
    # Getting Started button
    with st.sidebar.expander("Quick Tips", expanded=False):
        st.markdown("""
        **Welcome!** This dashboard helps you analyze Bitcoin data easily.
        
        **Quick Tips:**
        1. Select a date range to focus on specific periods
        2. Choose an analysis type from the menu
        3. Use Search to find specific data points
        4. Ask Questions in plain English
        
        **For Beginners:**
        - Start with "Overview" to see general trends
        - Try "Ask Questions" to get simple answers
        - Use "Search & Explore" to look for specific data
        """)
    
    # Date range selector
    st.sidebar.subheader("Select Time Period")
    date_range = st.sidebar.date_input(
        "Choose the date range to analyze",
        value=(df_features.index.min().date(), df_features.index.max().date()),
        min_value=df_features.index.min().date(),
        max_value=df_features.index.max().date()
    )
    st.sidebar.caption("Select a shorter period for detailed analysis")
    
    # Filter data based on date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = df_features[(df_features.index.date >= start_date) & 
                                   (df_features.index.date <= end_date)]
    else:
        filtered_data = df_features
    
    # Analysis type selector
    st.sidebar.subheader("What to Explore")
    
    # Add descriptions for each analysis type
    analysis_options = {
        "Overview": "See everything at a glance (best for beginners)",
        "Price Analysis": "Detailed price trends and technical indicators",
        "Clustering Analysis": "Market patterns and regimes",
        "Anomaly Detection": "Unusual events in the market",
        "Feature Importance": "Which factors affect Bitcoin most",
        "Machine Learning": "AI predictions and patterns",
        "Search & Explore": "Find specific data easily",
        "Ask Questions": "Ask questions in plain English (recommended for beginners)"
    }
    
    # Display as selectbox with better descriptions
    analysis_type = st.sidebar.selectbox(
        "Select what you want to analyze",
        options=list(analysis_options.keys()),
        format_func=lambda x: analysis_options[x]
    )
    
    # Show what's currently selected
    st.sidebar.info(f"Selected: {analysis_type}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${filtered_data['price'].iloc[-1]:,.2f}",
            delta=f"{filtered_data['price_change_pct'].iloc[-1]:.2f}%" if 'price_change_pct' in filtered_data.columns else None
        )
    
    with col2:
        st.metric(
            label="Total Days",
            value=f"{len(filtered_data):,}"
        )
    
    with col3:
        if cluster_labels is not None:
            st.metric(
                label="Market Regimes",
                value=f"{len(cluster_labels['cluster'].unique())}"
            )
    
    with col4:
        if anomaly_labels is not None:
            anomaly_count = anomaly_labels['anomaly'].sum()
            st.metric(
                label="Anomalies Detected",
                value=f"{anomaly_count}"
            )
    
    # Main content based on analysis type
    if analysis_type == "Overview":
        st.subheader("Bitcoin Market Overview")
        
        # Get friendly summary
        summary = get_friendly_summary(filtered_data, anomaly_labels)
        
        # Display market summary in friendly terms
        st.markdown("### Quick Market Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <h4>Market Status</h4>
                <p style="font-size:1.2em;"><strong>{summary['market_status']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-box">
                <h4>Buy/Sell Indicator (RSI)</h4>
                <p style="font-size:1.2em;"><strong>{summary['rsi_status']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-box">
                <h4>Unusual Events</h4>
                <p style="font-size:1.2em;"><strong>{summary['anomaly_status']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Price chart with explanation
        st.markdown("### Price Trend & Technical Indicators")
        st.caption("Move your mouse over the chart to see details. Zoom by dragging to select an area.")
        
        fig = create_price_chart(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add legend for technical indicators
        with st.expander("ℹ️ What do these lines mean?"):
            st.markdown("""
            **Chart Colors Explained:**
            - **Orange Candle Sticks**: Open & Close prices for each day
            - **Orange Line (SMA 20)**: Average price over last 20 days - helps identify short-term trends
            - **Red Line (SMA 50)**: Average price over last 50 days - helps identify long-term trends
            - **Gray Dashed Lines**: Bollinger Bands - show when price is too high or too low
            - **Purple Line (RSI)**: Relative Strength Index - Below 30 = Oversold, Above 70 = Overbought
            - **Blue/Red Lines (MACD)**: Shows whether price momentum is increasing or decreasing
            """)
        
        st.markdown("---")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Statistics")
            price_stats = filtered_data['price'].describe()
            st.dataframe(price_stats)
        
        with col2:
            st.subheader("Volume Statistics")
            volume_stats = filtered_data['volume'].describe()
            st.dataframe(volume_stats)
    
    elif analysis_type == "Price Analysis":
        st.subheader("Price Analysis with Technical Indicators")
        
        # Price chart
        fig = create_price_chart(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicator analysis
        col1, col2 = st.columns(2)
        
        with col1:
            if 'rsi' in filtered_data.columns:
                st.subheader("RSI Analysis")
                st.caption("RSI (Relative Strength Index) tells you if Bitcoin is overbought or oversold")
                current_rsi = filtered_data['rsi'].iloc[-1]
                st.metric("Current RSI", f"{current_rsi:.2f}")
                
                if current_rsi > 70:
                    st.warning("Overbought - Price might be too high, consider selling")
                elif current_rsi < 30:
                    st.success("Oversold - Price might be too low, buying opportunity")
                else:
                    st.info("✅ **Normal** - RSI is balanced (30-70 range)")
        
        with col2:
            if 'bb_position' in filtered_data.columns:
                st.subheader("Bollinger Bands Analysis")
                st.caption("Bollinger Bands show if price is near extreme highs or lows")
                current_bb_pos = filtered_data['bb_position'].iloc[-1]
                st.metric("BB Position", f"{current_bb_pos:.2f}")
                
                if current_bb_pos > 0.8:
                    st.warning("Price too high - Near upper band, might drop")
                elif current_bb_pos < 0.2:
                    st.success("🛒 **Price too low** - Near lower band, might rise")
                else:
                    st.info("✅ **Normal range** - Price is balanced")
    
    elif analysis_type == "Clustering Analysis":
        st.subheader("Market Regime Clustering")
        st.info("What is Clustering? This groups similar market conditions together. For example, high volume days, low price days, etc. Each color represents a different market pattern.")
        
        if cluster_labels is not None:
            # Clustering chart
            fig = create_clustering_chart(filtered_data, cluster_labels)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Cluster Distribution")
                cluster_counts = cluster_labels['cluster'].value_counts()
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=[f'Cluster {i}' for i in cluster_counts.index],
                    values=cluster_counts.values,
                    hole=0.3
                )])
                fig_pie.update_layout(title="Market Regime Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("Cluster Details")
                for cluster_id in cluster_labels['cluster'].unique():
                    cluster_size = (cluster_labels['cluster'] == cluster_id).sum()
                    percentage = (cluster_size / len(cluster_labels)) * 100
                    st.write(f"**Cluster {cluster_id}**: {cluster_size} samples ({percentage:.1f}%)")
        else:
            st.warning("Clustering data not available. Please run the clustering analysis first.")
    
    elif analysis_type == "Anomaly Detection":
        st.subheader("Anomaly Detection Results")
        st.info("What are Anomalies? These are unusual events that stand out from normal patterns - like sudden price spikes or drops, unusual trading volumes, or unexpected market behavior. Red dots mark these unusual dates.")
        
        if anomaly_labels is not None:
            # Anomaly chart
            fig = create_anomaly_chart(filtered_data, anomaly_labels)
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Anomaly Statistics")
                total_anomalies = anomaly_labels['anomaly'].sum()
                anomaly_percentage = (total_anomalies / len(anomaly_labels)) * 100
                
                st.metric("Total Anomalies", total_anomalies)
                st.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
            
            with col2:
                st.subheader("🔍 Recent Anomalies")
                anomaly_data = filtered_data.copy()
                min_length = min(len(anomaly_data), len(anomaly_labels))
                anomaly_data = anomaly_data.iloc[:min_length]
                anomaly_data['anomaly'] = anomaly_labels['anomaly'].iloc[:min_length].values
                
                recent_anomalies = anomaly_data[anomaly_data['anomaly'] == 1].tail(5)
                if len(recent_anomalies) > 0:
                    st.dataframe(recent_anomalies[['price', 'volume', 'price_change_pct']])
                else:
                    st.info("No recent anomalies found")
        else:
            st.warning("Anomaly detection data not available. Please run the anomaly detection analysis first.")
    
    elif analysis_type == "Feature Importance":
        st.subheader("Feature Importance Analysis")
        
        if feature_importance is not None:
            # Feature importance chart
            top_n = st.slider("Number of top features to display", 5, 30, 15)
            fig = create_feature_importance_chart(feature_importance, top_n)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.subheader("Feature Importance Details")
            st.dataframe(feature_importance.head(20))
        else:
            st.warning("Feature importance data not available. Please run the machine learning analysis first.")
    
    elif analysis_type == "Machine Learning":
        st.subheader("Machine Learning Analysis")
        
        if st.button("Run Real-time Classification"):
            with st.spinner('Training Random Forest model...'):
                X_scaled, y_classification, feature_columns, scaler = prepare_ml_data(filtered_data)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_classification, test_size=0.2, random_state=42, stratify=y_classification
                )
                
                # Train model
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = rf_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Model Accuracy", f"{accuracy:.4f}")
                    
                    # Classification report
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred, target_names=['Price Down', 'Price Up'], output_dict=True)
                    st.json(report)
                
                with col2:
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted Down', 'Predicted Up'],
                        y=['Actual Down', 'Actual Up'],
                        colorscale='Blues'
                    ))
                    fig_cm.update_layout(title="Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)
    
    elif analysis_type == "Search & Explore":
        st.subheader("Search & Explore Data")
        
        st.info("How to Search: Type keywords like 'anomaly', 'high volume', or use the filters below to find specific data points. It's like searching a database!")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "🔎 Search by keyword",
                placeholder="Try: 'anomaly', 'volume', or leave empty and use filters below",
                help="Type keywords to find matching data. Leave empty to use filters only."
            )
        
        with col2:
            rows_per_page = st.selectbox("How many rows to show", [10, 25, 50, 100, 200], help="Choose how many results to display per page")
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters (for finding specific values)"):
            st.caption("💡 Use filters to find data where a value is greater than, less than, or equal to a specific number")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_column = st.selectbox(
                    "What to filter by",
                    ["None"] + list(filtered_data.select_dtypes(include=[np.number]).columns),
                    help="Choose what measurement to filter by (price, volume, etc.)"
                )
            
            with col2:
                filter_operator = st.selectbox(
                    "How to filter",
                    ["None", "Greater than (>)", "Less than (<)", "Equals (=)"],
                    help="Greater than (>): values above your number\nLess than (<): values below your number\nEquals (=): exact match"
                )
            
            with col3:
                if filter_operator != "None":
                    filter_value = st.number_input("Enter number", value=0.0, help="The value to compare against")
                    filter_enabled = True
                else:
                    filter_value = 0.0
                    filter_enabled = False
        
        # Apply filters
        display_data = filtered_data.copy()
        
        if filter_enabled and filter_column != "None":
            if filter_operator == "Greater than (>)":
                display_data = display_data[display_data[filter_column] > filter_value]
            elif filter_operator == "Less than (<)":
                display_data = display_data[display_data[filter_column] < filter_value]
            elif filter_operator == "Equals (=)":
                display_data = display_data[display_data[filter_column] == filter_value]
        
        # Display search results
        if search_query:
            # Simple text-based search
            st.info(f"🔍 Showing results for: '{search_query}'")
            
            # Highlight relevant data
            search_lower = search_query.lower()
            if search_lower in ['anomaly', 'anomalies'] and anomaly_labels is not None:
                # Show anomaly dates
                min_length = min(len(display_data), len(anomaly_labels))
                anomaly_data = display_data.iloc[:min_length].copy()
                anomaly_data['anomaly'] = anomaly_labels['anomaly'].iloc[:min_length].values
                display_data = anomaly_data[anomaly_data['anomaly'] == 1]
        
        # Display filtered data table
        if len(display_data) > 0:
            st.subheader(f"📋 Results: {len(display_data)} rows")
            
            # Select columns to display
            with st.expander("👁️ Select columns to display"):
                all_columns = list(display_data.columns)
                selected_columns = st.multiselect(
                    "Choose columns",
                    all_columns,
                    default=['price', 'volume', 'price_change_pct', 'rsi', 'macd'] if all(
                        c in all_columns for c in ['price', 'volume', 'price_change_pct', 'rsi', 'macd']
                    ) else all_columns[:10]
                )
            
            if selected_columns:
                display_data_filtered = display_data[selected_columns]
            else:
                display_data_filtered = display_data
            
            # Pagination
            total_rows = len(display_data_filtered)
            total_pages = (total_rows - 1) // rows_per_page + 1
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page_number = st.number_input(
                    f"Page (1-{total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    key="pagination"
                )
            
            # Display page
            start_idx = (page_number - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, total_rows)
            page_data = display_data_filtered.iloc[start_idx:end_idx]
            
            st.dataframe(
                page_data,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = page_data.to_csv(index=True)
            st.download_button(
                label="📥 Download current page as CSV",
                data=csv,
                file_name=f"bitcoin_search_results_page_{page_number}.csv",
                mime="text/csv"
            )
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            if 'price' in display_data.columns:
                with col1:
                    st.metric("Average Price", f"${display_data['price'].mean():,.2f}")
                with col2:
                    st.metric("Price Range", f"${display_data['price'].min():,.2f} - ${display_data['price'].max():,.2f}")
            
            if 'volume' in display_data.columns:
                with col3:
                    st.metric("Average Volume", f"${display_data['volume'].mean():,.0f}")
            
            with col4:
                st.metric("Date Range", f"{display_data.index.min().strftime('%Y-%m-%d')} to {display_data.index.max().strftime('%Y-%m-%d')}")
        else:
            st.warning("No data matches your search criteria. Try adjusting your filters.")
    
    elif analysis_type == "Ask Questions":
        st.subheader("Ask Questions About Your Data")
        
        st.info("Welcome! This is the easiest way to get insights. Just ask questions in plain English, like talking to a friend. No technical knowledge needed!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Price Questions")
            st.markdown("""
            - **"What was the highest price?"**
            - **"What is the current price?"**
            - **"Show me the average price"**
            - **"What is the lowest price?"**
            """)
        
        with col2:
            st.markdown("### Market Questions")
            st.markdown("""
            - **"Is Bitcoin overbought or oversold?"**
            - **"What is the price trend?"**
            - **"How volatile is Bitcoin?"**
            - **"Show me statistics"**
            """)
        
        st.markdown("### Ask Your Question")
        question = st.text_input(
            "Type your question in the box below (in plain English):",
            placeholder="Example: What was the highest price?",
            help="Just ask naturally! The system will understand."
        )
        
        # Quick question buttons
        st.markdown("**Or click a quick question:**")
        col1, col2, col3, col4 = st.columns(4)
        
        quick_questions = [
            "What is the current price?",
            "What was the highest price?",
            "What is the price trend?",
            "Show me statistics"
        ]
        
        for i, q in enumerate(quick_questions):
            with [col1, col2, col3, col4][i]:
                if st.button(f"{q}", key=f"quick_{i}"):
                    question = q
        
        st.markdown("---")
        
        # Ask button
        if st.button("Get Answer"):
            if question:
                with st.spinner("Analyzing data..."):
                    answer = answer_question(question, filtered_data, cluster_labels, anomaly_labels)
                    st.success(answer)
                    st.info("Tip: You can use the date range filter in the sidebar to analyze specific time periods.")
            else:
                st.warning("Please enter a question.")
        
        # Quick insights panel
        st.markdown("---")
        st.subheader("Quick Insights")
        
        if st.button("Generate Quick Insights"):
            insights = []
            
            # Price insights
            price_change_pct = ((filtered_data['price'].iloc[-1] - filtered_data['price'].iloc[0]) / 
                               filtered_data['price'].iloc[0]) * 100
            if abs(price_change_pct) > 5:
                direction = "increased" if price_change_pct > 0 else "decreased"
                insights.append(f"📈 Price {direction} by {abs(price_change_pct):.2f}% over the selected period")
            
            # Volume insights
            avg_volume = filtered_data['volume'].mean()
            recent_volume = filtered_data['volume'].iloc[-5:].mean()
            if recent_volume > avg_volume * 1.2:
                insights.append(f"Recent volume is {(recent_volume/avg_volume - 1)*100:.1f}% above average")
            elif recent_volume < avg_volume * 0.8:
                insights.append(f"Recent volume is {(1 - recent_volume/avg_volume)*100:.1f}% below average")
            
            # Highlighting anomalies
            if anomaly_labels is not None:
                anomaly_count = anomaly_labels['anomaly'].sum()
                if anomaly_count > 0:
                    insights.append(f"🚨 {anomaly_count} anomaly(ies) detected in the dataset")
            
            # RSI insights
            if 'rsi' in filtered_data.columns:
                current_rsi = filtered_data['rsi'].iloc[-1]
                if current_rsi > 70:
                    insights.append(f"⚠️ Current RSI ({current_rsi:.1f}) indicates overbought conditions")
                elif current_rsi < 30:
                    insights.append(f"🛒 Current RSI ({current_rsi:.1f}) indicates oversold conditions")
            
            if insights:
                for insight in insights:
                    st.info(insight)
            else:
                st.info("ℹ️ No significant insights found for the selected period.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Bitcoin Market Analysis Dashboard | Built with Streamlit</p>
        <p>Data: CoinGecko API | Analysis: Python & Scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

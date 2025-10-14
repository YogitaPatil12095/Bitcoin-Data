"""
Bitcoin Market Analysis - Interactive Streamlit Dashboard
=======================================================

This Streamlit application provides an interactive dashboard for Bitcoin market analysis,
integrating all the data mining algorithms and visualizations from the existing project.
"""

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
    y_classification = df_features['price_direction'].dropna()
    
    # Align X and y
    X = X.iloc[:-1]
    
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
    st.sidebar.title("🔧 Dashboard Controls")
    
    # Date range selector
    st.sidebar.subheader("📅 Date Range")
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(df_features.index.min().date(), df_features.index.max().date()),
        min_value=df_features.index.min().date(),
        max_value=df_features.index.max().date()
    )
    
    # Filter data based on date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = df_features[(df_features.index.date >= start_date) & 
                                   (df_features.index.date <= end_date)]
    else:
        filtered_data = df_features
    
    # Analysis type selector
    st.sidebar.subheader("📊 Analysis Type")
    analysis_type = st.sidebar.selectbox(
        "Choose analysis to display",
        ["Overview", "Price Analysis", "Clustering Analysis", "Anomaly Detection", "Feature Importance", "Machine Learning"]
    )
    
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
        st.subheader("📈 Bitcoin Market Overview")
        
        # Price chart
        fig = create_price_chart(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Price Statistics")
            price_stats = filtered_data['price'].describe()
            st.dataframe(price_stats)
        
        with col2:
            st.subheader("📊 Volume Statistics")
            volume_stats = filtered_data['volume'].describe()
            st.dataframe(volume_stats)
    
    elif analysis_type == "Price Analysis":
        st.subheader("📈 Price Analysis with Technical Indicators")
        
        # Price chart
        fig = create_price_chart(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicator analysis
        col1, col2 = st.columns(2)
        
        with col1:
            if 'rsi' in filtered_data.columns:
                st.subheader("🔍 RSI Analysis")
                current_rsi = filtered_data['rsi'].iloc[-1]
                st.metric("Current RSI", f"{current_rsi:.2f}")
                
                if current_rsi > 70:
                    st.warning("RSI indicates overbought conditions")
                elif current_rsi < 30:
                    st.success("RSI indicates oversold conditions")
                else:
                    st.info("RSI is in neutral territory")
        
        with col2:
            if 'bb_position' in filtered_data.columns:
                st.subheader("📊 Bollinger Bands Analysis")
                current_bb_pos = filtered_data['bb_position'].iloc[-1]
                st.metric("BB Position", f"{current_bb_pos:.2f}")
                
                if current_bb_pos > 0.8:
                    st.warning("Price near upper Bollinger Band")
                elif current_bb_pos < 0.2:
                    st.success("Price near lower Bollinger Band")
                else:
                    st.info("Price within normal Bollinger Band range")
    
    elif analysis_type == "Clustering Analysis":
        st.subheader("🎯 Market Regime Clustering")
        
        if cluster_labels is not None:
            # Clustering chart
            fig = create_clustering_chart(filtered_data, cluster_labels)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Cluster Distribution")
                cluster_counts = cluster_labels['cluster'].value_counts()
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=[f'Cluster {i}' for i in cluster_counts.index],
                    values=cluster_counts.values,
                    hole=0.3
                )])
                fig_pie.update_layout(title="Market Regime Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("📋 Cluster Details")
                for cluster_id in cluster_labels['cluster'].unique():
                    cluster_size = (cluster_labels['cluster'] == cluster_id).sum()
                    percentage = (cluster_size / len(cluster_labels)) * 100
                    st.write(f"**Cluster {cluster_id}**: {cluster_size} samples ({percentage:.1f}%)")
        else:
            st.warning("Clustering data not available. Please run the clustering analysis first.")
    
    elif analysis_type == "Anomaly Detection":
        st.subheader("🚨 Anomaly Detection Results")
        
        if anomaly_labels is not None:
            # Anomaly chart
            fig = create_anomaly_chart(filtered_data, anomaly_labels)
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Anomaly Statistics")
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
        st.subheader("🔍 Feature Importance Analysis")
        
        if feature_importance is not None:
            # Feature importance chart
            top_n = st.slider("Number of top features to display", 5, 30, 15)
            fig = create_feature_importance_chart(feature_importance, top_n)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.subheader("📋 Feature Importance Details")
            st.dataframe(feature_importance.head(20))
        else:
            st.warning("Feature importance data not available. Please run the machine learning analysis first.")
    
    elif analysis_type == "Machine Learning":
        st.subheader("🤖 Machine Learning Analysis")
        
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
                    st.subheader("📊 Classification Report")
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

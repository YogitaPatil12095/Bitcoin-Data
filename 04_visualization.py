"""
Bitcoin Market Analysis - Step 5: Comprehensive Visualization
============================================================

This script creates comprehensive visualizations for all data mining results:
1. Candlestick chart with technical indicators
2. Clustering visualization
3. Anomaly detection visualization
4. Feature importance plots
5. Association rules network
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

def load_all_data():
    """
    Load all processed data and results
    
    Returns:
        tuple: All loaded data and results
    """
    print("📂 Loading all data and results...")
    
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
        print("✅ Association rules loaded")
    except:
        association_rules = pd.DataFrame()
        print("⚠️  No association rules file found")
    
    print("✅ All data loaded successfully!")
    return df_features, feature_importance, cluster_labels, anomaly_labels, association_rules

def create_candlestick_chart(df_features):
    """
    Create candlestick chart with technical indicators
    """
    print("📊 Creating candlestick chart with technical indicators...")
    
    # Create subplots
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
    
    # MACD Histogram
    colors = ['green' if val >= 0 else 'red' for val in df_features['macd_histogram']]
    fig.add_trace(
        go.Bar(
            x=df_features.index,
            y=df_features['macd_histogram'],
            name='MACD Histogram',
            marker_color=colors
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Bitcoin Market Analysis - Technical Indicators',
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=True
    )
    
    # Save the plot
    fig.write_html('bitcoin_candlestick_chart.html')
    fig.write_image('bitcoin_candlestick_chart.png', width=1200, height=1000)
    
    print("✅ Candlestick chart saved as 'bitcoin_candlestick_chart.html' and '.png'")
    return fig

def create_clustering_visualization(df_features, cluster_labels):
    """
    Create clustering visualization
    """
    print("📊 Creating clustering visualization...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bitcoin Market Regime Clustering Analysis', fontsize=16, fontweight='bold')
    
    # Prepare data - align lengths
    cluster_data = df_features.copy()
    # Ensure we have the same number of rows
    min_length = min(len(cluster_data), len(cluster_labels))
    cluster_data = cluster_data.iloc[:min_length]
    cluster_data['cluster'] = cluster_labels['cluster'].iloc[:min_length].values
    
    # 1. Price vs Volume colored by cluster
    scatter = axes[0, 0].scatter(
        cluster_data['price'], 
        cluster_data['volume'],
        c=cluster_data['cluster'],
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    axes[0, 0].set_xlabel('Price (USD)')
    axes[0, 0].set_ylabel('Volume')
    axes[0, 0].set_title('Price vs Volume by Market Regime')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
    
    # 2. RSI vs Price Change colored by cluster
    scatter2 = axes[0, 1].scatter(
        cluster_data['rsi'], 
        cluster_data['price_change_pct'],
        c=cluster_data['cluster'],
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    axes[0, 1].set_xlabel('RSI')
    axes[0, 1].set_ylabel('Price Change (%)')
    axes[0, 1].set_title('RSI vs Price Change by Market Regime')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')
    
    # 3. Time series of price with cluster colors
    for cluster_id in cluster_data['cluster'].unique():
        cluster_subset = cluster_data[cluster_data['cluster'] == cluster_id]
        axes[1, 0].scatter(
            cluster_subset.index,
            cluster_subset['price'],
            label=f'Cluster {cluster_id}',
            alpha=0.7,
            s=30
        )
    
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Price (USD)')
    axes[1, 0].set_title('Price Timeline by Market Regime')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cluster distribution pie chart
    cluster_counts = cluster_data['cluster'].value_counts()
    cluster_labels_text = [f'Cluster {i}' for i in cluster_counts.index]
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
    
    wedges, texts, autotexts = axes[1, 1].pie(
        cluster_counts.values,
        labels=cluster_labels_text,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    axes[1, 1].set_title('Market Regime Distribution')
    
    plt.tight_layout()
    plt.savefig('bitcoin_clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Clustering visualization saved as 'bitcoin_clustering_analysis.png'")

def create_anomaly_visualization(df_features, anomaly_labels):
    """
    Create anomaly detection visualization
    """
    print("📊 Creating anomaly detection visualization...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Bitcoin Anomaly Detection Analysis', fontsize=16, fontweight='bold')
    
    # Prepare data - align lengths
    anomaly_data = df_features.copy()
    # Ensure we have the same number of rows
    min_length = min(len(anomaly_data), len(anomaly_labels))
    anomaly_data = anomaly_data.iloc[:min_length]
    anomaly_data['anomaly'] = anomaly_labels['anomaly'].iloc[:min_length].values
    anomaly_data['anomaly'] = anomaly_data['anomaly'].astype(bool)
    
    # 1. Price with anomalies highlighted
    normal_dates = anomaly_data[~anomaly_data['anomaly']].index
    anomaly_dates = anomaly_data[anomaly_data['anomaly']].index
    
    axes[0].plot(normal_dates, anomaly_data.loc[normal_dates, 'price'], 
                'b-', alpha=0.7, label='Normal', linewidth=1)
    axes[0].scatter(anomaly_dates, anomaly_data.loc[anomaly_dates, 'price'], 
                   color='red', s=100, label='Anomalies', zorder=5)
    
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price (USD)')
    axes[0].set_title('Bitcoin Price with Detected Anomalies')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Volume with anomalies highlighted
    axes[1].plot(normal_dates, anomaly_data.loc[normal_dates, 'volume'], 
                'g-', alpha=0.7, label='Normal', linewidth=1)
    axes[1].scatter(anomaly_dates, anomaly_data.loc[anomaly_dates, 'volume'], 
                   color='red', s=100, label='Anomalies', zorder=5)
    
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Volume')
    axes[1].set_title('Bitcoin Volume with Detected Anomalies')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Price change percentage with anomalies highlighted
    axes[2].plot(normal_dates, anomaly_data.loc[normal_dates, 'price_change_pct'], 
                'purple', alpha=0.7, label='Normal', linewidth=1)
    axes[2].scatter(anomaly_dates, anomaly_data.loc[anomaly_dates, 'price_change_pct'], 
                   color='red', s=100, label='Anomalies', zorder=5)
    
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Price Change (%)')
    axes[2].set_title('Bitcoin Price Change with Detected Anomalies')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bitcoin_anomaly_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Anomaly detection visualization saved as 'bitcoin_anomaly_detection.png'")

def create_feature_importance_plot(feature_importance):
    """
    Create feature importance visualization
    """
    print("📊 Creating feature importance plot...")
    
    # Get top 20 features
    top_features = feature_importance.head(20)
    
    # Create horizontal bar plot
    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    
    # Customize the plot
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 20 Most Important Features for Price Direction Prediction', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('bitcoin_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Feature importance plot saved as 'bitcoin_feature_importance.png'")

def create_correlation_heatmap(df_features):
    """
    Create correlation heatmap of key technical indicators
    """
    print("📊 Creating correlation heatmap...")
    
    # Select key technical indicators
    key_indicators = [
        'price', 'price_change_pct', 'volume', 'volume_ratio',
        'rsi', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'macd', 'macd_signal', 'bb_position', 'bb_width'
    ]
    
    # Filter available indicators
    available_indicators = [col for col in key_indicators if col in df_features.columns]
    
    # Calculate correlation matrix
    correlation_matrix = df_features[available_indicators].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={'shrink': 0.8}
    )
    
    plt.title('Correlation Matrix of Bitcoin Technical Indicators', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bitcoin_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Correlation heatmap saved as 'bitcoin_correlation_heatmap.png'")

def create_association_rules_network(association_rules):
    """
    Create association rules network visualization (if rules exist)
    """
    if len(association_rules) == 0:
        print("⚠️  No association rules to visualize")
        return
    
    print("📊 Creating association rules network...")
    
    # Get top 20 rules by confidence
    top_rules = association_rules.head(20)
    
    # Create a simple visualization of rule strengths
    plt.figure(figsize=(15, 8))
    
    # Plot confidence vs support
    scatter = plt.scatter(
        top_rules['support'],
        top_rules['confidence'],
        s=top_rules['lift'] * 100,  # Size based on lift
        alpha=0.7,
        c=top_rules['lift'],
        cmap='viridis'
    )
    
    plt.xlabel('Support', fontsize=12)
    plt.ylabel('Confidence', fontsize=12)
    plt.title('Association Rules: Support vs Confidence (Size = Lift)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Lift')
    plt.grid(True, alpha=0.3)
    
    # Add rule numbers as annotations
    for i, (_, rule) in enumerate(top_rules.iterrows()):
        plt.annotate(
            f'R{i+1}',
            (rule['support'], rule['confidence']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    plt.tight_layout()
    plt.savefig('bitcoin_association_rules.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Association rules visualization saved as 'bitcoin_association_rules.png'")

def create_summary_dashboard(df_features, cluster_labels, anomaly_labels, feature_importance):
    """
    Create a comprehensive summary dashboard
    """
    print("📊 Creating comprehensive summary dashboard...")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define the grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Price timeline with clusters
    ax1 = fig.add_subplot(gs[0, :2])
    cluster_data = df_features.copy()
    # Ensure we have the same number of rows
    min_length = min(len(cluster_data), len(cluster_labels))
    cluster_data = cluster_data.iloc[:min_length]
    cluster_data['cluster'] = cluster_labels['cluster'].iloc[:min_length].values
    
    for cluster_id in cluster_data['cluster'].unique():
        cluster_subset = cluster_data[cluster_data['cluster'] == cluster_id]
        ax1.plot(cluster_subset.index, cluster_subset['price'], 
                label=f'Cluster {cluster_id}', alpha=0.8, linewidth=2)
    
    ax1.set_title('Bitcoin Price Timeline by Market Regime', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Price timeline with anomalies
    ax2 = fig.add_subplot(gs[0, 2:])
    anomaly_data = df_features.copy()
    # Ensure we have the same number of rows
    min_length = min(len(anomaly_data), len(anomaly_labels))
    anomaly_data = anomaly_data.iloc[:min_length]
    anomaly_data['anomaly'] = anomaly_labels['anomaly'].iloc[:min_length].values
    
    ax2.plot(anomaly_data.index, anomaly_data['price'], 'b-', alpha=0.7, linewidth=1)
    anomaly_dates = anomaly_data[anomaly_data['anomaly'] == 1].index
    ax2.scatter(anomaly_dates, anomaly_data.loc[anomaly_dates, 'price'], 
               color='red', s=50, label='Anomalies', zorder=5)
    
    ax2.set_title('Bitcoin Price with Detected Anomalies', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price (USD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature importance (top 10)
    ax3 = fig.add_subplot(gs[1, :2])
    top_10_features = feature_importance.head(10)
    bars = ax3.barh(range(len(top_10_features)), top_10_features['importance'], color='lightcoral')
    ax3.set_yticks(range(len(top_10_features)))
    ax3.set_yticklabels(top_10_features['feature'], fontsize=10)
    ax3.set_xlabel('Importance')
    ax3.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. RSI with overbought/oversold levels
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.plot(df_features.index, df_features['rsi'], 'purple', linewidth=2)
    ax4.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax4.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax4.fill_between(df_features.index, 70, 100, alpha=0.2, color='red')
    ax4.fill_between(df_features.index, 0, 30, alpha=0.2, color='green')
    ax4.set_title('RSI with Overbought/Oversold Levels', fontsize=14, fontweight='bold')
    ax4.set_ylabel('RSI')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Volume analysis
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.plot(df_features.index, df_features['volume'], 'lightblue', linewidth=1)
    ax5.fill_between(df_features.index, 0, df_features['volume'], alpha=0.3, color='lightblue')
    ax5.set_title('Bitcoin Trading Volume', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Volume')
    ax5.grid(True, alpha=0.3)
    
    # 6. Bollinger Bands
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.plot(df_features.index, df_features['price'], 'black', linewidth=2, label='Price')
    ax6.plot(df_features.index, df_features['bb_upper'], 'gray', linestyle='--', alpha=0.7)
    ax6.plot(df_features.index, df_features['bb_lower'], 'gray', linestyle='--', alpha=0.7)
    ax6.plot(df_features.index, df_features['bb_middle'], 'orange', linewidth=1, label='BB Middle')
    ax6.fill_between(df_features.index, df_features['bb_lower'], df_features['bb_upper'], 
                    alpha=0.1, color='gray')
    ax6.set_title('Bollinger Bands', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Price (USD)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. MACD
    ax7 = fig.add_subplot(gs[3, :2])
    ax7.plot(df_features.index, df_features['macd'], 'blue', linewidth=2, label='MACD')
    ax7.plot(df_features.index, df_features['macd_signal'], 'red', linewidth=2, label='Signal')
    colors = ['green' if val >= 0 else 'red' for val in df_features['macd_histogram']]
    ax7.bar(df_features.index, df_features['macd_histogram'], color=colors, alpha=0.7, label='Histogram')
    ax7.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax7.set_title('MACD Indicator', fontsize=14, fontweight='bold')
    ax7.set_ylabel('MACD')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Market statistics
    ax8 = fig.add_subplot(gs[3, 2:])
    ax8.axis('off')
    
    # Calculate statistics
    total_days = len(df_features)
    price_change = ((df_features['price'].iloc[-1] - df_features['price'].iloc[0]) / df_features['price'].iloc[0]) * 100
    max_price = df_features['price'].max()
    min_price = df_features['price'].min()
    avg_volume = df_features['volume'].mean() / 1e9  # Convert to billions
    
    anomaly_count = anomaly_labels['anomaly'].sum()
    cluster_counts = cluster_labels['cluster'].value_counts()
    
    stats_text = f"""
    📊 BITCOIN MARKET ANALYSIS SUMMARY
    {'='*50}
    
    📅 Analysis Period: {df_features.index[0].strftime('%Y-%m-%d')} to {df_features.index[-1].strftime('%Y-%m-%d')}
    📈 Total Days Analyzed: {total_days:,}
    
    💰 PRICE ANALYSIS:
    • Starting Price: ${df_features['price'].iloc[0]:,.2f}
    • Ending Price: ${df_features['price'].iloc[-1]:,.2f}
    • Total Change: {price_change:+.2f}%
    • Highest Price: ${max_price:,.2f}
    • Lowest Price: ${min_price:,.2f}
    
    📊 VOLUME ANALYSIS:
    • Average Daily Volume: ${avg_volume:.2f}B
    • Max Volume: ${df_features['volume'].max()/1e9:.2f}B
    • Min Volume: ${df_features['volume'].min()/1e9:.2f}B
    
    🔍 DATA MINING RESULTS:
    • Anomalies Detected: {anomaly_count} ({anomaly_count/total_days*100:.1f}%)
    • Market Regimes: {len(cluster_counts)}
    • Most Common Regime: Cluster {cluster_counts.index[0]} ({cluster_counts.iloc[0]/total_days*100:.1f}%)
    """
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Bitcoin Market Analysis - Comprehensive Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig('bitcoin_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Comprehensive dashboard saved as 'bitcoin_comprehensive_dashboard.png'")

def main():
    """
    Main function to create all visualizations
    """
    print("🎨 Bitcoin Market Analysis - Comprehensive Visualization")
    print("=" * 70)
    
    # Load all data
    df_features, feature_importance, cluster_labels, anomaly_labels, association_rules = load_all_data()
    
    # Create all visualizations
    print("\n📊 Creating visualizations...")
    
    # 1. Candlestick chart
    candlestick_fig = create_candlestick_chart(df_features)
    
    # 2. Clustering visualization
    create_clustering_visualization(df_features, cluster_labels)
    
    # 3. Anomaly detection visualization
    create_anomaly_visualization(df_features, anomaly_labels)
    
    # 4. Feature importance plot
    create_feature_importance_plot(feature_importance)
    
    # 5. Correlation heatmap
    create_correlation_heatmap(df_features)
    
    # 6. Association rules network (if available)
    create_association_rules_network(association_rules)
    
    # 7. Comprehensive dashboard
    create_summary_dashboard(df_features, cluster_labels, anomaly_labels, feature_importance)
    
    # Final summary
    print(f"\n🎉 All Visualizations Created Successfully!")
    print("=" * 60)
    print(f"📊 Generated Files:")
    print(f"   • bitcoin_candlestick_chart.html/.png - Interactive candlestick chart")
    print(f"   • bitcoin_clustering_analysis.png - Market regime clustering")
    print(f"   • bitcoin_anomaly_detection.png - Anomaly detection results")
    print(f"   • bitcoin_feature_importance.png - Feature importance analysis")
    print(f"   • bitcoin_correlation_heatmap.png - Technical indicator correlations")
    print(f"   • bitcoin_association_rules.png - Association rules network")
    print(f"   • bitcoin_comprehensive_dashboard.png - Complete analysis dashboard")
    
    print(f"\n✅ Bitcoin Market Analysis Project Completed Successfully!")
    print(f"   🎯 All 5 steps completed: Data Loading → Preprocessing → Feature Engineering → Data Mining → Visualization")
    
    return True

if __name__ == "__main__":
    success = main()

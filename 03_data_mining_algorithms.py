"""
Bitcoin Market Analysis - Step 4: Data Mining Algorithms
======================================================

This script implements four key data mining algorithms:
1. Classification: Random Forest for price direction prediction
2. Clustering: K-Means for market regime identification
3. Anomaly Detection: Isolation Forest for unusual market events
4. Association Rules: Apriori for pattern discovery
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import silhouette_score, adjusted_rand_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """
    Load the processed Bitcoin data and features
    
    Returns:
        tuple: (df_features, X_scaled, y_classification, feature_columns)
    """
    print("📂 Loading processed Bitcoin data...")
    
    # Load main dataset
    df_features = pd.read_csv('bitcoin_processed_data.csv')
    df_features['datetime'] = pd.to_datetime(df_features['datetime'])
    df_features.set_index('datetime', inplace=True)
    
    # Load scaled features and target
    X_scaled = pd.read_csv('bitcoin_features_scaled.csv', index_col=0)
    y_classification = pd.read_csv('bitcoin_target_classification.csv', index_col=0).squeeze()
    
    # Align X and y (remove last row from y since we can't predict it)
    y_classification = y_classification.iloc[:-1]
    
    print(f"   • Aligned feature matrix: {X_scaled.shape}")
    print(f"   • Aligned target variable: {len(y_classification)} samples")
    
    # Load feature info
    feature_info = pd.read_csv('bitcoin_feature_info.csv')
    feature_columns = feature_info['feature_name'].tolist()
    
    print(f"✅ Data loaded successfully!")
    print(f"   • Dataset shape: {df_features.shape}")
    print(f"   • Feature matrix: {X_scaled.shape}")
    print(f"   • Target variable: {len(y_classification)} samples")
    
    return df_features, X_scaled, y_classification, feature_columns

def classification_random_forest(X_scaled, y_classification, feature_columns):
    """
    Classification Task: Predict next day's price direction using Random Forest
    
    Args:
        X_scaled (pd.DataFrame): Scaled feature matrix
        y_classification (pd.Series): Target variable (price direction)
        feature_columns (list): List of feature names
        
    Returns:
        tuple: (model, X_test, y_test, y_pred, feature_importance)
    """
    print("\n🎯 Task 1: Classification - Price Direction Prediction")
    print("=" * 60)
    
    # Split data: 80% training, 20% testing
    print("📊 Splitting data: 80% training, 20% testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_classification, test_size=0.2, random_state=42, stratify=y_classification
    )
    
    print(f"   • Training set: {len(X_train)} samples")
    print(f"   • Testing set: {len(X_test)} samples")
    print(f"   • Training class distribution: {y_train.value_counts().to_dict()}")
    
    # Hyperparameter tuning for Random Forest
    print("🔧 Optimizing Random Forest parameters...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Use a subset for faster grid search
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf_base, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"✅ Best parameters found: {best_params}")
    
    # Train final model with best parameters
    print("🚀 Training final Random Forest model...")
    rf_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📈 Model Performance:")
    print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • AUC Score: {auc_score:.4f}")
    
    # Cross-validation score
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"   • CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # Detailed classification report
    print(f"\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Price Down', 'Price Up']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print(f"   • True Negatives: {cm[0,0]} (Correctly predicted price down)")
    print(f"   • False Positives: {cm[0,1]} (Incorrectly predicted price up)")
    print(f"   • False Negatives: {cm[1,0]} (Incorrectly predicted price down)")
    print(f"   • True Positives: {cm[1,1]} (Correctly predicted price up)")
    
    return rf_model, X_test, y_test, y_pred, feature_importance

def clustering_kmeans(X_scaled, feature_columns):
    """
    Clustering Task: Identify market regimes using K-Means
    
    Args:
        X_scaled (pd.DataFrame): Scaled feature matrix
        feature_columns (list): List of feature names
        
    Returns:
        tuple: (model, labels, cluster_centers)
    """
    print("\n🎯 Task 2: Clustering - Market Regime Identification")
    print("=" * 60)
    
    # Select key features for clustering (price and volume related)
    clustering_features = [
        'price_change_pct', 'volume_ratio', 'rsi', 'bb_position',
        'price_volatility_7', 'volume_volatility_7', 'macd'
    ]
    
    # Filter features that exist in our dataset
    available_features = [f for f in clustering_features if f in feature_columns]
    if len(available_features) < 4:
        # Fallback to first few numeric features
        available_features = feature_columns[:10]
    
    print(f"📊 Using {len(available_features)} features for clustering:")
    for feature in available_features:
        print(f"   • {feature}")
    
    X_clustering = X_scaled[available_features]
    
    # Determine optimal number of clusters using elbow method and silhouette score
    print("\n🔍 Determining optimal number of clusters...")
    max_clusters = min(8, len(X_clustering) // 10)  # Don't exceed reasonable limit
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_clustering)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_clustering, cluster_labels))
    
    # Choose optimal k based on silhouette score
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"✅ Optimal number of clusters: {optimal_k}")
    print(f"   • Silhouette score: {max(silhouette_scores):.4f}")
    
    # Train final K-Means model
    print(f"🚀 Training K-Means with {optimal_k} clusters...")
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(X_clustering)
    
    # Analyze clusters
    silhouette_avg = silhouette_score(X_clustering, cluster_labels)
    print(f"\n📈 Clustering Performance:")
    print(f"   • Silhouette Score: {silhouette_avg:.4f}")
    print(f"   • Inertia: {kmeans_model.inertia_:.2f}")
    
    # Cluster analysis
    cluster_centers = kmeans_model.cluster_centers_
    cluster_df = pd.DataFrame(cluster_centers, columns=available_features)
    cluster_df.index.name = 'cluster'
    
    print(f"\n📊 Cluster Analysis:")
    for i in range(optimal_k):
        cluster_size = np.sum(cluster_labels == i)
        percentage = (cluster_size / len(cluster_labels)) * 100
        print(f"   • Cluster {i}: {cluster_size} samples ({percentage:.1f}%)")
        
        # Identify cluster characteristics
        cluster_center = cluster_df.iloc[i]
        top_features = cluster_center.abs().nlargest(3)
        print(f"     Key characteristics: {', '.join([f'{f}={cluster_center[f]:.3f}' for f in top_features.index])}")
    
    # Define market regimes based on cluster characteristics
    market_regimes = []
    for i in range(optimal_k):
        center = cluster_df.iloc[i]
        
        # Simple rule-based regime classification
        if center.get('price_change_pct', 0) > 0.5:
            regime = "Bull Market"
        elif center.get('price_change_pct', 0) < -0.5:
            regime = "Bear Market"
        elif center.get('rsi', 0.5) > 0.7:
            regime = "Overbought"
        elif center.get('rsi', 0.5) < 0.3:
            regime = "Oversold"
        elif center.get('price_volatility_7', 0.5) > 0.7:
            regime = "High Volatility"
        else:
            regime = "Stable Market"
        
        market_regimes.append(regime)
    
    print(f"\n🏷️  Market Regime Classification:")
    for i, regime in enumerate(market_regimes):
        cluster_size = np.sum(cluster_labels == i)
        percentage = (cluster_size / len(cluster_labels)) * 100
        print(f"   • Cluster {i} ({regime}): {cluster_size} samples ({percentage:.1f}%)")
    
    return kmeans_model, cluster_labels, cluster_df

def anomaly_detection_isolation_forest(X_scaled, feature_columns):
    """
    Anomaly Detection Task: Identify unusual market events using Isolation Forest
    
    Args:
        X_scaled (pd.DataFrame): Scaled feature matrix
        feature_columns (list): List of feature names
        
    Returns:
        tuple: (model, anomaly_labels, anomaly_scores)
    """
    print("\n🎯 Task 3: Anomaly Detection - Unusual Market Events")
    print("=" * 60)
    
    # Select key features for anomaly detection
    anomaly_features = [
        'price_change_pct', 'volume_ratio', 'rsi', 'bb_position',
        'price_volatility_7', 'volume_volatility_7', 'macd_histogram'
    ]
    
    # Filter features that exist in our dataset
    available_features = [f for f in anomaly_features if f in feature_columns]
    if len(available_features) < 3:
        # Fallback to first few numeric features
        available_features = feature_columns[:8]
    
    print(f"📊 Using {len(available_features)} features for anomaly detection:")
    for feature in available_features:
        print(f"   • {feature}")
    
    X_anomaly = X_scaled[available_features]
    
    # Train Isolation Forest with 5% contamination (appropriate for financial data)
    print("🚀 Training Isolation Forest model...")
    contamination_rate = 0.05  # 5% of data considered anomalous
    iso_forest = IsolationForest(
        contamination=contamination_rate,
        random_state=42,
        n_estimators=100
    )
    
    anomaly_labels = iso_forest.fit_predict(X_anomaly)
    anomaly_scores = iso_forest.score_samples(X_anomaly)
    
    # Convert labels: -1 (anomaly) to 1, 1 (normal) to 0
    anomaly_labels = (anomaly_labels == -1).astype(int)
    
    n_anomalies = np.sum(anomaly_labels)
    anomaly_percentage = (n_anomalies / len(anomaly_labels)) * 100
    
    print(f"\n📈 Anomaly Detection Results:")
    print(f"   • Total anomalies detected: {n_anomalies}")
    print(f"   • Anomaly percentage: {anomaly_percentage:.2f}%")
    print(f"   • Contamination rate: {contamination_rate*100:.1f}%")
    
    # Analyze anomaly characteristics
    if n_anomalies > 0:
        normal_data = X_anomaly[anomaly_labels == 0]
        anomaly_data = X_anomaly[anomaly_labels == 1]
        
        print(f"\n📊 Anomaly Analysis:")
        for feature in available_features:
            normal_mean = normal_data[feature].mean()
            anomaly_mean = anomaly_data[feature].mean()
            difference = anomaly_mean - normal_mean
            
            print(f"   • {feature}:")
            print(f"     Normal mean: {normal_mean:.4f}")
            print(f"     Anomaly mean: {anomaly_mean:.4f}")
            print(f"     Difference: {difference:+.4f}")
    
    # Score distribution analysis
    print(f"\n📈 Anomaly Score Distribution:")
    print(f"   • Min score: {anomaly_scores.min():.4f}")
    print(f"   • Max score: {anomaly_scores.max():.4f}")
    print(f"   • Mean score: {anomaly_scores.mean():.4f}")
    print(f"   • Std score: {anomaly_scores.std():.4f}")
    
    return iso_forest, anomaly_labels, anomaly_scores

def association_rules_apriori(X_scaled, feature_columns):
    """
    Association Rules Task: Discover market patterns using Apriori algorithm
    
    Args:
        X_scaled (pd.DataFrame): Scaled feature matrix
        feature_columns (list): List of feature names
        
    Returns:
        tuple: (frequent_itemsets, rules)
    """
    print("\n🎯 Task 4: Association Rules - Market Pattern Discovery")
    print("=" * 60)
    
    # Select key features for association rules
    rule_features = [
        'price_change_pct', 'volume_ratio', 'rsi', 'bb_position',
        'macd_bullish', 'rsi_oversold', 'rsi_overbought'
    ]
    
    # Filter features that exist in our dataset
    available_features = [f for f in rule_features if f in feature_columns]
    print(f"📊 Using {len(available_features)} features for association rules:")
    for feature in available_features:
        print(f"   • {feature}")
    
    X_rules = X_scaled[available_features]
    
    # Discretize continuous features into categories
    print("🔧 Discretizing features into categories...")
    X_discretized = X_rules.copy()
    
    # Discretize price_change_pct
    if 'price_change_pct' in available_features:
        X_discretized['price_change_pct'] = pd.cut(
            X_rules['price_change_pct'], 
            bins=[-np.inf, -0.3, 0.3, np.inf], 
            labels=['Negative', 'Neutral', 'Positive']
        )
    
    # Discretize volume_ratio
    if 'volume_ratio' in available_features:
        X_discretized['volume_ratio'] = pd.cut(
            X_rules['volume_ratio'], 
            bins=[0, 0.8, 1.2, np.inf], 
            labels=['Low', 'Normal', 'High']
        )
    
    # Discretize RSI
    if 'rsi' in available_features:
        X_discretized['rsi'] = pd.cut(
            X_rules['rsi'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Oversold', 'Neutral', 'Overbought']
        )
    
    # Discretize Bollinger Band position
    if 'bb_position' in available_features:
        X_discretized['bb_position'] = pd.cut(
            X_rules['bb_position'], 
            bins=[0, 0.2, 0.8, 1.0], 
            labels=['Below_Lower', 'Normal', 'Above_Upper']
        )
    
    # Convert to binary format for Apriori
    print("🔄 Converting to binary format...")
    X_binary = pd.get_dummies(X_discretized, prefix_sep='=')
    
    # Convert to boolean (required by mlxtend)
    X_binary = X_binary.astype(bool)
    
    print(f"   • Binary features created: {X_binary.shape[1]}")
    print(f"   • Sample size: {X_binary.shape[0]}")
    
    # Apply Apriori algorithm
    print("🚀 Mining frequent itemsets with Apriori...")
    min_support = 0.1  # 10% minimum support
    frequent_itemsets = apriori(X_binary, min_support=min_support, use_colnames=True)
    
    print(f"✅ Found {len(frequent_itemsets)} frequent itemsets")
    
    if len(frequent_itemsets) > 0:
        print(f"\n📊 Frequent Itemsets (min support: {min_support}):")
        for i, (_, itemset) in enumerate(frequent_itemsets.head(10).iterrows(), 1):
            support = itemset['support']
            items = list(itemset['itemsets'])
            print(f"   {i:2d}. {items} (support: {support:.3f})")
        
        # Generate association rules
        print(f"\n🔗 Generating association rules...")
        min_confidence = 0.3  # 30% minimum confidence
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        if len(rules) > 0:
            print(f"✅ Found {len(rules)} association rules")
            
            # Sort by confidence and lift
            rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
            
            print(f"\n📋 Top Association Rules (min confidence: {min_confidence}):")
            for i, (_, rule) in enumerate(rules.head(10).iterrows(), 1):
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                confidence = rule['confidence']
                lift = rule['lift']
                support = rule['support']
                
                print(f"   {i:2d}. IF {antecedents} THEN {consequents}")
                print(f"       Support: {support:.3f}, Confidence: {confidence:.3f}, Lift: {lift:.3f}")
        else:
            print("⚠️  No association rules found with the given thresholds")
            rules = pd.DataFrame()
    else:
        print("⚠️  No frequent itemsets found with the given support threshold")
        rules = pd.DataFrame()
    
    return frequent_itemsets, rules

def save_results(rf_model, kmeans_model, iso_forest, feature_importance, cluster_labels, anomaly_labels, rules):
    """
    Save all data mining results
    
    Args:
        rf_model: Trained Random Forest model
        kmeans_model: Trained K-Means model
        iso_forest: Trained Isolation Forest model
        feature_importance: Feature importance from Random Forest
        cluster_labels: Cluster labels from K-Means
        anomaly_labels: Anomaly labels from Isolation Forest
        rules: Association rules
    """
    print("\n💾 Saving Data Mining Results...")
    
    # Save feature importance
    feature_importance.to_csv('bitcoin_feature_importance.csv', index=False)
    
    # Save cluster labels
    cluster_df = pd.DataFrame({'cluster': cluster_labels})
    cluster_df.to_csv('bitcoin_cluster_labels.csv')
    
    # Save anomaly labels
    anomaly_df = pd.DataFrame({'anomaly': anomaly_labels})
    anomaly_df.to_csv('bitcoin_anomaly_labels.csv')
    
    # Save association rules
    if len(rules) > 0:
        rules.to_csv('bitcoin_association_rules.csv', index=False)
    
    print("✅ Results saved successfully!")
    print("   • Feature importance: bitcoin_feature_importance.csv")
    print("   • Cluster labels: bitcoin_cluster_labels.csv")
    print("   • Anomaly labels: bitcoin_anomaly_labels.csv")
    print("   • Association rules: bitcoin_association_rules.csv")

def main():
    """
    Main function to execute all data mining algorithms
    """
    print("🔍 Bitcoin Market Analysis - Data Mining Algorithms")
    print("=" * 70)
    
    # Load processed data
    df_features, X_scaled, y_classification, feature_columns = load_processed_data()
    
    # Task 1: Classification
    rf_model, X_test, y_test, y_pred, feature_importance = classification_random_forest(
        X_scaled, y_classification, feature_columns
    )
    
    # Task 2: Clustering
    kmeans_model, cluster_labels, cluster_centers = clustering_kmeans(X_scaled, feature_columns)
    
    # Task 3: Anomaly Detection
    iso_forest, anomaly_labels, anomaly_scores = anomaly_detection_isolation_forest(
        X_scaled, feature_columns
    )
    
    # Task 4: Association Rules
    frequent_itemsets, rules = association_rules_apriori(X_scaled, feature_columns)
    
    # Save all results
    save_results(rf_model, kmeans_model, iso_forest, feature_importance, 
                cluster_labels, anomaly_labels, rules)
    
    # Final summary
    print(f"\n🎉 Data Mining Algorithms Completed Successfully!")
    print("=" * 60)
    print(f"📊 Summary of Results:")
    print(f"   • Classification: Random Forest trained and evaluated")
    print(f"   • Clustering: {len(np.unique(cluster_labels))} market regimes identified")
    print(f"   • Anomaly Detection: {np.sum(anomaly_labels)} anomalies found")
    print(f"   • Association Rules: {len(rules)} rules discovered")
    
    print(f"\n✅ Ready for Step 5: Visualization!")
    
    return {
        'rf_model': rf_model,
        'kmeans_model': kmeans_model,
        'iso_forest': iso_forest,
        'feature_importance': feature_importance,
        'cluster_labels': cluster_labels,
        'anomaly_labels': anomaly_labels,
        'rules': rules,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

if __name__ == "__main__":
    results = main()

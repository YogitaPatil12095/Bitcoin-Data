"""
Simple script to run the Streamlit dashboard
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Bitcoin Market Analysis Dashboard...")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit>=1.28.0"])
        print("✅ Streamlit installed successfully")
    
    # Check if required data files exist
    required_files = [
        'bitcoin_processed_data.csv',
        'bitcoin_feature_importance.csv',
        'bitcoin_cluster_labels.csv',
        'bitcoin_anomaly_labels.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  Warning: Some data files are missing:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n📝 To generate the missing data, run:")
        print("   python 01_data_loading.py")
        print("   python 02_preprocessing_feature_engineering.py")
        print("   python 03_data_mining_algorithms.py")
        print("\n🔄 Continuing anyway...")
    
    # Run Streamlit
    print("\n🌐 Launching Streamlit dashboard...")
    print("📱 The dashboard will open in your default web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n⏹️  To stop the dashboard, press Ctrl+C")
    print("=" * 60)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error running dashboard: {e}")

if __name__ == "__main__":
    main()

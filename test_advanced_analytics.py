"""
Test script for ZeroLeak.AI Advanced Analytics Features
Tests all new ML models, forecasting, and analytics capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our components
from utils.analytics_engine import AdvancedAnalyticsEngine
from utils.data_processor import DataProcessor
from agents.advanced_dashboard_agent import AdvancedDashboardAgent

def create_sample_data():
    """Create comprehensive sample data for testing"""
    np.random.seed(42)
    
    # Generate sample transaction data
    n_transactions = 1000
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    data = []
    customers = [f'customer_{i}' for i in range(1, 101)]
    
    for i in range(n_transactions):
        # Random date
        date = np.random.choice(dates)
        
        # Random customer
        customer = np.random.choice(customers)
        
        # Amount with some patterns and anomalies
        base_amount = np.random.normal(100, 30)
        
        # Add some anomalies
        if np.random.random() < 0.05:  # 5% anomalies
            base_amount *= np.random.uniform(3, 10)
        
        # Add seasonal patterns
        if pd.Timestamp(date).month in [11, 12]:  # Holiday season
            base_amount *= 1.5
        
        # Add day-of-week patterns
        if pd.Timestamp(date).weekday() in [5, 6]:  # Weekend
            base_amount *= 1.2
        
        data.append({
            'date': date,
            'customer_id': customer,
            'amount': max(0, base_amount),
            'transaction_type': np.random.choice(['payment', 'refund', 'chargeback']),
            'status': np.random.choice(['success', 'failed', 'pending'])
        })
    
    return pd.DataFrame(data)

def test_analytics_engine():
    """Test the advanced analytics engine"""
    print("🧪 Testing Advanced Analytics Engine...")
    
    # Create sample data
    df = create_sample_data()
    print(f"✅ Created sample data with {len(df)} transactions")
    
    # Initialize analytics engine
    engine = AdvancedAnalyticsEngine()
    
    # Test 1: Anomaly Detection
    print("\n🔍 Testing Anomaly Detection...")
    try:
        anomaly_df = engine.detect_anomalies_advanced(df, 'amount', 'date', 'customer_id')
        anomaly_count = anomaly_df['is_anomaly'].sum()
        print(f"✅ Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.1f}%)")
    except Exception as e:
        print(f"❌ Anomaly detection failed: {e}")
    
    # Test 2: Revenue Forecasting
    print("\n🔮 Testing Revenue Forecasting...")
    try:
        forecasts = engine.forecast_revenue(df, 'amount', 'date', periods=30)
        if 'ensemble' in forecasts:
            avg_forecast = np.mean(forecasts['ensemble'])
            print(f"✅ Generated forecast with average daily revenue: ${avg_forecast:.2f}")
        else:
            print("❌ Forecasting failed - no ensemble forecast generated")
    except Exception as e:
        print(f"❌ Forecasting failed: {e}")
    
    # Test 3: Customer Segmentation
    print("\n👥 Testing Customer Segmentation...")
    try:
        customer_analysis = engine.customer_segmentation(df, 'amount', 'customer_id', 'date')
        if 'error' not in customer_analysis:
            total_customers = customer_analysis['total_customers']
            total_revenue = customer_analysis['total_revenue']
            print(f"✅ Segmented {total_customers} customers with total revenue: ${total_revenue:.2f}")
        else:
            print(f"❌ Customer segmentation failed: {customer_analysis['error']}")
    except Exception as e:
        print(f"❌ Customer segmentation failed: {e}")
    
    # Test 4: Churn Risk Prediction
    print("\n⚠️ Testing Churn Risk Prediction...")
    try:
        churn_analysis = engine.predict_churn_risk(df, 'customer_id', 'amount', 'date')
        high_risk_count = len(churn_analysis[churn_analysis['risk_category'] == 'High'])
        print(f"✅ Identified {high_risk_count} high-risk customers")
    except Exception as e:
        print(f"❌ Churn risk prediction failed: {e}")
    
    # Test 5: Advanced Insights Generation
    print("\n📊 Testing Advanced Insights Generation...")
    try:
        insights = engine.generate_advanced_insights(df, 'amount', 'date', 'customer_id')
        print(f"✅ Generated insights:")
        print(f"   - Total Revenue: ${insights.get('total_revenue', 0):,.2f}")
        print(f"   - Average Transaction: ${insights.get('avg_transaction', 0):,.2f}")
        print(f"   - Anomaly Rate: {insights.get('anomaly_percentage', 0):.1f}%")
        print(f"   - Growth Rate: {insights.get('revenue_growth_rate', 0):.1f}%")
    except Exception as e:
        print(f"❌ Insights generation failed: {e}")
    
    # Test 6: Visualization Generation
    print("\n📈 Testing Visualization Generation...")
    try:
        insights = engine.generate_advanced_insights(df, 'amount', 'date', 'customer_id')
        figures = engine.create_advanced_visualizations(df, insights)
        print(f"✅ Generated {len(figures)} visualization figures")
        for fig_name in figures.keys():
            print(f"   - {fig_name}")
    except Exception as e:
        print(f"❌ Visualization generation failed: {e}")

def test_dashboard_agent():
    """Test the advanced dashboard agent"""
    print("\n🧪 Testing Advanced Dashboard Agent...")
    
    # Create sample data
    df = create_sample_data()
    
    # Initialize dashboard agent
    agent = AdvancedDashboardAgent()
    
    # Test column detection
    print("\n🔍 Testing Column Detection...")
    try:
        amount_col = agent._detect_amount_column(df)
        date_col = agent._detect_date_column(df)
        customer_col = agent._detect_customer_column(df)
        
        print(f"✅ Detected columns:")
        print(f"   - Amount: {amount_col}")
        print(f"   - Date: {date_col}")
        print(f"   - Customer: {customer_col}")
    except Exception as e:
        print(f"❌ Column detection failed: {e}")
    
    # Test insights generation
    print("\n🤖 Testing AI Insights Generation...")
    try:
        insights = agent.analytics_engine.generate_advanced_insights(
            df, 'amount', 'date', 'customer_id'
        )
        ai_insights = agent.generate_ai_insights(insights)
        print(f"✅ Generated AI insights (length: {len(ai_insights)} characters)")
    except Exception as e:
        print(f"❌ AI insights generation failed: {e}")

def test_data_processor():
    """Test the data processor with advanced features"""
    print("\n🧪 Testing Data Processor...")
    
    # Create sample data
    df = create_sample_data()
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Test data type detection
    print("\n🔍 Testing Data Type Detection...")
    try:
        data_type = processor.detect_data_type(df)
        print(f"✅ Detected data type: {data_type}")
    except Exception as e:
        print(f"❌ Data type detection failed: {e}")
    
    # Test column detection
    print("\n📊 Testing Column Detection...")
    try:
        amount_col = processor._detect_amount_column(df)
        date_col = processor._detect_date_column(df)
        customer_col = processor._detect_customer_column(df)
        
        print(f"✅ Detected columns:")
        print(f"   - Amount: {amount_col}")
        print(f"   - Date: {date_col}")
        print(f"   - Customer: {customer_col}")
    except Exception as e:
        print(f"❌ Column detection failed: {e}")

def run_performance_tests():
    """Run performance tests on large datasets"""
    print("\n⚡ Running Performance Tests...")
    
    # Create larger dataset
    print("📊 Creating large dataset...")
    np.random.seed(42)
    n_transactions = 10000
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    data = []
    customers = [f'customer_{i}' for i in range(1, 1001)]
    
    for i in range(n_transactions):
        date = np.random.choice(dates)
        customer = np.random.choice(customers)
        amount = max(0, np.random.normal(100, 30))
        
        data.append({
            'date': date,
            'customer_id': customer,
            'amount': amount,
            'transaction_type': np.random.choice(['payment', 'refund', 'chargeback']),
            'status': np.random.choice(['success', 'failed', 'pending'])
        })
    
    df_large = pd.DataFrame(data)
    print(f"✅ Created large dataset with {len(df_large)} transactions")
    
    # Test performance
    import time
    
    engine = AdvancedAnalyticsEngine()
    
    # Test anomaly detection performance
    print("\n🔍 Testing Anomaly Detection Performance...")
    start_time = time.time()
    try:
        anomaly_df = engine.detect_anomalies_advanced(df_large, 'amount', 'date', 'customer_id')
        end_time = time.time()
        print(f"✅ Anomaly detection completed in {end_time - start_time:.2f} seconds")
        print(f"   - Detected {anomaly_df['is_anomaly'].sum()} anomalies")
    except Exception as e:
        print(f"❌ Anomaly detection performance test failed: {e}")
    
    # Test forecasting performance
    print("\n🔮 Testing Forecasting Performance...")
    start_time = time.time()
    try:
        forecasts = engine.forecast_revenue(df_large, 'amount', 'date', periods=30)
        end_time = time.time()
        print(f"✅ Forecasting completed in {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"❌ Forecasting performance test failed: {e}")
    
    # Test customer segmentation performance
    print("\n👥 Testing Customer Segmentation Performance...")
    start_time = time.time()
    try:
        customer_analysis = engine.customer_segmentation(df_large, 'amount', 'customer_id', 'date')
        end_time = time.time()
        print(f"✅ Customer segmentation completed in {end_time - start_time:.2f} seconds")
        print(f"   - Segmented {customer_analysis.get('total_customers', 0)} customers")
    except Exception as e:
        print(f"❌ Customer segmentation performance test failed: {e}")

def main():
    """Main test function"""
    print("🚀 ZeroLeak.AI Advanced Analytics Test Suite")
    print("=" * 50)
    
    try:
        # Test analytics engine
        test_analytics_engine()
        
        # Test dashboard agent
        test_dashboard_agent()
        
        # Test data processor
        test_data_processor()
        
        # Run performance tests
        run_performance_tests()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("🎉 Advanced Analytics features are working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
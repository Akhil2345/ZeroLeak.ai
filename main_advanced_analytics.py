"""
ZeroLeak.AI - Advanced Analytics Dashboard
Enhanced version with predictive analytics, ML models, and interactive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Custom imports
from utils.data_processor import DataProcessor
from utils.analytics_engine import AdvancedAnalyticsEngine
from agents.leak_detector import LeakDetector
from agents.insight_agent import InsightAgent
from agents.billing_agent import BillingAgent
from agents.advanced_dashboard_agent import AdvancedDashboardAgent
from config import UI_CONFIG

# Page configuration
st.set_page_config(
    page_title="ZeroLeak.AI - Advanced Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2196F3, #21CBF3);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #2196F3;
    }
    
    .success-card {
        border-left: 4px solid #4CAF50;
    }
    
    .warning-card {
        border-left: 4px solid #FF9800;
    }
    
    .danger-card {
        border-left: 4px solid #F44336;
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2196F3;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all components with caching"""
    return {
        'data_processor': DataProcessor(),
        'analytics_engine': AdvancedAnalyticsEngine(),
        'leak_detector': LeakDetector(),
        'insight_agent': InsightAgent(),
        'billing_agent': BillingAgent(),
        'dashboard_agent': AdvancedDashboardAgent()
    }

# Main application
def main():
    """Main application function"""
    
    # Initialize components
    components = initialize_components()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 ZeroLeak.AI - Advanced Analytics Dashboard</h1>
        <p>AI-Powered Revenue Leakage Detection with Predictive Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Analysis Type
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Comprehensive Analysis", "Quick Scan", "Deep Dive", "Predictive Analytics"],
            help="Choose the depth of analysis"
        )
        
        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            enable_ml = st.checkbox("Enable ML Models", value=True)
            enable_forecasting = st.checkbox("Enable Forecasting", value=True)
            enable_anomaly_detection = st.checkbox("Enable Anomaly Detection", value=True)
            forecast_periods = st.slider("Forecast Periods (days)", 7, 90, 30)
        
        # Data Quality Check
        st.header("📋 Data Quality")
        st.info("Upload your CSV/Excel file to begin analysis")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your revenue data (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel (.xlsx, .xls)"
        )
    
    with col2:
        st.header("📊 Sample Data")
        if uploaded_file is None:
            st.info("Upload a file to see sample data")
        else:
            try:
                sample_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.dataframe(sample_df.head(3), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            with st.spinner("🔄 Processing data..."):
                # Load and process data
                df = components['data_processor'].load_data(uploaded_file)
                data_type = components['data_processor'].detect_data_type(df)
                
                st.success(f"✅ Data loaded successfully! Detected type: {data_type}")
            
            # Data Overview
            st.header("📈 Data Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            
            with col2:
                st.metric("Columns", f"{len(df.columns)}")
            
            with col3:
                st.metric("Data Type", data_type)
            
            with col4:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            
            # Data Preview
            with st.expander("👀 Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Analysis Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🔍 Leak Detection", 
                "📊 Advanced Analytics", 
                "🔮 Predictions", 
                "👥 Customer Insights",
                "📋 AI Insights"
            ])
            
            # Tab 1: Leak Detection
            with tab1:
                st.header("🔍 Revenue Leak Detection")
                
                if st.button("🚀 Run Leak Detection", type="primary"):
                    with st.spinner("🔍 Detecting revenue leaks..."):
                        # Run leak detection
                        leak_results = components['leak_detector'].detect_leaks(df)
                        
                        # Display results
                        if not leak_results.empty:
                            st.success(f"✅ Found {len(leak_results)} potential revenue leaks!")
                            
                            # Leak summary
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Total Leaks", 
                                    len(leak_results),
                                    delta=f"{len(leak_results)} issues"
                                )
                            
                            with col2:
                                potential_loss = leak_results.get('amount', pd.Series([0])).sum()
                                st.metric(
                                    "Potential Loss", 
                                    f"${potential_loss:,.2f}"
                                )
                            
                            with col3:
                                avg_leak = leak_results.get('amount', pd.Series([0])).mean()
                                st.metric(
                                    "Avg Leak Size", 
                                    f"${avg_leak:,.2f}"
                                )
                            
                            # Leak details
                            st.subheader("Detected Issues")
                            st.dataframe(leak_results, use_container_width=True)
                            
                            # Download results
                            csv = leak_results.to_csv(index=False)
                            st.download_button(
                                "📥 Download Leak Report",
                                csv,
                                "zeroleak_report.csv",
                                "text/csv"
                            )
                        else:
                            st.success("🎉 No revenue leaks detected!")
            
            # Tab 2: Advanced Analytics
            with tab2:
                if enable_ml:
                    st.header("📊 Advanced Analytics Dashboard")
                    
                    # Run advanced analytics
                    with st.spinner("🧠 Running advanced analytics..."):
                        components['dashboard_agent'].render_advanced_dashboard(df, {})
                else:
                    st.warning("Enable ML Models in Advanced Options to access this feature.")
            
            # Tab 3: Predictions
            with tab3:
                if enable_forecasting:
                    st.header("🔮 Revenue Predictions")
                    
                    # Get column mappings
                    amount_col = components['data_processor']._detect_amount_column(df)
                    date_col = components['data_processor']._detect_date_column(df)
                    
                    if date_col:
                        with st.spinner("🔮 Generating forecasts..."):
                            forecasts = components['analytics_engine'].forecast_revenue(
                                df, amount_col, date_col, forecast_periods
                            )
                            
                            # Display forecasts
                            if 'ensemble' in forecasts:
                                st.subheader("Revenue Forecast")
                                
                                # Create forecast chart
                                fig_forecast = go.Figure()
                                
                                # Historical data
                                df_temp = df.copy()
                                df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                                daily_revenue = df_temp.groupby(df_temp[date_col].dt.date)[amount_col].sum()
                                
                                fig_forecast.add_trace(go.Scatter(
                                    x=daily_revenue.index,
                                    y=daily_revenue.values,
                                    mode='lines+markers',
                                    name='Historical Revenue',
                                    line=dict(color='#2196F3', width=2)
                                ))
                                
                                # Forecast
                                forecast_dates = pd.date_range(
                                    start=pd.Timestamp.now().date(),
                                    periods=len(forecasts['ensemble']),
                                    freq='D'
                                )
                                
                                fig_forecast.add_trace(go.Scatter(
                                    x=forecast_dates,
                                    y=forecasts['ensemble'],
                                    mode='lines+markers',
                                    name='Forecasted Revenue',
                                    line=dict(color='#FF9800', width=2, dash='dash')
                                ))
                                
                                fig_forecast.update_layout(
                                    title="Revenue Forecast",
                                    xaxis_title="Date",
                                    yaxis_title="Revenue ($)",
                                    height=500
                                )
                                
                                st.plotly_chart(fig_forecast, use_container_width=True)
                                
                                # Forecast metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    next_week = sum(forecasts['ensemble'][:7])
                                    st.metric("Next Week", f"${next_week:,.2f}")
                                
                                with col2:
                                    next_month = sum(forecasts['ensemble'][:30])
                                    st.metric("Next Month", f"${next_month:,.2f}")
                                
                                with col3:
                                    avg_daily = np.mean(forecasts['ensemble'])
                                    st.metric("Avg Daily", f"${avg_daily:,.2f}")
                    else:
                        st.warning("Date column required for forecasting.")
                else:
                    st.warning("Enable Forecasting in Advanced Options to access this feature.")
            
            # Tab 4: Customer Insights
            with tab4:
                st.header("👥 Customer Insights")
                
                customer_col = components['data_processor']._detect_customer_column(df)
                amount_col = components['data_processor']._detect_amount_column(df)
                date_col = components['data_processor']._detect_date_column(df)
                
                if customer_col:
                    with st.spinner("👥 Analyzing customer behavior..."):
                        # Customer segmentation
                        customer_analysis = components['analytics_engine'].customer_segmentation(
                            df, amount_col, customer_col, date_col
                        )
                        
                        if 'error' not in customer_analysis:
                            # Customer metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Total Customers", 
                                    customer_analysis['total_customers']
                                )
                            
                            with col2:
                                st.metric(
                                    "Total Revenue", 
                                    f"${customer_analysis['total_revenue']:,.2f}"
                                )
                            
                            with col3:
                                avg_revenue = customer_analysis['total_revenue'] / customer_analysis['total_customers']
                                st.metric(
                                    "Avg Customer Revenue", 
                                    f"${avg_revenue:,.2f}"
                                )
                            
                            # Customer segments
                            if 'segment_analysis' in customer_analysis and not customer_analysis['segment_analysis'].empty:
                                st.subheader("Customer Segments")
                                
                                segment_data = customer_analysis['segment_analysis']
                                fig_segments = px.bar(
                                    x=segment_data.index,
                                    y=segment_data['avg_revenue'],
                                    title="Average Revenue by Customer Segment",
                                    color_discrete_sequence=['#4CAF50']
                                )
                                fig_segments.update_layout(height=400)
                                st.plotly_chart(fig_segments, use_container_width=True)
                            
                            # Churn analysis
                            if date_col:
                                churn_analysis = components['analytics_engine'].predict_churn_risk(
                                    df, customer_col, amount_col, date_col
                                )
                                
                                st.subheader("Customer Churn Risk")
                                
                                risk_counts = churn_analysis['risk_category'].value_counts()
                                fig_churn = px.pie(
                                    values=risk_counts.values,
                                    names=risk_counts.index,
                                    title="Customer Churn Risk Distribution",
                                    color_discrete_sequence=['#4CAF50', '#FF9800', '#F44336']
                                )
                                fig_churn.update_layout(height=400)
                                st.plotly_chart(fig_churn, use_container_width=True)
                                
                                # High-risk customers
                                high_risk = churn_analysis[churn_analysis['risk_category'] == 'High']
                                if not high_risk.empty:
                                    st.subheader("High-Risk Customers")
                                    st.dataframe(
                                        high_risk[['customer_id', 'days_since_last', 'purchase_count', 'total_spent']].head(10),
                                        use_container_width=True
                                    )
                        else:
                            st.error(customer_analysis['error'])
                else:
                    st.warning("Customer column not detected. Please ensure your data includes customer identifiers.")
            
            # Tab 5: AI Insights
            with tab5:
                st.header("📋 AI-Powered Insights")
                
                if st.button("🧠 Generate AI Insights", type="primary"):
                    with st.spinner("🧠 Generating AI insights..."):
                        # Generate insights
                        insights = components['analytics_engine'].generate_advanced_insights(
                            df, 
                            components['data_processor']._detect_amount_column(df),
                            components['data_processor']._detect_date_column(df),
                            components['data_processor']._detect_customer_column(df)
                        )
                        
                        # Generate AI summary
                        ai_insights = components['dashboard_agent'].generate_ai_insights(insights)
                        
                        st.markdown("### 🤖 AI Analysis Summary")
                        st.markdown(ai_insights)
                        
                        # Key insights cards
                        st.subheader("📊 Key Insights")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card success-card">
                                <h4>💰 Revenue Overview</h4>
                                <p><strong>Total Revenue:</strong> ${insights.get('total_revenue', 0):,.2f}</p>
                                <p><strong>Average Transaction:</strong> ${insights.get('avg_transaction', 0):,.2f}</p>
                                <p><strong>Growth Rate:</strong> {insights.get('revenue_growth_rate', 0):.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card warning-card">
                                <h4>⚠️ Risk Assessment</h4>
                                <p><strong>Anomaly Rate:</strong> {insights.get('anomaly_percentage', 0):.1f}%</p>
                                <p><strong>Anomaly Count:</strong> {insights.get('anomaly_count', 0)}</p>
                                <p><strong>Potential Loss:</strong> ${insights.get('anomaly_count', 0) * insights.get('avg_transaction', 0):,.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main() 
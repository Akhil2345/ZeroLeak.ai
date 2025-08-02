"""
ZeroLeak.AI Advanced Analytics Demo
Showcases all the new advanced features with sample data
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

from utils.analytics_engine import AdvancedAnalyticsEngine
from agents.advanced_dashboard_agent import AdvancedDashboardAgent

def create_demo_data():
    """Create comprehensive demo data"""
    np.random.seed(42)
    
    # Generate realistic transaction data
    n_transactions = 2000
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    data = []
    customers = [f'customer_{i:03d}' for i in range(1, 201)]
    
    for i in range(n_transactions):
        # Random date with some patterns
        date = np.random.choice(dates)
        
        # Random customer
        customer = np.random.choice(customers)
        
        # Base amount with realistic patterns
        base_amount = np.random.normal(150, 50)
        
        # Add seasonal patterns
        if pd.Timestamp(date).month in [11, 12]:  # Holiday season
            base_amount *= 1.8
        elif pd.Timestamp(date).month in [6, 7, 8]:  # Summer
            base_amount *= 1.3
        
        # Add day-of-week patterns
        if pd.Timestamp(date).weekday() in [5, 6]:  # Weekend
            base_amount *= 1.4
        
        # Add some anomalies (5% of transactions)
        if np.random.random() < 0.05:
            base_amount *= np.random.uniform(3, 15)
        
        # Add some failed transactions (3% of transactions)
        status = 'success'
        if np.random.random() < 0.03:
            status = 'failed'
            base_amount *= 0.1  # Failed transactions have low amounts
        
        data.append({
            'date': date,
            'customer_id': customer,
            'amount': max(0, base_amount),
            'transaction_type': np.random.choice(['payment', 'refund', 'chargeback']),
            'status': status,
            'product_category': np.random.choice(['subscription', 'one_time', 'service']),
            'payment_method': np.random.choice(['credit_card', 'paypal', 'bank_transfer'])
        })
    
    return pd.DataFrame(data)

def main():
    """Main demo function"""
    st.set_page_config(
        page_title="ZeroLeak.AI - Advanced Analytics Demo",
        page_icon="🚀",
        layout="wide"
    )
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #2196F3, #21CBF3); padding: 2rem; border-radius: 10px; color: white; text-align: center;">
        <h1>🚀 ZeroLeak.AI - Advanced Analytics Demo</h1>
        <p>Experience the power of AI-powered revenue leakage detection with predictive analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    analytics_engine = AdvancedAnalyticsEngine()
    dashboard_agent = AdvancedDashboardAgent()
    
    # Create demo data
    with st.spinner("🔄 Generating demo data..."):
        df = create_demo_data()
    
    st.success(f"✅ Generated {len(df):,} demo transactions for {df['customer_id'].nunique()} customers")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"${df['amount'].sum():,.2f}")
    
    with col2:
        st.metric("Avg Transaction", f"${df['amount'].mean():,.2f}")
    
    with col3:
        st.metric("Success Rate", f"{(df['status'] == 'success').mean()*100:.1f}%")
    
    with col4:
        st.metric("Unique Customers", f"{df['customer_id'].nunique():,}")
    
    # Demo tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Anomaly Detection", 
        "🔮 Revenue Forecasting", 
        "👥 Customer Segmentation", 
        "⚠️ Churn Risk",
        "📊 Advanced Insights"
    ])
    
    # Tab 1: Anomaly Detection
    with tab1:
        st.header("🔍 Advanced Anomaly Detection")
        st.write("Detecting revenue anomalies using multiple ML methods...")
        
        with st.spinner("🔍 Running anomaly detection..."):
            anomaly_df = analytics_engine.detect_anomalies_advanced(
                df, 'amount', 'date', 'customer_id'
            )
        
        anomaly_count = anomaly_df['is_anomaly'].sum()
        anomaly_rate = (anomaly_count / len(df)) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Anomalies", anomaly_count)
        
        with col2:
            st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
        
        with col3:
            potential_loss = anomaly_df[anomaly_df['is_anomaly']]['amount'].sum()
            st.metric("Potential Loss", f"${potential_loss:,.2f}")
        
        # Anomaly visualization
        fig_anomaly = go.Figure()
        
        # Normal transactions
        normal_data = anomaly_df[~anomaly_df['is_anomaly']]
        fig_anomaly.add_trace(go.Scatter(
            x=normal_data['date'],
            y=normal_data['amount'],
            mode='markers',
            name='Normal',
            marker=dict(color='#4CAF50', size=4, opacity=0.6)
        ))
        
        # Anomalous transactions
        anomalies = anomaly_df[anomaly_df['is_anomaly']]
        fig_anomaly.add_trace(go.Scatter(
            x=anomalies['date'],
            y=anomalies['amount'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='#F44336', size=8, symbol='x')
        ))
        
        fig_anomaly.update_layout(
            title="Transaction Anomalies Over Time",
            xaxis_title="Date",
            yaxis_title="Amount ($)",
            height=500
        )
        
        st.plotly_chart(fig_anomaly, use_container_width=True)
        
        # Show top anomalies
        if not anomalies.empty:
            st.subheader("Top Anomalies Detected")
            top_anomalies = anomalies.nlargest(10, 'amount')[['date', 'customer_id', 'amount', 'anomaly_score']]
            st.dataframe(top_anomalies, use_container_width=True)
    
    # Tab 2: Revenue Forecasting
    with tab2:
        st.header("🔮 Revenue Forecasting")
        st.write("Predicting future revenue using multiple ML models...")
        
        with st.spinner("🔮 Generating forecasts..."):
            forecasts = analytics_engine.forecast_revenue(df, 'amount', 'date', periods=30)
        
        # Forecast visualization
        fig_forecast = go.Figure()
        
        # Historical data
        daily_revenue = df.groupby(df['date'].dt.date)['amount'].sum()
        
        fig_forecast.add_trace(go.Scatter(
            x=daily_revenue.index,
            y=daily_revenue.values,
            mode='lines+markers',
            name='Historical Revenue',
            line=dict(color='#2196F3', width=3)
        ))
        
        # Forecast
        if 'ensemble' in forecasts:
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
                line=dict(color='#FF9800', width=3, dash='dash')
            ))
        
        fig_forecast.update_layout(
            title="30-Day Revenue Forecast",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            height=500
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast metrics
        if 'ensemble' in forecasts:
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
    
    # Tab 3: Customer Segmentation
    with tab3:
        st.header("👥 Customer Segmentation")
        st.write("Analyzing customer behavior and creating segments...")
        
        with st.spinner("👥 Performing customer segmentation..."):
            customer_analysis = analytics_engine.customer_segmentation(
                df, 'amount', 'customer_id', 'date'
            )
        
        if 'error' not in customer_analysis:
            # Customer metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Customers", customer_analysis['total_customers'])
            
            with col2:
                st.metric("Total Revenue", f"${customer_analysis['total_revenue']:,.2f}")
            
            with col3:
                avg_revenue = customer_analysis['total_revenue'] / customer_analysis['total_customers']
                st.metric("Avg Customer Revenue", f"${avg_revenue:,.2f}")
            
            # Segment visualization
            if 'segment_analysis' in customer_analysis and not customer_analysis['segment_analysis'].empty:
                segment_data = customer_analysis['segment_analysis']
                
                fig_segments = px.bar(
                    x=segment_data.index,
                    y=segment_data['avg_revenue'],
                    title="Average Revenue by Customer Segment",
                    color_discrete_sequence=['#4CAF50']
                )
                fig_segments.update_layout(height=400)
                st.plotly_chart(fig_segments, use_container_width=True)
                
                # Segment details
                st.subheader("Segment Analysis")
                st.dataframe(segment_data, use_container_width=True)
        else:
            st.error(customer_analysis['error'])
    
    # Tab 4: Churn Risk
    with tab4:
        st.header("⚠️ Customer Churn Risk Analysis")
        st.write("Identifying customers at risk of churning...")
        
        with st.spinner("⚠️ Analyzing churn risk..."):
            churn_analysis = analytics_engine.predict_churn_risk(
                df, 'customer_id', 'amount', 'date'
            )
        
        # Risk distribution
        risk_counts = churn_analysis['risk_category'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("High Risk", risk_counts.get('High', 0))
        
        with col2:
            st.metric("Medium Risk", risk_counts.get('Medium', 0))
        
        with col3:
            st.metric("Low Risk", risk_counts.get('Low', 0))
        
        # Risk visualization
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
    
    # Tab 5: Advanced Insights
    with tab5:
        st.header("📊 Advanced Business Insights")
        st.write("AI-powered insights and recommendations...")
        
        with st.spinner("🤖 Generating AI insights..."):
            insights = analytics_engine.generate_advanced_insights(
                df, 'amount', 'date', 'customer_id'
            )
        
        # Key insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4CAF50;">
                <h4>💰 Revenue Overview</h4>
                <p><strong>Total Revenue:</strong> ${:,.2f}</p>
                <p><strong>Average Transaction:</strong> ${:,.2f}</p>
                <p><strong>Growth Rate:</strong> {:.1f}%</p>
            </div>
            """.format(
                insights.get('total_revenue', 0),
                insights.get('avg_transaction', 0),
                insights.get('revenue_growth_rate', 0)
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #FF9800;">
                <h4>⚠️ Risk Assessment</h4>
                <p><strong>Anomaly Rate:</strong> {:.1f}%</p>
                <p><strong>Anomaly Count:</strong> {}</p>
                <p><strong>Potential Loss:</strong> ${:,.2f}</p>
            </div>
            """.format(
                insights.get('anomaly_percentage', 0),
                insights.get('anomaly_count', 0),
                insights.get('anomaly_count', 0) * insights.get('avg_transaction', 0)
            ), unsafe_allow_html=True)
        
        # AI-generated insights
        st.subheader("🤖 AI-Powered Recommendations")
        
        try:
            ai_insights = dashboard_agent.generate_ai_insights(insights)
            st.markdown(ai_insights)
        except Exception as e:
            st.info("AI insights generation requires API key configuration. Please set up your LLM API keys in the .env file.")
        
        # Advanced visualizations
        st.subheader("📈 Advanced Visualizations")
        
        try:
            figures = analytics_engine.create_advanced_visualizations(df, insights)
            
            for fig_name, fig in figures.items():
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.warning(f"Visualization generation failed: {str(e)}")

if __name__ == "__main__":
    main() 
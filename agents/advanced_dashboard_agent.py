"""
Advanced Dashboard Agent for ZeroLeak.AI
Provides comprehensive analytics dashboard with predictive insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.analytics_engine import AdvancedAnalyticsEngine
from utils.llm_utils import LLMClient
from config import LLM_PROVIDERS, DEFAULT_PROVIDER, DEFAULT_MODEL

class AdvancedDashboardAgent:
    """
    Advanced dashboard agent providing comprehensive analytics and insights
    """
    
    def __init__(self):
        self.analytics_engine = AdvancedAnalyticsEngine()
        self.llm_client = LLMClient()
        
    def render_advanced_dashboard(self, df: pd.DataFrame, analysis_results: Dict[str, Any]):
        """
        Render the advanced analytics dashboard
        """
        st.markdown("## 🚀 Advanced Analytics Dashboard")
        
        # Get column mappings
        amount_col = self._detect_amount_column(df)
        date_col = self._detect_date_column(df)
        customer_col = self._detect_customer_column(df)
        
        # Generate advanced insights
        with st.spinner("🔍 Generating advanced insights..."):
            insights = self.analytics_engine.generate_advanced_insights(
                df, amount_col, date_col, customer_col
            )
        
        # Key Metrics Row
        self._render_key_metrics(insights)
        
        # Advanced Analytics Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Revenue Analytics", 
            "👥 Customer Insights", 
            "🔮 Predictions", 
            "⚠️ Anomaly Detection",
            "📈 Trends & Patterns"
        ])
        
        with tab1:
            self._render_revenue_analytics(df, insights, amount_col, date_col)
            
        with tab2:
            self._render_customer_insights(df, insights, customer_col)
            
        with tab3:
            self._render_predictions(insights)
            
        with tab4:
            self._render_anomaly_detection(df, insights, amount_col, date_col)
            
        with tab5:
            self._render_trends_patterns(df, insights, amount_col, date_col)
    
    def _detect_amount_column(self, df: pd.DataFrame) -> str:
        """Detect amount/price column"""
        amount_keywords = ['amount', 'price', 'value', 'total', 'revenue', 'cost']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in amount_keywords):
                return col
        return df.columns[0]  # fallback
    
    def _detect_date_column(self, df: pd.DataFrame) -> str:
        """Detect date column"""
        date_keywords = ['date', 'time', 'created', 'timestamp']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                return col
        return None
    
    def _detect_customer_column(self, df: pd.DataFrame) -> str:
        """Detect customer column"""
        customer_keywords = ['customer', 'user', 'client', 'email', 'name']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in customer_keywords):
                return col
        return None
    
    def _render_key_metrics(self, insights: Dict[str, Any]):
        """Render key performance metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Revenue", 
                f"${insights.get('total_revenue', 0):,.2f}",
                delta=f"{insights.get('revenue_growth_rate', 0):.1f}%"
            )
        
        with col2:
            st.metric(
                "Avg Transaction", 
                f"${insights.get('avg_transaction', 0):,.2f}"
            )
        
        with col3:
            st.metric(
                "Total Transactions", 
                f"{insights.get('total_transactions', 0):,}"
            )
        
        with col4:
            st.metric(
                "Anomaly Rate", 
                f"{insights.get('anomaly_percentage', 0):.1f}%",
                delta=f"-{insights.get('anomaly_count', 0)} issues"
            )
    
    def _render_revenue_analytics(self, df: pd.DataFrame, insights: Dict[str, Any], 
                                amount_col: str, date_col: str):
        """Render revenue analytics section"""
        st.subheader("📊 Revenue Analytics")
        
        # Revenue Trend Chart
        if 'daily_revenue_trend' in insights:
            fig_trend = self.analytics_engine.create_advanced_visualizations(df, insights)['revenue_trend']
            st.plotly_chart(fig_trend, use_container_width=True)
        
        # Revenue Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Revenue Distribution")
            fig_dist = px.histogram(
                df, x=amount_col, nbins=30,
                title="Transaction Amount Distribution",
                color_discrete_sequence=['#2196F3']
            )
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            st.subheader("Revenue by Time Period")
            if date_col:
                df_temp = df.copy()
                df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                df_temp['month'] = df_temp[date_col].dt.to_period('M')
                
                monthly_revenue = df_temp.groupby('month')[amount_col].sum().reset_index()
                monthly_revenue['month'] = monthly_revenue['month'].astype(str)
                
                fig_monthly = px.bar(
                    monthly_revenue, x='month', y=amount_col,
                    title="Monthly Revenue",
                    color_discrete_sequence=['#4CAF50']
                )
                fig_monthly.update_layout(height=400)
                st.plotly_chart(fig_monthly, use_container_width=True)
    
    def _render_customer_insights(self, df: pd.DataFrame, insights: Dict[str, Any], 
                                customer_col: str):
        """Render customer insights section"""
        st.subheader("👥 Customer Insights")
        
        if not customer_col or 'customer_analysis' not in insights:
            st.warning("Customer analysis requires a customer identifier column.")
            return
        
        customer_analysis = insights['customer_analysis']
        
        # Customer Segments
        if 'segment_analysis' in customer_analysis and not customer_analysis['segment_analysis'].empty:
            fig_segments = self.analytics_engine.create_advanced_visualizations(df, insights)['customer_segments']
            st.plotly_chart(fig_segments, use_container_width=True)
        
        # Customer Metrics Table
        st.subheader("Customer Metrics")
        customer_metrics = customer_analysis.get('customer_metrics', pd.DataFrame())
        if not customer_metrics.empty:
            st.dataframe(
                customer_metrics.head(10),
                use_container_width=True
            )
        
        # Churn Analysis
        if 'churn_analysis' in insights:
            st.subheader("Customer Churn Risk")
            churn_data = insights['churn_analysis']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_churn = self.analytics_engine.create_advanced_visualizations(df, insights)['churn_risk']
                st.plotly_chart(fig_churn, use_container_width=True)
            
            with col2:
                st.subheader("High Risk Customers")
                high_risk = churn_data[churn_data['risk_category'] == 'High']
                if not high_risk.empty:
                    st.dataframe(
                        high_risk[['customer_id', 'days_since_last', 'purchase_count', 'total_spent']].head(5),
                        use_container_width=True
                    )
                else:
                    st.success("No high-risk customers detected! 🎉")
    
    def _render_predictions(self, insights: Dict[str, Any]):
        """Render predictions and forecasting section"""
        st.subheader("🔮 Revenue Predictions")
        
        if 'forecasts' not in insights:
            st.warning("Forecasting requires time series data with sufficient history.")
            return
        
        forecasts = insights['forecasts']
        
        # Forecast Chart
        fig_forecast = self.analytics_engine.create_advanced_visualizations(pd.DataFrame(), insights)['forecast']
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast Summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'ensemble' in forecasts:
                next_week_forecast = sum(forecasts['ensemble'][:7])
                st.metric(
                    "Next Week Forecast",
                    f"${next_week_forecast:,.2f}"
                )
        
        with col2:
            if 'ensemble' in forecasts:
                next_month_forecast = sum(forecasts['ensemble'][:30])
                st.metric(
                    "Next Month Forecast",
                    f"${next_month_forecast:,.2f}"
                )
        
        with col3:
            if 'revenue_growth_rate' in insights:
                st.metric(
                    "Growth Trend",
                    f"{insights['revenue_growth_rate']:.1f}%"
                )
        
        # Model Performance
        st.subheader("Forecast Model Performance")
        model_metrics = []
        for model_name, forecast in forecasts.items():
            if isinstance(forecast, list):
                avg_forecast = np.mean(forecast)
                model_metrics.append({
                    'Model': model_name.title(),
                    'Average Forecast': f"${avg_forecast:,.2f}",
                    'Confidence': 'High' if model_name == 'ensemble' else 'Medium'
                })
        
        if model_metrics:
            st.dataframe(pd.DataFrame(model_metrics), use_container_width=True)
    
    def _render_anomaly_detection(self, df: pd.DataFrame, insights: Dict[str, Any], 
                                amount_col: str, date_col: str):
        """Render anomaly detection section"""
        st.subheader("⚠️ Anomaly Detection")
        
        # Anomaly Summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Anomalies",
                insights.get('anomaly_count', 0)
            )
        
        with col2:
            st.metric(
                "Anomaly Rate",
                f"{insights.get('anomaly_percentage', 0):.1f}%"
            )
        
        with col3:
            potential_loss = insights.get('anomaly_count', 0) * insights.get('avg_transaction', 0)
            st.metric(
                "Potential Loss",
                f"${potential_loss:,.2f}"
            )
        
        # Anomaly Detection Results
        if date_col:
            anomaly_df = self.analytics_engine.detect_anomalies_advanced(
                df, amount_col, date_col
            )
            
            # Show anomalies
            anomalies = anomaly_df[anomaly_df['is_anomaly'] == True]
            if not anomalies.empty:
                st.subheader("Detected Anomalies")
                st.dataframe(
                    anomalies[[date_col, amount_col, 'anomaly_score']].head(10),
                    use_container_width=True
                )
                
                # Anomaly Visualization
                fig_anomaly = go.Figure()
                
                # Normal transactions
                normal_data = anomaly_df[anomaly_df['is_anomaly'] == False]
                fig_anomaly.add_trace(go.Scatter(
                    x=normal_data[date_col],
                    y=normal_data[amount_col],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='#4CAF50', size=6)
                ))
                
                # Anomalous transactions
                fig_anomaly.add_trace(go.Scatter(
                    x=anomalies[date_col],
                    y=anomalies[amount_col],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='#F44336', size=10, symbol='x')
                ))
                
                fig_anomaly.update_layout(
                    title="Transaction Anomalies Over Time",
                    xaxis_title="Date",
                    yaxis_title="Amount ($)",
                    height=400
                )
                
                st.plotly_chart(fig_anomaly, use_container_width=True)
            else:
                st.success("No anomalies detected! 🎉")
    
    def _render_trends_patterns(self, df: pd.DataFrame, insights: Dict[str, Any], 
                              amount_col: str, date_col: str):
        """Render trends and patterns analysis"""
        st.subheader("📈 Trends & Patterns")
        
        if not date_col:
            st.warning("Trend analysis requires a date column.")
            return
        
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
        
        # Time-based patterns
        col1, col2 = st.columns(2)
        
        with col1:
            # Day of week analysis
            df_temp['day_of_week'] = df_temp[date_col].dt.day_name()
            daily_pattern = df_temp.groupby('day_of_week')[amount_col].mean().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            
            fig_daily = px.bar(
                x=daily_pattern.index,
                y=daily_pattern.values,
                title="Average Revenue by Day of Week",
                color_discrete_sequence=['#2196F3']
            )
            fig_daily.update_layout(height=400)
            st.plotly_chart(fig_daily, use_container_width=True)
        
        with col2:
            # Hour of day analysis (if time data available)
            df_temp['hour'] = df_temp[date_col].dt.hour
            hourly_pattern = df_temp.groupby('hour')[amount_col].mean()
            
            fig_hourly = px.line(
                x=hourly_pattern.index,
                y=hourly_pattern.values,
                title="Average Revenue by Hour of Day",
                color_discrete_sequence=['#4CAF50']
            )
            fig_hourly.update_layout(height=400)
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Seasonal decomposition (if enough data)
        if len(df_temp) > 50:
            st.subheader("Seasonal Decomposition")
            
            # Prepare time series data
            ts_data = df_temp.groupby(date_col)[amount_col].sum()
            ts_data = ts_data.sort_index()
            
            try:
                # Perform seasonal decomposition
                decomposition = seasonal_decompose(
                    ts_data, 
                    period=min(7, len(ts_data) // 4), 
                    extrapolate_trend='freq'
                )
                
                # Create subplot
                fig_decomp = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
                    vertical_spacing=0.05
                )
                
                fig_decomp.add_trace(
                    go.Scatter(x=ts_data.index, y=ts_data.values, name='Original'),
                    row=1, col=1
                )
                fig_decomp.add_trace(
                    go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, name='Trend'),
                    row=2, col=1
                )
                fig_decomp.add_trace(
                    go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, name='Seasonal'),
                    row=3, col=1
                )
                fig_decomp.add_trace(
                    go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, name='Residual'),
                    row=4, col=1
                )
                
                fig_decomp.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_decomp, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Seasonal decomposition not available: {str(e)}")
    
    def generate_ai_insights(self, insights: Dict[str, Any]) -> str:
        """Generate AI-powered insights using LLM"""
        try:
            prompt = f"""
            Based on the following business analytics data, provide actionable insights and recommendations:
            
            Key Metrics:
            - Total Revenue: ${insights.get('total_revenue', 0):,.2f}
            - Average Transaction: ${insights.get('avg_transaction', 0):,.2f}
            - Total Transactions: {insights.get('total_transactions', 0):,}
            - Anomaly Rate: {insights.get('anomaly_percentage', 0):.1f}%
            - Revenue Growth Rate: {insights.get('revenue_growth_rate', 0):.1f}%
            
            Customer Analysis:
            - Total Customers: {insights.get('customer_analysis', {}).get('total_customers', 0)}
            
            Please provide:
            1. Key business insights
            2. Revenue optimization opportunities
            3. Risk mitigation strategies
            4. Actionable recommendations
            
            Format the response in markdown with clear sections.
            """
            
            response = self.llm_client.call_llm(prompt)
            return response
            
        except Exception as e:
            return f"Unable to generate AI insights: {str(e)}" 
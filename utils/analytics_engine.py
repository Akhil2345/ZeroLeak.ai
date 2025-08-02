"""
Advanced Analytics Engine for ZeroLeak.AI
Provides predictive analytics, time series forecasting, and ML-powered insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb

# Time Series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Statistical Analysis
from scipy import stats
from scipy.stats import zscore, iqr
import statsmodels.api as sm

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class AdvancedAnalyticsEngine:
    """
    Advanced analytics engine for revenue leakage detection and prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.anomaly_detector = None
        self.forecast_model = None
        
    def detect_anomalies_advanced(self, df: pd.DataFrame, amount_col: str, 
                                date_col: str, customer_col: str = None) -> pd.DataFrame:
        """
        Advanced anomaly detection using multiple methods
        """
        df_anomaly = df.copy()
        
        # Method 1: Statistical Outliers (Z-score)
        z_scores = np.abs(zscore(df_anomaly[amount_col].dropna()))
        df_anomaly['z_score_anomaly'] = z_scores > 3
        
        # Method 2: IQR Method
        Q1 = df_anomaly[amount_col].quantile(0.25)
        Q3 = df_anomaly[amount_col].quantile(0.75)
        IQR = Q3 - Q1
        df_anomaly['iqr_anomaly'] = (
            (df_anomaly[amount_col] < (Q1 - 1.5 * IQR)) | 
            (df_anomaly[amount_col] > (Q3 + 1.5 * IQR))
        )
        
        # Method 3: Isolation Forest
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_scores = iso_forest.fit_predict(df_anomaly[[amount_col]].dropna())
            df_anomaly['isolation_forest_anomaly'] = iso_scores == -1
        except:
            df_anomaly['isolation_forest_anomaly'] = False
        
        # Method 4: Time-based anomalies (if date column exists)
        if date_col in df_anomaly.columns:
            df_anomaly = self._detect_temporal_anomalies(df_anomaly, amount_col, date_col)
        
        # Combined anomaly score
        anomaly_columns = [col for col in df_anomaly.columns if 'anomaly' in col]
        df_anomaly['anomaly_score'] = df_anomaly[anomaly_columns].sum(axis=1)
        df_anomaly['is_anomaly'] = df_anomaly['anomaly_score'] >= 2
        
        return df_anomaly
    
    def _detect_temporal_anomalies(self, df: pd.DataFrame, amount_col: str, 
                                  date_col: str) -> pd.DataFrame:
        """
        Detect anomalies based on temporal patterns
        """
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
        df_temp = df_temp.sort_values(date_col)
        
        # Rolling statistics
        window = min(30, len(df_temp) // 4)  # Adaptive window size
        if window > 1:
            rolling_mean = df_temp[amount_col].rolling(window=window, center=True).mean()
            rolling_std = df_temp[amount_col].rolling(window=window, center=True).std()
            
            # Detect deviations from rolling mean
            z_scores_temporal = np.abs((df_temp[amount_col] - rolling_mean) / rolling_std)
            df_temp['temporal_anomaly'] = z_scores_temporal > 2.5
        else:
            df_temp['temporal_anomaly'] = False
            
        return df_temp
    
    def forecast_revenue(self, df: pd.DataFrame, amount_col: str, 
                        date_col: str, periods: int = 30) -> Dict[str, Any]:
        """
        Forecast future revenue using multiple models
        """
        df_forecast = df.copy()
        df_forecast[date_col] = pd.to_datetime(df_forecast[date_col], errors='coerce')
        df_forecast = df_forecast.sort_values(date_col)
        
        # Prepare time series data
        ts_data = df_forecast.groupby(date_col)[amount_col].sum().reset_index()
        ts_data = ts_data.set_index(date_col)
        
        forecasts = {}
        
        # Model 1: Simple Moving Average
        try:
            ma_forecast = ts_data[amount_col].rolling(window=7).mean().iloc[-1]
            forecasts['moving_average'] = ma_forecast
        except:
            forecasts['moving_average'] = ts_data[amount_col].mean()
        
        # Model 2: Exponential Smoothing
        try:
            if len(ts_data) > 10:
                es_model = ExponentialSmoothing(ts_data[amount_col], 
                                              seasonal_periods=7 if len(ts_data) > 14 else None)
                es_fitted = es_model.fit()
                es_forecast = es_fitted.forecast(periods)
                forecasts['exponential_smoothing'] = es_forecast.tolist()
            else:
                forecasts['exponential_smoothing'] = [ts_data[amount_col].mean()] * periods
        except:
            forecasts['exponential_smoothing'] = [ts_data[amount_col].mean()] * periods
        
        # Model 3: ARIMA (if enough data)
        try:
            if len(ts_data) > 20:
                # Check for stationarity
                adf_result = adfuller(ts_data[amount_col].dropna())
                if adf_result[1] > 0.05:  # Not stationary
                    ts_data_diff = ts_data[amount_col].diff().dropna()
                else:
                    ts_data_diff = ts_data[amount_col]
                
                # Simple ARIMA model
                arima_model = ARIMA(ts_data_diff, order=(1, 1, 1))
                arima_fitted = arima_model.fit()
                arima_forecast = arima_fitted.forecast(steps=periods)
                forecasts['arima'] = arima_forecast.tolist()
            else:
                forecasts['arima'] = [ts_data[amount_col].mean()] * periods
        except:
            forecasts['arima'] = [ts_data[amount_col].mean()] * periods
        
        # Model 4: Linear Trend
        try:
            x = np.arange(len(ts_data))
            y = ts_data[amount_col].values
            slope, intercept, _, _, _ = stats.linregress(x, y)
            trend_forecast = [intercept + slope * (len(ts_data) + i) for i in range(periods)]
            forecasts['linear_trend'] = trend_forecast
        except:
            forecasts['linear_trend'] = [ts_data[amount_col].mean()] * periods
        
        # Ensemble forecast (average of all models)
        ensemble_forecast = []
        for i in range(periods):
            values = []
            for model_name, forecast in forecasts.items():
                if isinstance(forecast, list) and len(forecast) > i:
                    values.append(forecast[i])
                elif isinstance(forecast, (int, float)):
                    values.append(forecast)
            
            if values:
                ensemble_forecast.append(np.mean(values))
            else:
                ensemble_forecast.append(ts_data[amount_col].mean())
        
        forecasts['ensemble'] = ensemble_forecast
        
        return forecasts
    
    def customer_segmentation(self, df: pd.DataFrame, amount_col: str, 
                            customer_col: str, date_col: str = None) -> Dict[str, Any]:
        """
        Perform customer segmentation analysis
        """
        if customer_col not in df.columns:
            return {"error": "Customer column not found"}
        
        # Customer metrics
        customer_metrics = df.groupby(customer_col).agg({
            amount_col: ['sum', 'mean', 'count', 'std']
        }).round(2)
        
        customer_metrics.columns = ['total_revenue', 'avg_transaction', 'transaction_count', 'std_amount']
        customer_metrics = customer_metrics.reset_index()
        
        # RFM Analysis (if date column exists)
        if date_col and date_col in df.columns:
            customer_metrics = self._calculate_rfm(df, customer_col, amount_col, date_col, customer_metrics)
        
        # K-means clustering
        try:
            # Prepare features for clustering
            features = customer_metrics[['total_revenue', 'avg_transaction', 'transaction_count']].copy()
            
            # Handle missing values
            features = features.fillna(features.mean())
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Determine optimal number of clusters
            n_clusters = min(4, len(features) // 3)  # Adaptive clustering
            if n_clusters < 2:
                n_clusters = 2
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            customer_metrics['segment'] = kmeans.fit_predict(features_scaled)
            
            # Segment analysis
            segment_analysis = customer_metrics.groupby('segment').agg({
                'total_revenue': ['mean', 'sum', 'count'],
                'avg_transaction': 'mean',
                'transaction_count': 'mean'
            }).round(2)
            
            segment_analysis.columns = ['avg_revenue', 'total_revenue', 'customer_count', 'avg_transaction', 'avg_transactions']
            
        except Exception as e:
            customer_metrics['segment'] = 0
            segment_analysis = pd.DataFrame()
        
        return {
            'customer_metrics': customer_metrics,
            'segment_analysis': segment_analysis,
            'total_customers': len(customer_metrics),
            'total_revenue': customer_metrics['total_revenue'].sum()
        }
    
    def _calculate_rfm(self, df: pd.DataFrame, customer_col: str, amount_col: str, 
                      date_col: str, customer_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics
        """
        df_rfm = df.copy()
        df_rfm[date_col] = pd.to_datetime(df_rfm[date_col], errors='coerce')
        
        # Calculate RFM metrics
        rfm = df_rfm.groupby(customer_col).agg({
            date_col: lambda x: (pd.Timestamp.now() - x.max()).days,  # Recency
            amount_col: ['count', 'sum']  # Frequency, Monetary
        })
        
        rfm.columns = ['recency', 'frequency', 'monetary']
        rfm = rfm.reset_index()
        
        # Merge with existing metrics
        customer_metrics = customer_metrics.merge(rfm, on=customer_col, how='left')
        
        return customer_metrics
    
    def predict_churn_risk(self, df: pd.DataFrame, customer_col: str, 
                          amount_col: str, date_col: str) -> pd.DataFrame:
        """
        Predict customer churn risk based on behavior patterns
        """
        df_churn = df.copy()
        df_churn[date_col] = pd.to_datetime(df_churn[date_col], errors='coerce')
        
        # Calculate customer behavior metrics
        customer_behavior = df_churn.groupby(customer_col).agg({
            date_col: ['min', 'max', 'count'],
            amount_col: ['sum', 'mean', 'std']
        }).round(2)
        
        customer_behavior.columns = ['first_purchase', 'last_purchase', 'purchase_count', 
                                   'total_spent', 'avg_amount', 'amount_std']
        customer_behavior = customer_behavior.reset_index()
        
        # Calculate days since last purchase
        customer_behavior['days_since_last'] = (
            pd.Timestamp.now() - customer_behavior['last_purchase']
        ).dt.days
        
        # Calculate purchase frequency
        customer_behavior['purchase_frequency'] = (
            (customer_behavior['last_purchase'] - customer_behavior['first_purchase']).dt.days / 
            customer_behavior['purchase_count']
        ).fillna(0)
        
        # Churn risk indicators
        customer_behavior['churn_risk'] = 0
        
        # High risk indicators
        customer_behavior.loc[customer_behavior['days_since_last'] > 90, 'churn_risk'] += 3
        customer_behavior.loc[customer_behavior['purchase_count'] == 1, 'churn_risk'] += 2
        customer_behavior.loc[customer_behavior['avg_amount'] < customer_behavior['avg_amount'].quantile(0.25), 'churn_risk'] += 1
        
        # Risk categories
        customer_behavior['risk_category'] = pd.cut(
            customer_behavior['churn_risk'], 
            bins=[-1, 1, 3, 6], 
            labels=['Low', 'Medium', 'High']
        )
        
        return customer_behavior
    
    def generate_advanced_insights(self, df: pd.DataFrame, amount_col: str, 
                                 date_col: str, customer_col: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive business insights
        """
        insights = {}
        
        # Basic statistics
        insights['total_revenue'] = df[amount_col].sum()
        insights['avg_transaction'] = df[amount_col].mean()
        insights['total_transactions'] = len(df)
        insights['unique_customers'] = df[customer_col].nunique() if customer_col else None
        
        # Revenue trends
        if date_col in df.columns:
            df_trend = df.copy()
            df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
            df_trend = df_trend.sort_values(date_col)
            
            # Daily revenue
            daily_revenue = df_trend.groupby(df_trend[date_col].dt.date)[amount_col].sum()
            insights['daily_revenue_trend'] = daily_revenue.to_dict()
            
            # Growth rate
            if len(daily_revenue) > 1:
                growth_rate = ((daily_revenue.iloc[-1] - daily_revenue.iloc[0]) / daily_revenue.iloc[0]) * 100
                insights['revenue_growth_rate'] = round(growth_rate, 2)
        
        # Anomaly analysis
        anomaly_df = self.detect_anomalies_advanced(df, amount_col, date_col, customer_col)
        insights['anomaly_count'] = anomaly_df['is_anomaly'].sum()
        insights['anomaly_percentage'] = round((insights['anomaly_count'] / len(df)) * 100, 2)
        
        # Customer analysis
        if customer_col:
            customer_analysis = self.customer_segmentation(df, amount_col, customer_col, date_col)
            insights['customer_analysis'] = customer_analysis
            
            # Churn analysis
            churn_analysis = self.predict_churn_risk(df, customer_col, amount_col, date_col)
            insights['churn_analysis'] = churn_analysis
        
        # Forecasting
        if date_col in df.columns:
            forecasts = self.forecast_revenue(df, amount_col, date_col)
            insights['forecasts'] = forecasts
        
        return insights
    
    def create_advanced_visualizations(self, df: pd.DataFrame, insights: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Create advanced interactive visualizations
        """
        figures = {}
        
        # Revenue Trend Chart
        if 'daily_revenue_trend' in insights:
            dates = list(insights['daily_revenue_trend'].keys())
            revenues = list(insights['daily_revenue_trend'].values())
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=dates, y=revenues,
                mode='lines+markers',
                name='Daily Revenue',
                line=dict(color='#2196F3', width=3),
                marker=dict(size=6)
            ))
            
            fig_trend.update_layout(
                title='Revenue Trend Over Time',
                xaxis_title='Date',
                yaxis_title='Revenue ($)',
                template='plotly_white',
                height=400
            )
            figures['revenue_trend'] = fig_trend
        
        # Customer Segmentation Chart
        if 'customer_analysis' in insights and 'segment_analysis' in insights['customer_analysis']:
            segment_data = insights['customer_analysis']['segment_analysis']
            if not segment_data.empty:
                fig_segments = go.Figure(data=[
                    go.Bar(
                        x=segment_data.index,
                        y=segment_data['avg_revenue'],
                        name='Average Revenue',
                        marker_color='#4CAF50'
                    )
                ])
                
                fig_segments.update_layout(
                    title='Customer Segments by Average Revenue',
                    xaxis_title='Segment',
                    yaxis_title='Average Revenue ($)',
                    template='plotly_white',
                    height=400
                )
                figures['customer_segments'] = fig_segments
        
        # Churn Risk Distribution
        if 'churn_analysis' in insights:
            churn_data = insights['churn_analysis']
            risk_counts = churn_data['risk_category'].value_counts()
            
            fig_churn = go.Figure(data=[
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.4,
                    marker_colors=['#4CAF50', '#FF9800', '#F44336']
                )
            ])
            
            fig_churn.update_layout(
                title='Customer Churn Risk Distribution',
                template='plotly_white',
                height=400
            )
            figures['churn_risk'] = fig_churn
        
        # Forecast Chart
        if 'forecasts' in insights:
            forecast_data = insights['forecasts']
            if 'ensemble' in forecast_data:
                fig_forecast = go.Figure()
                
                # Historical data
                if 'daily_revenue_trend' in insights:
                    dates = list(insights['daily_revenue_trend'].keys())
                    revenues = list(insights['daily_revenue_trend'].values())
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=dates, y=revenues,
                        mode='lines+markers',
                        name='Historical Revenue',
                        line=dict(color='#2196F3', width=2)
                    ))
                
                # Forecast
                forecast_dates = pd.date_range(
                    start=pd.Timestamp.now().date(),
                    periods=len(forecast_data['ensemble']),
                    freq='D'
                )
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_data['ensemble'],
                    mode='lines+markers',
                    name='Forecasted Revenue',
                    line=dict(color='#FF9800', width=2, dash='dash')
                ))
                
                fig_forecast.update_layout(
                    title='Revenue Forecast',
                    xaxis_title='Date',
                    yaxis_title='Revenue ($)',
                    template='plotly_white',
                    height=400
                )
                figures['forecast'] = fig_forecast
        
        return figures 
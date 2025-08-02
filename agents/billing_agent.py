import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from utils.llm_utils import LLMClient
from config import DEFAULT_PROVIDER, DEFAULT_MODEL

class BillingAgent:
    def __init__(self, provider: str = DEFAULT_PROVIDER, model: str = DEFAULT_MODEL):
        self.llm_client = LLMClient(provider, model)
        
    def analyze_payment_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze payment patterns and identify potential issues"""
        analysis = {}
        
        # Find relevant columns
        amount_col = self._find_amount_column(df)
        status_col = self._find_status_column(df)
        date_col = self._find_date_column(df)
        customer_col = self._find_customer_column(df)
        
        if amount_col and status_col:
            # Payment success rate analysis
            total_transactions = len(df)
            successful_transactions = len(df[df[status_col].str.contains('success|completed|paid', case=False, na=False)])
            failed_transactions = len(df[df[status_col].str.contains('fail|decline|error', case=False, na=False)])
            
            analysis['payment_metrics'] = {
                'total_transactions': total_transactions,
                'successful_transactions': successful_transactions,
                'failed_transactions': failed_transactions,
                'success_rate': (successful_transactions / total_transactions * 100) if total_transactions > 0 else 0,
                'failure_rate': (failed_transactions / total_transactions * 100) if total_transactions > 0 else 0
            }
            
            # Amount analysis
            if amount_col:
                analysis['amount_analysis'] = {
                    'total_revenue': df[amount_col].sum(),
                    'average_transaction': df[amount_col].mean(),
                    'median_transaction': df[amount_col].median(),
                    'max_transaction': df[amount_col].max(),
                    'min_transaction': df[amount_col].min(),
                    'failed_revenue': df[df[status_col].str.contains('fail|decline|error', case=False, na=False)][amount_col].sum()
                }
                
        # Time-based analysis
        if date_col:
            df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            analysis['time_analysis'] = self._analyze_time_patterns(df, status_col, amount_col)
            
        # Customer analysis
        if customer_col:
            analysis['customer_analysis'] = self._analyze_customer_patterns(df, customer_col, status_col, amount_col)
            
        return analysis
        
    def _analyze_time_patterns(self, df: pd.DataFrame, status_col: str, amount_col: str) -> Dict[str, Any]:
        """Analyze payment patterns over time"""
        analysis = {}
        
        # Daily patterns
        daily_stats = df.groupby(df['date'].dt.date).agg({
            status_col: 'count',
            amount_col: 'sum'
        }).rename(columns={status_col: 'transaction_count', amount_col: 'daily_revenue'})
        
        analysis['daily_patterns'] = {
            'avg_daily_transactions': daily_stats['transaction_count'].mean(),
            'avg_daily_revenue': daily_stats['daily_revenue'].mean(),
            'peak_day': daily_stats['daily_revenue'].idxmax(),
            'peak_revenue': daily_stats['daily_revenue'].max()
        }
        
        # Weekly patterns
        df['day_of_week'] = df['date'].dt.day_name()
        weekly_stats = df.groupby('day_of_week').agg({
            status_col: 'count',
            amount_col: 'sum'
        }).rename(columns={status_col: 'transaction_count', amount_col: 'weekly_revenue'})
        
        analysis['weekly_patterns'] = {
            'best_day': weekly_stats['weekly_revenue'].idxmax(),
            'worst_day': weekly_stats['weekly_revenue'].idxmin(),
            'weekend_performance': weekly_stats.loc[['Saturday', 'Sunday'], 'weekly_revenue'].sum() if 'Saturday' in weekly_stats.index else 0
        }
        
        # Monthly trends
        df['month'] = df['date'].dt.month
        monthly_stats = df.groupby('month').agg({
            status_col: 'count',
            amount_col: 'sum'
        }).rename(columns={status_col: 'transaction_count', amount_col: 'monthly_revenue'})
        
        analysis['monthly_trends'] = {
            'best_month': monthly_stats['monthly_revenue'].idxmax(),
            'worst_month': monthly_stats['monthly_revenue'].idxmin(),
            'monthly_growth': self._calculate_growth_rate(monthly_stats['monthly_revenue'])
        }
        
        return analysis
        
    def _analyze_customer_patterns(self, df: pd.DataFrame, customer_col: str, status_col: str, amount_col: str) -> Dict[str, Any]:
        """Analyze customer payment patterns"""
        analysis = {}
        
        # Customer transaction frequency
        customer_frequency = df.groupby(customer_col).size().sort_values(ascending=False)
        analysis['customer_frequency'] = {
            'most_active_customer': customer_frequency.index[0],
            'avg_transactions_per_customer': customer_frequency.mean(),
            'customers_with_multiple_transactions': len(customer_frequency[customer_frequency > 1])
        }
        
        # Customer revenue analysis
        customer_revenue = df.groupby(customer_col)[amount_col].sum().sort_values(ascending=False)
        analysis['customer_revenue'] = {
            'top_customer': customer_revenue.index[0],
            'top_customer_revenue': customer_revenue.iloc[0],
            'avg_customer_revenue': customer_revenue.mean(),
            'revenue_concentration': (customer_revenue.head(5).sum() / customer_revenue.sum() * 100) if customer_revenue.sum() > 0 else 0
        }
        
        # Customer failure analysis
        failed_customers = df[df[status_col].str.contains('fail|decline|error', case=False, na=False)].groupby(customer_col).size()
        analysis['customer_failures'] = {
            'customers_with_failures': len(failed_customers),
            'customer_with_most_failures': failed_customers.index[0] if len(failed_customers) > 0 else None,
            'avg_failures_per_customer': failed_customers.mean() if len(failed_customers) > 0 else 0
        }
        
        return analysis
        
    def detect_billing_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect billing anomalies and suspicious patterns"""
        anomalies = []
        
        amount_col = self._find_amount_column(df)
        status_col = self._find_status_column(df)
        customer_col = self._find_customer_column(df)
        date_col = self._find_date_column(df)
        
        if amount_col:
            # Detect amount outliers using IQR method
            Q1 = df[amount_col].quantile(0.25)
            Q3 = df[amount_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[amount_col] < lower_bound) | (df[amount_col] > upper_bound)]
            
            for idx, row in outliers.iterrows():
                anomalies.append({
                    'row_index': idx,
                    'anomaly_type': 'amount_outlier',
                    'severity': 'medium',
                    'description': f'Unusual amount: ${row[amount_col]:,.2f}',
                    'customer': row[customer_col] if customer_col else 'Unknown',
                    'date': row[date_col] if date_col else 'Unknown',
                    'amount': row[amount_col]
                })
                
        # Detect duplicate transactions
        if customer_col and amount_col and date_col:
            duplicates = self._find_duplicate_transactions(df, customer_col, amount_col, date_col)
            for dup in duplicates:
                anomalies.append({
                    'row_index': dup['row_index'],
                    'anomaly_type': 'duplicate_transaction',
                    'severity': 'high',
                    'description': f'Potential duplicate transaction',
                    'customer': dup['customer'],
                    'date': dup['date'],
                    'amount': dup['amount']
                })
                
        # Detect rapid successive transactions
        if customer_col and date_col:
            rapid_transactions = self._detect_rapid_transactions(df, customer_col, date_col)
            for rapid in rapid_transactions:
                anomalies.append({
                    'row_index': rapid['row_index'],
                    'anomaly_type': 'rapid_transactions',
                    'severity': 'medium',
                    'description': f'Rapid successive transactions: {rapid["count"]} in {rapid["timeframe"]}',
                    'customer': rapid['customer'],
                    'date': rapid['date'],
                    'amount': rapid['amount']
                })
                
        return pd.DataFrame(anomalies)
        
    def _find_duplicate_transactions(self, df: pd.DataFrame, customer_col: str, amount_col: str, date_col: str) -> List[Dict]:
        """Find potential duplicate transactions"""
        duplicates = []
        
        # Group by customer and amount
        grouped = df.groupby([customer_col, amount_col])
        
        for (customer, amount), group in grouped:
            if len(group) > 1:
                # Check for transactions within 24 hours
                group['date'] = pd.to_datetime(group[date_col], errors='coerce')
                group_sorted = group.sort_values('date')
                
                for i in range(len(group_sorted) - 1):
                    time_diff = group_sorted.iloc[i+1]['date'] - group_sorted.iloc[i]['date']
                    if time_diff <= timedelta(hours=24):
                        duplicates.append({
                            'row_index': group_sorted.iloc[i+1].name,
                            'customer': customer,
                            'date': group_sorted.iloc[i+1][date_col],
                            'amount': amount
                        })
                        
        return duplicates
        
    def _detect_rapid_transactions(self, df: pd.DataFrame, customer_col: str, date_col: str) -> List[Dict]:
        """Detect rapid successive transactions from the same customer"""
        rapid_transactions = []
        
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        for customer in df[customer_col].unique():
            customer_data = df[df[customer_col] == customer].sort_values('date')
            
            if len(customer_data) > 1:
                # Check for transactions within 1 hour
                for i in range(len(customer_data) - 1):
                    time_diff = customer_data.iloc[i+1]['date'] - customer_data.iloc[i]['date']
                    if time_diff <= timedelta(hours=1):
                        rapid_transactions.append({
                            'row_index': customer_data.iloc[i+1].name,
                            'customer': customer,
                            'date': customer_data.iloc[i+1][date_col],
                            'amount': customer_data.iloc[i+1].get(self._find_amount_column(df), 0),
                            'count': len(customer_data[customer_data['date'] >= customer_data.iloc[i]['date'] - timedelta(hours=1)]),
                            'timeframe': '1 hour'
                        })
                        
        return rapid_transactions
        
    def _calculate_growth_rate(self, series: pd.Series) -> float:
        """Calculate growth rate between first and last values"""
        if len(series) < 2:
            return 0.0
        return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100) if series.iloc[0] != 0 else 0.0
        
    def _find_amount_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the amount/price column"""
        amount_keywords = ['amount', 'price', 'cost', 'charge', 'total', 'value']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in amount_keywords):
                return col
        return None
        
    def _find_status_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the status column"""
        status_keywords = ['status', 'state', 'condition', 'result']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in status_keywords):
                return col
        return None
        
    def _find_customer_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the customer column"""
        customer_keywords = ['customer', 'client', 'user', 'account', 'name']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in customer_keywords):
                return col
        return None
        
    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the date column"""
        date_keywords = ['date', 'time', 'created', 'updated', 'timestamp']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                return col
        return None
        
    def generate_billing_report(self, df: pd.DataFrame) -> str:
        """Generate comprehensive billing analysis report"""
        analysis = self.analyze_payment_patterns(df)
        anomalies = self.detect_billing_anomalies(df)
        
        prompt = f"""
You are a billing and payment processing expert. Based on the following billing analysis, provide a comprehensive report with insights and recommendations.

**Payment Metrics:**
{analysis.get('payment_metrics', {})}

**Amount Analysis:**
{analysis.get('amount_analysis', {})}

**Time Patterns:**
{analysis.get('time_analysis', {})}

**Customer Analysis:**
{analysis.get('customer_analysis', {})}

**Detected Anomalies:**
{anomalies.to_string() if not anomalies.empty else 'No anomalies detected'}

Provide a detailed report covering:
1. **Payment Performance Overview** - Success rates, revenue trends
2. **Key Insights** - Patterns, anomalies, opportunities
3. **Risk Assessment** - Potential revenue loss, fraud indicators
4. **Actionable Recommendations** - Specific steps to improve billing processes
5. **Customer Retention Strategies** - How to reduce failed payments and improve customer experience

Focus on practical insights that a startup can implement immediately.
"""
        
        return self.llm_client.call_llm(prompt, temperature=0.7, max_tokens=1000) 
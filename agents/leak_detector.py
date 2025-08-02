import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import re
from config import LEAKAGE_RULES

class LeakDetector:
    def __init__(self):
        self.rules = LEAKAGE_RULES
        
    def detect_leaks(self, df: pd.DataFrame, data_type: str = 'billing') -> pd.DataFrame:
        """Main method to detect revenue leaks based on data type"""
        if data_type == 'billing':
            return self._detect_billing_leaks(df)
        elif data_type == 'support':
            return self._detect_support_leaks(df)
        elif data_type == 'operations':
            return self._detect_operations_leaks(df)
        else:
            return self._detect_generic_leaks(df)
            
    def _detect_billing_leaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect billing-related revenue leaks"""
    issues = []

        # Find amount and status columns
        amount_col = self._find_amount_column(df)
        status_col = self._find_status_column(df)
        customer_col = self._find_customer_column(df)
        date_col = self._find_date_column(df)

    for idx, row in df.iterrows():
            row_issues = []
            
            # Check for failed charges
            if status_col and self._is_failed_charge(row[status_col]):
                row_issues.append({
                    'issue_type': 'failed_charge',
                    'severity': 'high',
                    'description': f'Failed charge: {row[status_col]}',
                    'potential_loss': row[amount_col] if amount_col and pd.notna(row[amount_col]) else 0
                })
                
            # Check for missing amounts
        if amount_col and pd.isna(row[amount_col]):
                row_issues.append({
                    'issue_type': 'missing_amount',
                    'severity': 'critical',
                    'description': 'Transaction with missing amount',
                    'potential_loss': 0  # Unknown amount
                })
                
            # Check for zero or negative amounts
            if amount_col and pd.notna(row[amount_col]) and row[amount_col] <= 0:
                row_issues.append({
                    'issue_type': 'invalid_amount',
                    'severity': 'medium',
                    'description': f'Invalid amount: {row[amount_col]}',
                    'potential_loss': abs(row[amount_col])
                })
                
            # Check for duplicate transactions
            if customer_col and amount_col and date_col:
                duplicates = self._find_duplicate_transactions(df, row, customer_col, amount_col, date_col)
                if duplicates:
                    row_issues.append({
                        'issue_type': 'duplicate_charge',
                        'severity': 'medium',
                        'description': f'Potential duplicate transaction',
                        'potential_loss': row[amount_col] if pd.notna(row[amount_col]) else 0
                    })
                    
            # Check for unusual amounts (outliers)
            if amount_col and pd.notna(row[amount_col]):
                if self._is_amount_outlier(df, row[amount_col], amount_col):
                    row_issues.append({
                        'issue_type': 'unusual_amount',
                        'severity': 'low',
                        'description': f'Unusual amount: {row[amount_col]}',
                        'potential_loss': 0
                    })
                    
            # Add all issues for this row
            for issue in row_issues:
                issues.append({
                    'row_index': idx,
                    'customer': row[customer_col] if customer_col else 'Unknown',
                    'date': row[date_col] if date_col else 'Unknown',
                    'amount': row[amount_col] if amount_col else 'Unknown',
                    'status': row[status_col] if status_col else 'Unknown',
                    **issue
                })
                
        return pd.DataFrame(issues)
        
    def _detect_support_leaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect support-related revenue leaks"""
        issues = []
        
        # Find relevant columns
        customer_col = self._find_customer_column(df)
        ticket_col = self._find_ticket_column(df)
        status_col = self._find_status_column(df)
        date_col = self._find_date_column(df)
        
        # Group by customer to analyze patterns
        if customer_col:
            customer_groups = df.groupby(customer_col)
            
            for customer, group in customer_groups:
                # Check for high ticket volume
                if len(group) > 5:  # Threshold for high support usage
                    issues.append({
                        'row_index': group.index[0],
                        'customer': customer,
                        'issue_type': 'high_support_volume',
                        'severity': 'medium',
                        'description': f'High support volume: {len(group)} tickets',
                        'potential_loss': len(group) * 50  # Estimate $50 per support interaction
                    })
                    
                # Check for escalation patterns
                if status_col:
                    escalated_tickets = group[group[status_col].str.contains('escalat|urgent|critical', case=False, na=False)]
                    if len(escalated_tickets) > 2:
                        issues.append({
                            'row_index': group.index[0],
                            'customer': customer,
                            'issue_type': 'escalation_pattern',
                            'severity': 'high',
                            'description': f'Multiple escalations: {len(escalated_tickets)} tickets',
                            'potential_loss': len(escalated_tickets) * 100  # Higher cost for escalations
                        })
                        
        return pd.DataFrame(issues)
        
    def _detect_operations_leaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect operations-related revenue leaks"""
        issues = []
        
        # Find relevant columns
        product_col = self._find_product_column(df)
        inventory_col = self._find_inventory_column(df)
        delivery_col = self._find_delivery_column(df)
        
        # Check for inventory mismatches
        if inventory_col and product_col:
            for idx, row in df.iterrows():
                if pd.notna(row[inventory_col]) and row[inventory_col] < 0:
                    issues.append({
                        'row_index': idx,
                        'product': row[product_col] if product_col else 'Unknown',
                        'issue_type': 'inventory_mismatch',
                        'severity': 'high',
                        'description': f'Negative inventory: {row[inventory_col]}',
                        'potential_loss': abs(row[inventory_col]) * 50  # Estimate cost
                    })
                    
        # Check for delivery issues
        if delivery_col:
            delivery_issues = df[df[delivery_col].str.contains('failed|delayed|returned', case=False, na=False)]
            for idx, row in delivery_issues.iterrows():
                issues.append({
                    'row_index': idx,
                    'issue_type': 'delivery_issue',
                    'severity': 'medium',
                    'description': f'Delivery issue: {row[delivery_col]}',
                    'potential_loss': 25  # Estimate delivery cost
                })
                
        return pd.DataFrame(issues)
        
    def _detect_generic_leaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generic leak detection for unknown data types"""
        issues = []
        
        # Check for missing data
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > len(df) * 0.1:  # More than 10% missing
                issues.append({
                    'row_index': 0,
                    'issue_type': 'missing_data',
                    'severity': 'medium',
                    'description': f'High missing data in column: {col}',
                    'potential_loss': 0
                })
                
        # Check for duplicate rows
        if df.duplicated().any():
            issues.append({
                'row_index': 0,
                'issue_type': 'duplicate_data',
                'severity': 'low',
                'description': 'Duplicate rows detected',
                'potential_loss': 0
            })
            
        return pd.DataFrame(issues)
        
    def _find_amount_column(self, df: pd.DataFrame) -> str:
        """Find the amount/price column"""
        amount_keywords = ['amount', 'price', 'cost', 'charge', 'total', 'value']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in amount_keywords):
                return col
        return None
        
    def _find_status_column(self, df: pd.DataFrame) -> str:
        """Find the status column"""
        status_keywords = ['status', 'state', 'condition', 'result']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in status_keywords):
                return col
        return None
        
    def _find_customer_column(self, df: pd.DataFrame) -> str:
        """Find the customer column"""
        customer_keywords = ['customer', 'client', 'user', 'account', 'name']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in customer_keywords):
                return col
        return None
        
    def _find_date_column(self, df: pd.DataFrame) -> str:
        """Find the date column"""
        date_keywords = ['date', 'time', 'created', 'updated', 'timestamp']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                return col
        return None
        
    def _find_ticket_column(self, df: pd.DataFrame) -> str:
        """Find the ticket column"""
        ticket_keywords = ['ticket', 'case', 'issue', 'request']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ticket_keywords):
                return col
        return None
        
    def _find_product_column(self, df: pd.DataFrame) -> str:
        """Find the product column"""
        product_keywords = ['product', 'item', 'sku', 'goods']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in product_keywords):
                return col
        return None
        
    def _find_inventory_column(self, df: pd.DataFrame) -> str:
        """Find the inventory column"""
        inventory_keywords = ['inventory', 'stock', 'quantity', 'qty']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in inventory_keywords):
                return col
        return None
        
    def _find_delivery_column(self, df: pd.DataFrame) -> str:
        """Find the delivery column"""
        delivery_keywords = ['delivery', 'shipping', 'status', 'tracking']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in delivery_keywords):
                return col
        return None
        
    def _is_failed_charge(self, status: Any) -> bool:
        """Check if a charge failed"""
        if pd.isna(status):
            return False
        failed_keywords = ['fail', 'decline', 'error', 'invalid', 'rejected']
        return any(keyword in str(status).lower() for keyword in failed_keywords)
        
    def _find_duplicate_transactions(self, df: pd.DataFrame, row: pd.Series, customer_col: str, amount_col: str, date_col: str) -> List[int]:
        """Find potential duplicate transactions"""
        if pd.isna(row[customer_col]) or pd.isna(row[amount_col]) or pd.isna(row[date_col]):
            return []
            
        # Look for transactions with same customer, amount, and within 24 hours
        same_customer = df[df[customer_col] == row[customer_col]]
        same_amount = same_customer[same_customer[amount_col] == row[amount_col]]
        
        # Check for transactions within 24 hours
        try:
            # Convert dates to datetime for comparison
            same_amount_copy = same_amount.copy()
            same_amount_copy['date_dt'] = pd.to_datetime(same_amount_copy[date_col], errors='coerce')
            row_date = pd.to_datetime(row[date_col], errors='coerce')
            
            if pd.notna(row_date):
                date_threshold = pd.Timedelta(hours=24)
                duplicates = same_amount_copy[
                    (pd.notna(same_amount_copy['date_dt'])) &
                    (abs(same_amount_copy['date_dt'] - row_date) <= date_threshold)
                ]
                return duplicates.index.tolist()
        except Exception:
            # If date parsing fails, return empty list
            pass
        
        return []
        
    def _is_amount_outlier(self, df: pd.DataFrame, amount: float, amount_col: str) -> bool:
        """Check if an amount is an outlier using IQR method"""
        if pd.isna(amount):
            return False
            
        Q1 = df[amount_col].quantile(0.25)
        Q3 = df[amount_col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return amount < lower_bound or amount > upper_bound
        
    def calculate_total_potential_loss(self, issues_df: pd.DataFrame) -> float:
        """Calculate total potential revenue loss"""
        if issues_df.empty:
            return 0.0
        return issues_df['potential_loss'].sum()
        
    def get_issue_summary(self, issues_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary of detected issues"""
        if issues_df.empty:
            return {
                'total_issues': 0,
                'total_potential_loss': 0,
                'issues_by_severity': {},
                'issues_by_type': {}
            }
            
        summary = {
            'total_issues': len(issues_df),
            'total_potential_loss': self.calculate_total_potential_loss(issues_df),
            'issues_by_severity': issues_df['severity'].value_counts().to_dict(),
            'issues_by_type': issues_df['issue_type'].value_counts().to_dict()
        }
        
        return summary

# Backward compatibility
def detect_leaks(df):
    """Legacy function for backward compatibility"""
    detector = LeakDetector()
    return detector.detect_leaks(df)
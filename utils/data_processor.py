import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime, timedelta
import re

class DataProcessor:
    def __init__(self):
        self.supported_formats = {
            'csv': self._load_csv,
            'xlsx': self._load_excel,
            'xls': self._load_excel,
            'json': self._load_json
        }
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from various file formats"""
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        return self.supported_formats[file_extension](file_path)
        
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with automatic encoding detection"""
        try:
            return pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding='latin-1')
            
    def _load_excel(self, file_path: str) -> pd.DataFrame:
        """Load Excel file"""
        return pd.read_excel(file_path)
        
    def _load_json(self, file_path: str) -> pd.DataFrame:
        """Load JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
        
    def detect_data_type(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect the type of data (billing, support, operations, etc.)"""
        column_names = [col.lower() for col in df.columns]
        
        # Billing indicators
        billing_keywords = ['amount', 'charge', 'payment', 'invoice', 'subscription', 'billing', 'price', 'cost']
        billing_score = sum(1 for keyword in billing_keywords if any(keyword in col for col in column_names))
        
        # Support indicators
        support_keywords = ['ticket', 'support', 'issue', 'complaint', 'escalation', 'resolution', 'customer']
        support_score = sum(1 for keyword in support_keywords if any(keyword in col for col in column_names))
        
        # Operations indicators
        operations_keywords = ['inventory', 'delivery', 'shipping', 'order', 'product', 'quality', 'warehouse']
        operations_score = sum(1 for keyword in operations_keywords if any(keyword in col for col in column_names))
        
        scores = {
            'billing': billing_score,
            'support': support_score,
            'operations': operations_score
        }
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'unknown'
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        df_clean = df.copy()
        
        # Remove completely empty rows and columns
        df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all')
        
        # Standardize column names
        df_clean.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df_clean.columns]
        
        # Convert date columns
        date_columns = [col for col in df_clean.columns if 'date' in col or 'time' in col]
        for col in date_columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            
        # Convert amount/price columns to numeric
        amount_columns = [col for col in df_clean.columns if any(keyword in col for keyword in ['amount', 'price', 'cost', 'charge'])]
        for col in amount_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
        return df_clean
        
    def extract_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract relevant features for analysis"""
        features = {}
        
        # Basic statistics
        features['total_rows'] = len(df)
        features['total_columns'] = len(df.columns)
        
        # Date range if available
        date_columns = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
        if date_columns:
            features['date_range'] = {
                'start': df[date_columns[0]].min(),
                'end': df[date_columns[0]].max()
            }
            
        # Amount statistics if available
        amount_columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and any(keyword in col for keyword in ['amount', 'price', 'cost'])]
        if amount_columns:
            features['amount_stats'] = {
                'total': df[amount_columns].sum().sum(),
                'mean': df[amount_columns].mean().mean(),
                'median': df[amount_columns].median().median(),
                'std': df[amount_columns].std().mean()
            }
            
        # Missing data analysis
        features['missing_data'] = {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': df.columns[df.isnull().any()].tolist()
        }
        
        return features
        
    def validate_data(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate data quality and return issues"""
        issues = []
        warnings = []
        
        # Check for missing data
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            warnings.append(f"Missing data found in columns: {missing_cols}")
            
        # Check for duplicate rows
        if df.duplicated().any():
            issues.append("Duplicate rows detected")
            
        # Check for empty columns
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            issues.append(f"Empty columns detected: {empty_cols}")
            
        # Check for data type inconsistencies
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric data is stored as strings
                numeric_pattern = re.compile(r'^\d+(\.\d+)?$')
                if df[col].dropna().apply(lambda x: isinstance(x, str) and numeric_pattern.match(x)).any():
                    warnings.append(f"Column '{col}' contains numeric data stored as strings")
                    
        return {'issues': issues, 'warnings': warnings}
        
    def prepare_for_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for leakage analysis"""
        df_prepared = self.clean_data(df)
        
        # Add derived columns for analysis
        if 'amount' in df_prepared.columns or 'price' in df_prepared.columns:
            amount_col = next((col for col in ['amount', 'price'] if col in df_prepared.columns), None)
            if amount_col:
                df_prepared['amount_category'] = pd.cut(
                    df_prepared[amount_col], 
                    bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                    labels=['Micro', 'Small', 'Medium', 'Large', 'XL', 'Enterprise']
                )
                
        # Add time-based features
        date_columns = [col for col in df_prepared.columns if df_prepared[col].dtype == 'datetime64[ns]']
        if date_columns:
            date_col = date_columns[0]
            df_prepared['day_of_week'] = df_prepared[date_col].dt.day_name()
            df_prepared['month'] = df_prepared[date_col].dt.month
            df_prepared['quarter'] = df_prepared[date_col].dt.quarter
            
        return df_prepared 
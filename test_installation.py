#!/usr/bin/env python3
"""
ZeroLeak.AI Installation Test
Verifies that all components are working correctly
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported")
        
        import plotly.express as px
        print("✅ Plotly imported")
        
        import altair as alt
        print("✅ Altair imported")
        
        import requests
        print("✅ Requests imported")
        
        import openpyxl
        print("✅ OpenPyXL imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    
    try:
        from config import DEFAULT_PROVIDER, DEFAULT_MODEL, LEAKAGE_RULES
        print(f"✅ Configuration loaded: {DEFAULT_PROVIDER}, {DEFAULT_MODEL}")
        print(f"✅ Leakage rules loaded: {len(LEAKAGE_RULES)} categories")
        return True
    except ImportError as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_data_processor():
    """Test data processor functionality"""
    print("\n🔍 Testing data processor...")
    
    try:
        from utils.data_processor import DataProcessor
        
        processor = DataProcessor()
        
        # Create test data
        test_data = pd.DataFrame({
            'customer': ['Alice', 'Bob', 'Charlie'],
            'amount': [100, 200, 300],
            'status': ['success', 'failed', 'success'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        
        # Test data type detection
        data_type = processor.detect_data_type(test_data)
        print(f"✅ Data type detected: {data_type}")
        
        # Test data cleaning
        cleaned_data = processor.clean_data(test_data)
        print(f"✅ Data cleaned: {len(cleaned_data)} rows")
        
        # Test feature extraction
        features = processor.extract_features(cleaned_data)
        print(f"✅ Features extracted: {len(features)} features")
        
        return True
    except Exception as e:
        print(f"❌ Data processor error: {e}")
        return False

def test_leak_detector():
    """Test leak detector functionality"""
    print("\n🔍 Testing leak detector...")
    
    try:
        from agents.leak_detector import LeakDetector
        
        detector = LeakDetector()
        
        # Create test data with issues
        test_data = pd.DataFrame({
            'customer': ['Alice', 'Bob', 'Charlie', 'David'],
            'amount': [100, None, 300, -50],
            'status': ['success', 'success', 'failed', 'success'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
        })
        
        # Test leak detection
        issues = detector.detect_leaks(test_data, 'billing')
        print(f"✅ Leaks detected: {len(issues)} issues")
        
        # Test issue summary
        summary = detector.get_issue_summary(issues)
        print(f"✅ Issue summary generated: {summary['total_issues']} total issues")
        
        return True
    except Exception as e:
        print(f"❌ Leak detector error: {e}")
        return False

def test_insight_agent():
    """Test insight agent functionality"""
    print("\n🔍 Testing insight agent...")
    
    try:
        from agents.insight_agent import InsightAgent
        
        agent = InsightAgent()
        
        # Create test issues
        test_issues = pd.DataFrame({
            'issue_type': ['failed_charge', 'missing_amount'],
            'severity': ['high', 'critical'],
            'potential_loss': [100, 0],
            'description': ['Test issue 1', 'Test issue 2']
        })
        
        # Test summary generation (without LLM call)
        summary = agent._generate_executive_summary(test_issues)
        print("✅ Executive summary generated")
        
        # Test risk assessment
        risk = agent._generate_risk_assessment(test_issues)
        print("✅ Risk assessment generated")
        
        return True
    except Exception as e:
        print(f"❌ Insight agent error: {e}")
        return False

def test_billing_agent():
    """Test billing agent functionality"""
    print("\n🔍 Testing billing agent...")
    
    try:
        from agents.billing_agent import BillingAgent
        
        agent = BillingAgent()
        
        # Create test billing data
        test_data = pd.DataFrame({
            'customer': ['Alice', 'Bob', 'Charlie'],
            'amount': [100, 200, 300],
            'status': ['success', 'failed', 'success'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        
        # Test payment pattern analysis
        analysis = agent.analyze_payment_patterns(test_data)
        print(f"✅ Payment patterns analyzed: {len(analysis)} categories")
        
        # Test anomaly detection
        anomalies = agent.detect_billing_anomalies(test_data)
        print(f"✅ Anomalies detected: {len(anomalies)} anomalies")
        
        return True
    except Exception as e:
        print(f"❌ Billing agent error: {e}")
        return False

def test_llm_utils():
    """Test LLM utilities (without making API calls)"""
    print("\n🔍 Testing LLM utilities...")
    
    try:
        from utils.llm_utils import LLMClient
        
        # Test client initialization
        client = LLMClient('openrouter', 'mistral-7b')
        print("✅ LLM client initialized")
        
        # Test configuration access
        config = client.config
        print(f"✅ LLM config loaded: {len(config)} settings")
        
        return True
    except Exception as e:
        print(f"❌ LLM utilities error: {e}")
        return False

def test_sample_data():
    """Test sample data loading"""
    print("\n🔍 Testing sample data...")
    
    try:
        sample_file = Path("data/sample_stripe.csv")
        if sample_file.exists():
            sample_data = pd.read_csv(sample_file)
            print(f"✅ Sample data loaded: {len(sample_data)} rows")
            return True
        else:
            print("⚠️  Sample data file not found")
            return True  # Not critical
    except Exception as e:
        print(f"❌ Sample data error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 Running ZeroLeak.AI Installation Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration,
        test_data_processor,
        test_leak_detector,
        test_insight_agent,
        test_billing_agent,
        test_llm_utils,
        test_sample_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ZeroLeak.AI is ready to use.")
        print("\nNext steps:")
        print("1. Configure your API keys in .env file")
        print("2. Run: streamlit run main.py")
        print("3. Open http://localhost:8501 in your browser")
        return True
    else:
        print("❌ Some tests failed. Please check your installation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 
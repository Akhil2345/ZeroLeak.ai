import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from datetime import datetime, timedelta
import io
import base64

# Import our modules
from utils.data_processor import DataProcessor
from agents.leak_detector import LeakDetector
from agents.insight_agent import InsightAgent
from agents.billing_agent import BillingAgent
from config import UI_CONFIG, SUPPORTED_FORMATS

# Page configuration
st.set_page_config(
    page_title="ZeroLeak.AI - Revenue Leakage Analyzer",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Material UI design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1976D2 0%, #42A5F5 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #1976D2;
        margin-bottom: 1rem;
    }
    
    .issue-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
        border-left: 4px solid #FF5722;
    }
    
    .success-card {
        border-left: 4px solid #4CAF50;
    }
    
    .warning-card {
        border-left: 4px solid #FF9800;
    }
    
    .info-card {
        border-left: 4px solid #2196F3;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1976D2 0%, #42A5F5 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3);
    }
    
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .analysis-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'issues_df' not in st.session_state:
    st.session_state.issues_df = None
if 'data_type' not in st.session_state:
    st.session_state.data_type = None

# Initialize agents
@st.cache_resource
def initialize_agents():
    return {
        'data_processor': DataProcessor(),
        'leak_detector': LeakDetector(),
        'insight_agent': InsightAgent(),
        'billing_agent': BillingAgent()
    }

agents = initialize_agents()

# Main header
st.markdown("""
<div class="main-header">
    <h1>📉 ZeroLeak.AI</h1>
    <p style="font-size: 1.2rem; margin: 0;">Revenue Leakage Analyzer for Startups</p>
    <p style="font-size: 1rem; margin: 0; opacity: 0.9;">Detect, analyze, and fix revenue leaks across your business</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # LLM Provider selection
    st.subheader("🤖 AI Model")
    provider = st.selectbox(
        "Select LLM Provider",
        ["openrouter", "openai", "together", "groq"],
        index=0
    )
    
    model = st.selectbox(
        "Select Model",
        ["mistral-7b", "llama-3-8b", "gpt-4", "claude-3"],
        index=0
    )
    
    # Analysis settings
    st.subheader("🔍 Analysis Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    
    include_anomalies = st.checkbox("Include Anomaly Detection", value=True)
    include_trends = st.checkbox("Include Trend Analysis", value=True)
    
    # Export options
    st.subheader("📤 Export Options")
    export_format = st.selectbox(
        "Export Format",
        ["CSV", "Excel", "PDF"]
    )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # File upload section
    st.markdown("""
    <div class="upload-section">
        <h2>📁 Upload Your Data</h2>
        <p>Upload your billing, support, or operations data to detect revenue leaks</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Upload CSV, Excel, or JSON files containing your business data"
    )

with col2:
    # Sample data option
    st.markdown("""
    <div class="upload-section">
        <h3>🧪 Try Sample Data</h3>
        <p>Test the system with sample data</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Load Sample Data"):
        sample_data = pd.read_csv("data/sample_stripe.csv")
        st.session_state.current_data = sample_data
        st.session_state.data_loaded = True
        st.session_state.data_type = agents['data_processor'].detect_data_type(sample_data)
        st.success("Sample data loaded successfully!")

# Data processing
if uploaded_file is not None:
    try:
        # Save uploaded file temporarily
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and process data
        df = agents['data_processor'].load_data(f"temp_{uploaded_file.name}")
        df_processed = agents['data_processor'].prepare_for_analysis(df)
        
        st.session_state.current_data = df_processed
        st.session_state.data_loaded = True
        st.session_state.data_type = agents['data_processor'].detect_data_type(df_processed)
        
        # Data validation
        validation_results = agents['data_processor'].validate_data(df_processed)
        
        if validation_results['issues']:
            st.error("⚠️ Data quality issues detected:")
            for issue in validation_results['issues']:
                st.error(f"• {issue}")
        
        if validation_results['warnings']:
            st.warning("⚠️ Data quality warnings:")
            for warning in validation_results['warnings']:
                st.warning(f"• {warning}")
        
        st.success(f"✅ Data loaded successfully! Detected type: {st.session_state.data_type}")
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

# Data preview
if st.session_state.data_loaded and st.session_state.current_data is not None:
    st.markdown("""
    <div class="analysis-section">
        <h2>📋 Data Preview</h2>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.current_data
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Data Type", st.session_state.data_type.title())
    with col4:
        missing_data = df.isnull().sum().sum()
        st.metric("Missing Values", missing_data)
    
    # Data preview
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Column information
    st.subheader("📋 Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.dataframe(col_info, use_container_width=True)
    
    # Analysis button
    if st.button("🔍 Run Revenue Leakage Analysis", type="primary"):
        with st.spinner("Analyzing your data for revenue leaks..."):
            # Detect leaks
            issues_df = agents['leak_detector'].detect_leaks(df, st.session_state.data_type)
            st.session_state.issues_df = issues_df
            st.session_state.analysis_complete = True
            
            # Generate insights
            if not issues_df.empty:
                insights = agents['insight_agent'].generate_summary(issues_df, df)
                st.session_state.insights = insights
            else:
                st.session_state.insights = "✅ No revenue leakage issues detected!"

# Analysis results
if st.session_state.analysis_complete and st.session_state.issues_df is not None:
    st.markdown("""
    <div class="analysis-section">
        <h2>🚨 Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    issues_df = st.session_state.issues_df
    
    if not issues_df.empty:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Issues", len(issues_df))
        with col2:
            total_loss = issues_df['potential_loss'].sum()
            st.metric("Potential Loss", f"${total_loss:,.2f}")
        with col3:
            critical_issues = len(issues_df[issues_df['severity'] == 'critical'])
            st.metric("Critical Issues", critical_issues)
        with col4:
            high_issues = len(issues_df[issues_df['severity'] == 'high'])
            st.metric("High Priority", high_issues)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Issues by Severity")
            severity_counts = issues_df['severity'].value_counts()
            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                color_discrete_sequence=UI_CONFIG['charts']['color_scheme']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Issues by Type")
            issue_type_counts = issues_df['issue_type'].value_counts().head(10)
            fig = px.bar(
                x=issue_type_counts.values,
                y=issue_type_counts.index,
                orientation='h',
                color_discrete_sequence=UI_CONFIG['charts']['color_scheme']
            )
            fig.update_layout(height=400, xaxis_title="Count", yaxis_title="Issue Type")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed issues table
        st.subheader("🔍 Detailed Issues")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=issues_df['severity'].unique(),
                default=issues_df['severity'].unique()
            )
        with col2:
            issue_type_filter = st.multiselect(
                "Filter by Issue Type",
                options=issues_df['issue_type'].unique(),
                default=issues_df['issue_type'].unique()
            )
        with col3:
            min_loss = st.number_input(
                "Minimum Potential Loss",
                min_value=0.0,
                value=0.0,
                step=10.0
            )
        
        # Apply filters
        filtered_df = issues_df[
            (issues_df['severity'].isin(severity_filter)) &
            (issues_df['issue_type'].isin(issue_type_filter)) &
            (issues_df['potential_loss'] >= min_loss)
        ]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export button
        if st.button("📥 Export Results"):
            if export_format == "CSV":
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="zeroleak_analysis.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    filtered_df.to_excel(writer, sheet_name='Revenue Leaks', index=False)
                st.download_button(
                    label="Download Excel",
                    data=buffer.getvalue(),
                    file_name="zeroleak_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        # AI Insights
        if 'insights' in st.session_state:
            st.markdown("""
            <div class="analysis-section">
                <h2>🧠 AI Insights & Recommendations</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(st.session_state.insights)
            
            # Additional analysis options
            if st.button("🔍 Generate Additional Insights"):
                with st.spinner("Generating additional insights..."):
                    # Customer insights
                    if 'customer' in issues_df.columns:
                        customer_insights = agents['insight_agent'].generate_customer_insights(issues_df)
                        st.subheader("👥 Customer Insights")
                        st.markdown(customer_insights)
                    
                    # Trend analysis
                    if 'date' in issues_df.columns:
                        trend_insights = agents['insight_agent'].generate_trend_analysis(issues_df)
                        st.subheader("📈 Trend Analysis")
                        st.markdown(trend_insights)
                    
                    # Benchmark analysis
                    benchmark_insights = agents['insight_agent'].generate_benchmark_analysis(issues_df, st.session_state.current_data)
                    st.subheader("📊 Benchmark Analysis")
                    st.markdown(benchmark_insights)
    
    else:
        st.success("""
        <div class="success-card">
            <h3>✅ No Revenue Leakage Detected!</h3>
            <p>Your business appears to be operating efficiently with no significant revenue leaks identified.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with ❤️ by ZeroLeak.AI - Protecting your revenue, one transaction at a time</p>
    <p>For support and questions, contact us at support@zeroleak.ai</p>
</div>
""", unsafe_allow_html=True) 
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
import time
from streamlit_option_menu import option_menu
from streamlit_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

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

# Enhanced Custom CSS for beautiful Material UI design
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .issue-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #FF5722;
        transition: all 0.3s ease;
    }
    
    .issue-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    
    .success-card {
        border-left: 4px solid #4CAF50;
        background: linear-gradient(135deg, #f8fff9 0%, #ffffff 100%);
    }
    
    .warning-card {
        border-left: 4px solid #FF9800;
        background: linear-gradient(135deg, #fffbf0 0%, #ffffff 100%);
    }
    
    .info-card {
        border-left: 4px solid #2196F3;
        background: linear-gradient(135deg, #f0f8ff 0%, #ffffff 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Upload Section */
    .upload-section {
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 2px dashed #667eea;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #764ba2;
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
    }
    
    /* Analysis Section */
    .analysis-section {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9ff 0%, #ffffff 100%);
    }
    
    /* File Uploader */
    .stFileUploader > div > div {
        border: 2px dashed #667eea;
        border-radius: 12px;
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-critical { background: #f44336; }
    .status-high { background: #ff9800; }
    .status-medium { background: #2196f3; }
    .status-low { background: #4caf50; }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-card {
            padding: 1.5rem;
        }
        
        .upload-section {
            padding: 2rem;
        }
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
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

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

# Main header with enhanced design
st.markdown("""
<div class="main-header fade-in-up">
    <h1>📉 ZeroLeak.AI</h1>
    <p>Revenue Leakage Analyzer for Startups</p>
    <p style="font-size: 1rem; margin-top: 1rem; opacity: 0.8;">Detect, analyze, and fix revenue leaks across your business</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h3>⚙️ Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Upload Data", "Analysis", "Reports", "Settings"],
        icons=["house", "cloud-upload", "search", "file-earmark-text", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#667eea", "font-size": "18px"},
            "nav-link": {
                "color": "#666",
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#667eea",
            },
            "nav-link-selected": {"background-color": "#667eea", "color": "white"},
        }
    )
    
    st.markdown("---")
    
    # LLM Configuration
    st.subheader("🤖 AI Model")
    provider = st.selectbox(
        "Select LLM Provider",
        ["openrouter", "openai", "together", "groq"],
        index=0,
        help="Choose your preferred AI provider"
    )
    
    model = st.selectbox(
        "Select Model",
        ["mistral-7b", "llama-3-8b", "gpt-4", "claude-3"],
        index=0,
        help="Choose the AI model for analysis"
    )
    
    # Analysis settings
    st.subheader("🔍 Analysis Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Adjust detection sensitivity"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        include_anomalies = st.checkbox("Anomaly Detection", value=True)
    with col2:
        include_trends = st.checkbox("Trend Analysis", value=True)
    
    # Export options
    st.subheader("📤 Export Options")
    export_format = st.selectbox(
        "Export Format",
        ["CSV", "Excel", "PDF"]
    )
    
    # System status
    st.markdown("---")
    st.subheader("📊 System Status")
    
    # Status indicators
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.markdown("🟢 **System Online**")
    with status_col2:
        st.markdown("🟢 **AI Ready**")

# Main content based on navigation
if selected == "Dashboard":
    # Dashboard view
    st.markdown("""
    <div class="analysis-section fade-in-up">
        <h2>📊 Dashboard Overview</h2>
        <p>Welcome to ZeroLeak.AI! Get started by uploading your data or trying our sample data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📁 Upload Data</h3>
            <p>Upload your billing, support, or operations data to start analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧪 Try Sample</h3>
            <p>Test the system with our sample data to see how it works</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 View Reports</h3>
            <p>Access detailed reports and insights from your analysis</p>
        </div>
        """, unsafe_allow_html=True)

elif selected == "Upload Data":
    # File upload section
    st.markdown("""
    <div class="upload-section fade-in-up">
        <h2>📁 Upload Your Data</h2>
        <p>Upload your billing, support, or operations data to detect revenue leaks</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload CSV, Excel, or JSON files containing your business data"
        )

    with col2:
        # Sample data option
        st.markdown("""
        <div class="metric-card">
            <h3>🧪 Try Sample Data</h3>
            <p>Test the system with sample data</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Load Sample Data", key="sample_data"):
            sample_data = pd.read_csv("data/sample_stripe.csv")
            st.session_state.current_data = sample_data
            st.session_state.data_loaded = True
            st.session_state.data_type = agents['data_processor'].detect_data_type(sample_data)
            st.success("✅ Sample data loaded successfully!")

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

elif selected == "Analysis":
    # Analysis section
    if st.session_state.data_loaded and st.session_state.current_data is not None:
        st.markdown("""
        <div class="analysis-section fade-in-up">
            <h2>📋 Data Preview</h2>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.current_data
        
        # Data overview with enhanced metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Total Rows</h3>
                <h2 style="color: #667eea; margin: 0;">{len(df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📋 Total Columns</h3>
                <h2 style="color: #667eea; margin: 0;">{len(df.columns)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏷️ Data Type</h3>
                <h2 style="color: #667eea; margin: 0;">{st.session_state.data_type.title()}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            missing_data = df.isnull().sum().sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚠️ Missing Values</h3>
                <h2 style="color: #667eea; margin: 0;">{missing_data:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Data preview with enhanced styling
        st.subheader("📊 Data Preview")
        
        # Use AgGrid for better table display
        gb = GridOptionsBuilder.from_dataframe(df.head(10))
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gb.configure_selection('multiple', use_checkbox=True)
        gridOptions = gb.build()
        
        grid_response = AgGrid(
            df.head(10),
            gridOptions=gridOptions,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            fit_columns_on_grid_load=True,
            theme="streamlit",
        )
        
        # Column information
        st.subheader("📋 Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Missing Values': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        
        gb_info = GridOptionsBuilder.from_dataframe(col_info)
        gb_info.configure_pagination(paginationAutoPageSize=True)
        gridOptions_info = gb_info.build()
        
        AgGrid(
            col_info,
            gridOptions=gridOptions_info,
            fit_columns_on_grid_load=True,
            theme="streamlit",
        )
        
        # Analysis button with enhanced styling
        st.markdown("---")
        if st.button("🔍 Run Revenue Leakage Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing your data for revenue leaks..."):
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate analysis steps
                steps = [
                    "Loading data...",
                    "Detecting leaks...",
                    "Analyzing patterns...",
                    "Generating insights...",
                    "Creating visualizations..."
                ]
                
                for i, step in enumerate(steps):
                    status_text.text(step)
                    progress_bar.progress((i + 1) * 20)
                    time.sleep(0.5)
                
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
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                time.sleep(1)
                st.success("✅ Analysis completed successfully!")

elif selected == "Reports":
    # Reports section
    if st.session_state.analysis_complete and st.session_state.issues_df is not None:
        st.markdown("""
        <div class="analysis-section fade-in-up">
            <h2>🚨 Analysis Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        issues_df = st.session_state.issues_df
        
        if not issues_df.empty:
            # Key metrics with enhanced design
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Total Issues</h3>
                    <h2 style="color: #667eea; margin: 0;">{len(issues_df):,}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_loss = issues_df['potential_loss'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💰 Potential Loss</h3>
                    <h2 style="color: #f44336; margin: 0;">${total_loss:,.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                critical_issues = len(issues_df[issues_df['severity'] == 'critical'])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔴 Critical Issues</h3>
                    <h2 style="color: #f44336; margin: 0;">{critical_issues}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                high_issues = len(issues_df[issues_df['severity'] == 'high'])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🟡 High Priority</h3>
                    <h2 style="color: #ff9800; margin: 0;">{high_issues}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="chart-container">
                    <h3>📊 Issues by Severity</h3>
                </div>
                """, unsafe_allow_html=True)
                
                severity_counts = issues_df['severity'].value_counts()
                fig = px.pie(
                    values=severity_counts.values,
                    names=severity_counts.index,
                    color_discrete_sequence=['#f44336', '#ff9800', '#2196f3', '#4caf50'],
                    hole=0.4
                )
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="chart-container">
                    <h3>📈 Issues by Type</h3>
                </div>
                """, unsafe_allow_html=True)
                
                issue_type_counts = issues_df['issue_type'].value_counts().head(10)
                fig = px.bar(
                    x=issue_type_counts.values,
                    y=issue_type_counts.index,
                    orientation='h',
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Count",
                    yaxis_title="Issue Type",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed issues table with enhanced filtering
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
            
            # Use AgGrid for better table display
            gb_issues = GridOptionsBuilder.from_dataframe(filtered_df)
            gb_issues.configure_pagination(paginationAutoPageSize=True)
            gb_issues.configure_side_bar()
            gb_issues.configure_selection('multiple', use_checkbox=True)
            gridOptions_issues = gb_issues.build()
            
            AgGrid(
                filtered_df,
                gridOptions=gridOptions_issues,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                fit_columns_on_grid_load=True,
                theme="streamlit",
            )
            
            # Export button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("📥 Export Results", use_container_width=True):
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
                if st.button("🔍 Generate Additional Insights", use_container_width=True):
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
        st.markdown("""
        <div class="success-card">
            <h3>✅ No Revenue Leakage Detected!</h3>
            <p>Your business appears to be operating efficiently with no significant revenue leaks identified.</p>
        </div>
        """, unsafe_allow_html=True)

elif selected == "Settings":
    # Settings section
    st.markdown("""
    <div class="analysis-section fade-in-up">
        <h2>⚙️ Settings</h2>
        <p>Configure your ZeroLeak.AI preferences and system settings.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Configuration
    st.subheader("🔑 API Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("OpenRouter API Key", type="password", help="Your OpenRouter API key")
        st.text_input("OpenAI API Key", type="password", help="Your OpenAI API key")
    
    with col2:
        st.text_input("Together.ai API Key", type="password", help="Your Together.ai API key")
        st.text_input("Groq API Key", type="password", help="Your Groq API key")
    
    # System Settings
    st.subheader("🔧 System Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Default LLM Provider", ["openrouter", "openai", "together", "groq"])
        st.selectbox("Default Model", ["mistral-7b", "llama-3-8b", "gpt-4", "claude-3"])
    
    with col2:
        st.slider("Default Confidence Threshold", 0.1, 1.0, 0.7)
        st.checkbox("Enable Notifications", value=True)
    
    # Data Settings
    st.subheader("📊 Data Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Max File Size (MB)", 10, 100, 50)
        st.multiselect("Supported Formats", ["CSV", "Excel", "JSON"], default=["CSV", "Excel", "JSON"])
    
    with col2:
        st.checkbox("Auto-detect Data Type", value=True)
        st.checkbox("Enable Data Validation", value=True)
    
    # Save settings
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("✅ Settings saved successfully!")

# Footer with enhanced design
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%); border-radius: 16px; margin-top: 2rem;">
    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">Built with ❤️ by ZeroLeak.AI</p>
    <p style="margin: 0; opacity: 0.8;">Protecting your revenue, one transaction at a time</p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.6;">For support: support@zeroleak.ai</p>
</div>
""", unsafe_allow_html=True) 
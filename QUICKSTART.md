# 🚀 ZeroLeak.AI Quick Start Guide

Get up and running with ZeroLeak.AI in 5 minutes!

## ⚡ Quick Setup

### 1. Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit the `.env` file and add at least one API key:
```bash
# OpenRouter (Recommended - free tier available)
OPENROUTER_API_KEY=your_key_here

# Or OpenAI
OPENAI_API_KEY=your_key_here

# Or Together.ai
TOGETHER_API_KEY=your_key_here

# Or Groq
GROQ_API_KEY=your_key_here
```

### 3. Run the Application
```bash
streamlit run main.py
```

Open your browser to: **http://localhost:8501**

## 🎯 First Steps

### 1. Try Sample Data
- Click "📊 Load Sample Data" to test the system
- This will load example billing data with known issues

### 2. Upload Your Data
- Supported formats: CSV, Excel (xlsx, xls), JSON
- The system automatically detects data type (billing, support, operations)

### 3. Run Analysis
- Click "🔍 Run Revenue Leakage Analysis"
- View real-time results and insights

## 📊 Understanding Results

### Executive Summary
- **Total Issues**: Number of revenue leaks detected
- **Potential Loss**: Estimated revenue at risk
- **Critical Issues**: Require immediate attention

### Issue Types
- **Failed Charges**: Payment processing failures
- **Missing Amounts**: Transactions without pricing
- **Duplicate Transactions**: Potential double charges
- **Invalid Amounts**: Zero or negative amounts
- **Unusual Patterns**: Statistical outliers

### Severity Levels
- 🔴 **Critical**: Immediate action required (24-48 hours)
- 🟡 **High**: Address within 1 week
- 🟠 **Medium**: Address within 2 weeks
- 🟢 **Low**: Monitor and address as needed

## 🔧 Configuration Options

### LLM Provider Selection
- **OpenRouter**: Access to multiple models (recommended)
- **OpenAI**: GPT-4 and GPT-3.5-turbo
- **Together.ai**: Llama models
- **Groq**: Fast inference models

### Analysis Settings
- **Confidence Threshold**: Adjust detection sensitivity
- **Include Anomalies**: Statistical outlier detection
- **Include Trends**: Time-based pattern analysis

## 📈 Sample Data Format

### Billing Data
```csv
Customer,Date,Amount,Status
Alice,2024-06-01,99,Success
Bob,2024-06-01,,Success
Charlie,2024-06-01,199,Failed
```

### Support Data
```csv
Customer,Ticket,Status,Resolution_Time
Alice,T001,Open,2
Bob,T002,Escalated,5
Charlie,T003,Closed,1
```

### Operations Data
```csv
Product,Inventory,Delivery_Status
Widget A,50,Delivered
Widget B,-5,Failed
Widget C,100,In Transit
```

## 🎨 Features Overview

### 🔍 Multi-Agent AI Analysis
- **Leak Detector**: Identifies revenue leaks
- **Insight Agent**: Generates recommendations
- **Billing Agent**: Deep payment analysis
- **Data Processor**: Handles multiple formats

### 📊 Interactive Visualizations
- Issue distribution charts
- Severity breakdown
- Customer impact analysis
- Trend patterns

### 📤 Export Options
- **CSV**: Raw data export
- **Excel**: Formatted reports
- **PDF**: Professional reports (coming soon)

## 🆘 Troubleshooting

### Common Issues

**"No module named 'pandas'"**
```bash
# Activate virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

**"API key not configured"**
- Check your `.env` file
- Ensure at least one API key is set
- Verify the key is valid

**"File upload failed"**
- Check file format (CSV, Excel, JSON)
- Ensure file size < 50MB
- Verify file encoding (UTF-8 recommended)

### Getting Help
- **Documentation**: [docs.zeroleak.ai](https://docs.zeroleak.ai)
- **Issues**: [GitHub Issues](https://github.com/yourusername/zeroleak_ai/issues)
- **Email**: support@zeroleak.ai

## 🎉 Next Steps

1. **Upload Real Data**: Start with your billing or support data
2. **Review Issues**: Focus on critical and high-priority items
3. **Implement Fixes**: Follow AI-generated recommendations
4. **Monitor Progress**: Re-run analysis to track improvements
5. **Scale Up**: Add more data sources and team members

---

**Happy revenue leak hunting! 🚀**

*ZeroLeak.AI - Protecting your revenue, one transaction at a time.* 
# 📉 ZeroLeak.AI - Revenue Leakage Analyzer

**Protecting your revenue, one transaction at a time.**

ZeroLeak.AI is a comprehensive multi-agent AI system designed to detect, analyze, and help fix revenue leakage across early-stage and growth startups. Our system audits internal business workflows and finances to surface actionable insights that can save your business thousands in lost revenue.

## 🚀 Features

### 🔍 **Multi-Agent AI Analysis**
- **Leak Detector Agent**: Identifies revenue leaks across billing, support, and operations
- **Insight Agent**: Generates actionable recommendations and risk assessments
- **Billing Agent**: Deep analysis of payment patterns and anomalies
- **Data Processor**: Handles multiple data formats with intelligent preprocessing

### 📊 **Comprehensive Detection**
- **Billing Leaks**: Failed charges, missing amounts, duplicate transactions, pricing discrepancies
- **Support Leaks**: High churn customers, escalation patterns, resolution time issues
- **Operations Leaks**: Inventory mismatches, delivery issues, quality problems
- **Anomaly Detection**: Statistical outlier detection and pattern recognition

### 🎯 **Actionable Insights**
- Executive summaries with key metrics
- Detailed analysis by severity and type
- Risk assessment with timeline recommendations
- Customer-specific insights and retention strategies
- Trend analysis and benchmark comparisons

### 🎨 **Beautiful UI**
- Material UI design with white and blue color scheme
- Interactive charts and visualizations
- Real-time analysis with progress indicators
- Responsive design for all devices

## 🏗️ Architecture

```
zeroleak_ai/
├── main.py                 # Main Streamlit application
├── config.py              # Configuration and settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── agents/               # AI agents
│   ├── leak_detector.py  # Revenue leak detection
│   ├── insight_agent.py  # AI insights generation
│   └── billing_agent.py  # Billing analysis
├── utils/                # Utility modules
│   ├── data_processor.py # Data handling and preprocessing
│   └── llm_utils.py      # LLM integration
└── data/                 # Sample data
    └── sample_stripe.csv # Example dataset
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/zeroleak_ai.git
cd zeroleak_ai
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys
Create a `.env` file in the project root:
```bash
# Required: At least one LLM provider
OPENROUTER_API_KEY=your_openrouter_key_here
OPENAI_API_KEY=your_openai_key_here
TOGETHER_API_KEY=your_together_key_here
GROQ_API_KEY=your_groq_key_here
```

### Step 5: Run the Application
```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage

### 1. **Upload Your Data**
- Supported formats: CSV, Excel (xlsx, xls), JSON
- Maximum file size: 50MB
- The system automatically detects data type (billing, support, operations)

### 2. **Configure Analysis**
- Select your preferred LLM provider and model
- Adjust confidence thresholds
- Enable/disable specific analysis features

### 3. **Run Analysis**
- Click "Run Revenue Leakage Analysis"
- View real-time progress and results
- Explore interactive visualizations

### 4. **Review Results**
- **Executive Summary**: High-level metrics and key findings
- **Detailed Analysis**: Issues categorized by severity and type
- **AI Insights**: Actionable recommendations and risk assessment
- **Export Options**: Download results in CSV, Excel, or PDF format

## 🔧 Configuration

### LLM Providers
ZeroLeak.AI supports multiple LLM providers:

- **OpenRouter**: Access to multiple models (Mistral, Llama, GPT-4, Claude)
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Together.ai**: Llama models
- **Groq**: Fast inference models

### Analysis Settings
- **Confidence Threshold**: Adjust sensitivity of leak detection
- **Anomaly Detection**: Enable statistical outlier detection
- **Trend Analysis**: Include time-based pattern analysis

## 📊 Sample Data

The system includes sample data to test functionality:

```csv
Customer,Date,Amount,Status
Alice,2024-06-01,99,Success
Bob,2024-06-01,,Success
Charlie,2024-06-01,199,Failed
```

## 🎯 Supported Data Types

### Billing Data
- Transaction amounts and statuses
- Customer information
- Payment dates and methods
- Invoice details

### Support Data
- Ticket information
- Customer interactions
- Resolution times
- Escalation patterns

### Operations Data
- Inventory levels
- Delivery status
- Product information
- Quality metrics

## 🔍 Detection Capabilities

### Revenue Leak Detection
- **Failed Charges**: Identify failed payment attempts
- **Missing Amounts**: Find transactions with missing pricing
- **Duplicate Transactions**: Detect potential double charges
- **Invalid Amounts**: Flag zero or negative amounts
- **Unusual Patterns**: Statistical outlier detection

### Risk Assessment
- **Critical Issues**: Immediate attention required
- **High Priority**: Address within 1 week
- **Medium Priority**: Address within 2 weeks
- **Low Priority**: Monitor and address as needed

### Customer Analysis
- **High-Risk Customers**: Identify customers with multiple issues
- **Revenue Concentration**: Analyze customer revenue distribution
- **Churn Indicators**: Detect customers at risk of leaving

## 📈 Output and Reports

### Executive Summary
- Total issues detected
- Potential revenue loss
- Critical and high-priority issues
- Top issue types

### Detailed Analysis
- Issues by severity and type
- Customer impact analysis
- Time-based patterns
- Statistical summaries

### AI-Generated Insights
- Actionable recommendations
- Risk assessment
- Customer retention strategies
- Process improvement suggestions

### Export Options
- **CSV**: Raw data export
- **Excel**: Formatted reports with multiple sheets
- **PDF**: Professional reports (coming soon)

## 🚀 Advanced Features

### Multi-Provider LLM Support
Switch between different AI providers for optimal performance and cost:

```python
# Configure in config.py
DEFAULT_PROVIDER = "openrouter"  # or "openai", "together", "groq"
DEFAULT_MODEL = "mistral-7b"     # Model specific to provider
```

### Custom Detection Rules
Modify detection sensitivity in `config.py`:

```python
LEAKAGE_RULES = {
    "billing": {
        "failed_charges": {"weight": 0.8, "description": "Failed payment attempts"},
        "missing_amounts": {"weight": 0.9, "description": "Transactions with missing amounts"},
        # Add custom rules here
    }
}
```

### API Integration
Integrate ZeroLeak.AI into your existing systems:

```python
from agents.leak_detector import LeakDetector
from agents.insight_agent import InsightAgent

# Initialize agents
detector = LeakDetector()
insight_agent = InsightAgent()

# Analyze data
issues = detector.detect_leaks(your_dataframe)
insights = insight_agent.generate_summary(issues)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/zeroleak_ai.git
cd zeroleak_ai

# Create development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 .
black .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs.zeroleak.ai](https://docs.zeroleak.ai)
- **Issues**: [GitHub Issues](https://github.com/yourusername/zeroleak_ai/issues)
- **Email**: support@zeroleak.ai
- **Discord**: [Join our community](https://discord.gg/zeroleak)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the beautiful UI
- Powered by multiple LLM providers for intelligent analysis
- Inspired by the need to help startups protect their revenue

---

**Made with ❤️ for the startup community**

*ZeroLeak.AI - Protecting your revenue, one transaction at a time.* 
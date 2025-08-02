import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-api-key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# LLM Provider Configuration
LLM_PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "models": {
            "mistral-7b": "mistralai/mistral-7b-instruct",
            "llama-3-8b": "meta-llama/llama-3-8b-instruct",
            "gpt-4": "openai/gpt-4",
            "claude-3": "anthropic/claude-3-sonnet"
        }
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "models": {
            "gpt-4": "gpt-4",
            "gpt-3.5-turbo": "gpt-3.5-turbo"
        }
    },
    "together": {
        "base_url": "https://api.together.xyz/v1/chat/completions",
        "models": {
            "llama-3-8b": "meta-llama/Llama-3-8B-Instruct",
            "llama-3-70b": "meta-llama/Llama-3-70B-Instruct"
        }
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "models": {
            "llama-3-8b": "llama3-8b-8192",
            "llama-3-70b": "llama3-70b-8192",
            "mixtral": "mixtral-8x7b-32768"
        }
    }
}

# Default Configuration
DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "mistral-7b"

# Revenue Leakage Detection Rules
LEAKAGE_RULES = {
    "billing": {
        "failed_charges": {"weight": 0.8, "description": "Failed payment attempts"},
        "partial_refunds": {"weight": 0.6, "description": "Partial refunds without clear reason"},
        "missing_amounts": {"weight": 0.9, "description": "Transactions with missing amounts"},
        "duplicate_charges": {"weight": 0.7, "description": "Duplicate transactions"},
        "incorrect_pricing": {"weight": 0.8, "description": "Pricing discrepancies"}
    },
    "support": {
        "high_churn": {"weight": 0.9, "description": "Customers with high support tickets"},
        "escalation_patterns": {"weight": 0.7, "description": "Frequent escalations"},
        "resolution_time": {"weight": 0.6, "description": "Long resolution times"}
    },
    "operations": {
        "inventory_mismatch": {"weight": 0.8, "description": "Inventory vs billing mismatches"},
        "delivery_issues": {"weight": 0.7, "description": "Failed or delayed deliveries"},
        "quality_issues": {"weight": 0.8, "description": "Product quality complaints"}
    }
}

# UI Configuration
UI_CONFIG = {
    "theme": {
        "primary_color": "#1976D2",
        "secondary_color": "#42A5F5",
        "background_color": "#FFFFFF",
        "text_color": "#212121",
        "accent_color": "#FFC107"
    },
    "charts": {
        "color_scheme": ["#1976D2", "#42A5F5", "#90CAF9", "#BBDEFB", "#E3F2FD"]
    }
}

# File Upload Configuration
SUPPORTED_FORMATS = {
    "csv": ["csv"],
    "excel": ["xlsx", "xls"],
    "json": ["json"]
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB 
#!/usr/bin/env python3
"""
ZeroLeak.AI Setup Script
Helps users install and configure the system
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Print the ZeroLeak.AI banner"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    📉 ZeroLeak.AI Setup                     ║
║              Revenue Leakage Analyzer for Startups          ║
╚══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def create_virtual_environment():
    """Create virtual environment"""
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return
    
    print("🔧 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        sys.exit(1)

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    # Determine the correct pip path
    if os.name == 'nt':  # Windows
        pip_path = "venv\\Scripts\\pip"
    else:  # Unix/Linux/Mac
        pip_path = "venv/bin/pip"
    
    try:
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def create_env_file():
    """Create .env file from template"""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    print("🔧 Creating .env file...")
    
    env_content = """# ZeroLeak.AI Configuration
# Add your API keys below

# LLM Provider API Keys (you need at least one)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
TOGETHER_API_KEY=your_together_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Application Settings
DEFAULT_PROVIDER=openrouter
DEFAULT_MODEL=mistral-7b
CONFIDENCE_THRESHOLD=0.7
INCLUDE_ANOMALIES=true
INCLUDE_TRENDS=true
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ .env file created")
    print("⚠️  Please edit .env file and add your API keys")

def check_directories():
    """Check and create necessary directories"""
    directories = ["data", "logs", "exports"]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir()
            print(f"✅ Created directory: {directory}")

def run_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        import pandas as pd
        import streamlit as st
        import plotly.express as px
        print("✅ All required packages imported successfully")
        
        # Test configuration
        from config import DEFAULT_PROVIDER, DEFAULT_MODEL
        print(f"✅ Configuration loaded: {DEFAULT_PROVIDER}, {DEFAULT_MODEL}")
        
        # Test data processor
        from utils.data_processor import DataProcessor
        processor = DataProcessor()
        print("✅ Data processor initialized")
        
        # Test agents
        from agents.leak_detector import LeakDetector
        from agents.insight_agent import InsightAgent
        print("✅ AI agents initialized")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check your installation")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
    
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("""
🎉 Setup completed successfully!

Next steps:
1. Edit the .env file and add your API keys
2. Activate the virtual environment:
   - Windows: venv\\Scripts\\activate
   - Unix/Linux/Mac: source venv/bin/activate
3. Run the application:
   streamlit run main.py
4. Open your browser to http://localhost:8501

For help and documentation, visit: https://github.com/yourusername/zeroleak_ai

Happy revenue leak hunting! 🚀
    """)

def main():
    """Main setup function"""
    print_banner()
    
    print("🔍 Checking system requirements...")
    check_python_version()
    
    print("\n🔧 Setting up ZeroLeak.AI...")
    create_virtual_environment()
    install_dependencies()
    create_env_file()
    check_directories()
    
    print("\n🧪 Verifying installation...")
    if run_tests():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Please check the installation.")
        sys.exit(1)
    
    print_next_steps()

if __name__ == "__main__":
    main() 
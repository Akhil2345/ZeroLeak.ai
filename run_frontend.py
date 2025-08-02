#!/usr/bin/env python3
"""
ZeroLeak.AI Frontend Runner
Launches the beautiful HTML frontend with API integration
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def print_banner():
    """Print the ZeroLeak.AI banner"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    📉 ZeroLeak.AI Frontend                   ║
║              Revenue Leakage Analyzer for Startups          ║
╚══════════════════════════════════════════════════════════════╝
    """)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import pandas
        import streamlit
        import plotly
        import altair
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def setup_environment():
    """Setup the environment for the frontend"""
    print("🔧 Setting up environment...")
    
    # Change to the frontend directory
    frontend_dir = Path(__file__).parent / "frontend"
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    os.chdir(frontend_dir)
    print(f"✅ Changed to directory: {frontend_dir}")
    return True

def start_server(port=8080):
    """Start the HTTP server"""
    print(f"🚀 Starting server on port {port}...")
    
    try:
        # Start the server
        server_process = subprocess.Popen([
            sys.executable, "server.py", "--port", str(port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Check if server is running
        if server_process.poll() is None:
            print(f"✅ Server started successfully!")
            print(f"🌐 Open your browser to: http://localhost:{port}")
            print("⏹️  Press Ctrl+C to stop the server")
            
            # Open browser automatically
            try:
                webbrowser.open(f"http://localhost:{port}")
                print("🌐 Browser opened automatically")
            except:
                print("⚠️  Could not open browser automatically")
            
            return server_process
        else:
            stdout, stderr = server_process.communicate()
            print(f"❌ Server failed to start:")
            print(stderr.decode())
            return None
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

def main():
    """Main function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Setup environment
    if not setup_environment():
        return
    
    # Get port from command line or use default
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("⚠️  Invalid port number, using default port 8080")
    
    # Start server
    server_process = start_server(port)
    
    if server_process:
        try:
            # Keep the script running
            server_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            server_process.terminate()
            server_process.wait()
            print("✅ Server stopped")
    else:
        print("❌ Failed to start server")

if __name__ == "__main__":
    main() 
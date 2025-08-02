#!/usr/bin/env python3
"""
Simple HTTP server for ZeroLeak.AI frontend
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
import sys
import urllib.parse
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_processor import DataProcessor
from agents.leak_detector import LeakDetector
from agents.insight_agent import InsightAgent
from agents.billing_agent import BillingAgent

class ZeroLeakHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Initialize our agents
        self.data_processor = DataProcessor()
        self.leak_detector = LeakDetector()
        self.insight_agent = InsightAgent()
        self.billing_agent = BillingAgent()
        
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        """Handle POST requests for API endpoints"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            if self.path == '/api/analyze':
                self.handle_analyze_request(post_data)
            elif self.path == '/api/upload':
                self.handle_upload_request(post_data)
            else:
                self.send_error(404, "API endpoint not found")
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")
    
    def handle_analyze_request(self, post_data):
        """Handle data analysis requests"""
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            # Convert data to DataFrame
            import pandas as pd
            df = pd.DataFrame(data['data'])
            
            # Detect data type
            data_type = self.data_processor.detect_data_type(df)
            
            # Clean and prepare data
            df_processed = self.data_processor.prepare_for_analysis(df)
            
            # Detect leaks
            issues_df = self.leak_detector.detect_leaks(df_processed, data_type)
            
            # Generate insights
            if not issues_df.empty:
                insights = self.insight_agent.generate_summary(issues_df, df_processed)
            else:
                insights = "✅ No revenue leakage issues detected!"
            
            # Prepare response
            response = {
                'success': True,
                'data_type': data_type,
                'total_issues': len(issues_df),
                'issues': issues_df.to_dict('records') if not issues_df.empty else [],
                'insights': insights,
                'summary': {
                    'total_issues': len(issues_df),
                    'total_loss': issues_df['potential_loss'].sum() if not issues_df.empty else 0,
                    'critical_issues': len(issues_df[issues_df['severity'] == 'critical']) if not issues_df.empty else 0,
                    'high_issues': len(issues_df[issues_df['severity'] == 'high']) if not issues_df.empty else 0
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error(500, f"Analysis error: {str(e)}")
    
    def handle_upload_request(self, post_data):
        """Handle file upload requests"""
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            # For now, just return success
            response = {
                'success': True,
                'message': 'File uploaded successfully',
                'filename': data.get('filename', 'unknown')
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error(500, f"Upload error: {str(e)}")
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def end_headers(self):
        """Add CORS headers to all responses"""
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def run_server(port=8080):
    """Run the HTTP server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, ZeroLeakHandler)
    print(f"🚀 ZeroLeak.AI Frontend Server running on http://localhost:{port}")
    print("📁 Serving files from:", os.getcwd())
    print("🔗 Open your browser and navigate to the URL above")
    print("⏹️  Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        httpd.server_close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ZeroLeak.AI Frontend Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on (default: 8080)')
    parser.add_argument('--host', default='localhost', help='Host to bind to (default: localhost)')
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    📉 ZeroLeak.AI Frontend                   ║
║              Revenue Leakage Analyzer for Startups          ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    run_server(args.port) 
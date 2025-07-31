#!/usr/bin/env python3
"""
Startup script for GreenCast Agricultural Intelligence Platform
Starts both FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import time
import os
from pathlib import Path
import signal
import threading

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'streamlit', 'motor', 'beanie',
        'tensorflow', 'scikit-learn', 'xgboost', 'plotly',
        'streamlit-option-menu', 'python-multipart'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed")
    return True

def start_mongodb():
    """Start MongoDB if not running"""
    print("🗄️ Checking MongoDB...")
    
    try:
        # Check if MongoDB is running
        result = subprocess.run(['mongosh', '--eval', 'db.runCommand("ping")'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ MongoDB is running")
            return True
        else:
            print("⚠️ MongoDB is not running")
            print("🚀 Please start MongoDB manually:")
            print("   - macOS: brew services start mongodb-community")
            print("   - Linux: sudo systemctl start mongod")
            print("   - Windows: net start MongoDB")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️ MongoDB not found or not responding")
        print("📥 Please install and start MongoDB:")
        print("   - Visit: https://docs.mongodb.com/manual/installation/")
        return False

def start_backend():
    """Start FastAPI backend server"""
    print("🚀 Starting FastAPI backend...")
    
    backend_dir = Path(__file__).parent / "backend"
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Start FastAPI with uvicorn
    backend_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])
    
    return backend_process

def start_frontend():
    """Start Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    
    frontend_dir = Path(__file__).parent / "frontend"
    
    # Change to frontend directory
    os.chdir(frontend_dir)
    
    # Start Streamlit
    frontend_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        "main_complete.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ])
    
    return frontend_process

def wait_for_services():
    """Wait for services to start"""
    print("⏳ Waiting for services to start...")
    
    import requests
    import time
    
    # Wait for backend
    backend_ready = False
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                backend_ready = True
                print("✅ Backend is ready")
                break
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        if i % 5 == 0:
            print(f"⏳ Still waiting for backend... ({i+1}/30)")
    
    if not backend_ready:
        print("⚠️ Backend may not be ready yet")
    
    # Wait a bit more for frontend
    time.sleep(3)
    print("✅ Frontend should be ready")

def print_startup_info():
    """Print startup information"""
    print("\n" + "="*60)
    print("🌱 GREENCAST AGRICULTURAL INTELLIGENCE PLATFORM")
    print("="*60)
    print("🚀 Application started successfully!")
    print()
    print("📡 Backend API: http://localhost:8000")
    print("   - API Documentation: http://localhost:8000/api/v1/docs")
    print("   - Health Check: http://localhost:8000/health")
    print()
    print("🎨 Frontend Application: http://localhost:8501")
    print("   - Main Dashboard: http://localhost:8501")
    print()
    print("🔧 Features Available:")
    print("   ✅ Disease Detection with CNN")
    print("   ✅ Yield Prediction with ML")
    print("   ✅ Agricultural Alert System")
    print("   ✅ Field Logbook Management")
    print("   ✅ Analytics Dashboard")
    print()
    print("📚 Demo Login:")
    print("   - Email: any email address")
    print("   - Password: any password")
    print("   - All data is simulated for demonstration")
    print()
    print("⚠️  Press Ctrl+C to stop all services")
    print("="*60)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down GreenCast application...")
    
    # Terminate processes
    if 'backend_process' in globals():
        backend_process.terminate()
        print("✅ Backend stopped")
    
    if 'frontend_process' in globals():
        frontend_process.terminate()
        print("✅ Frontend stopped")
    
    print("👋 GreenCast application stopped successfully")
    sys.exit(0)

def main():
    """Main startup function"""
    print("🌱 Starting GreenCast Agricultural Intelligence Platform...")
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Please install missing dependencies first")
        sys.exit(1)
    
    # Check MongoDB
    if not start_mongodb():
        print("⚠️ MongoDB is not available. Some features may not work.")
        print("🔄 Continuing with limited functionality...")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start backend
        global backend_process
        backend_process = start_backend()
        
        # Start frontend
        global frontend_process  
        frontend_process = start_frontend()
        
        # Wait for services
        wait_for_services()
        
        # Print startup info
        print_startup_info()
        
        # Keep the main process running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                break
                
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                break
    
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

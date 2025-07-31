"""
Main Streamlit application for GreenCast Agricultural Intelligence Platform
"""

import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import pandas as pd
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="GreenCast - Agricultural Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_token' not in st.session_state:
    st.session_state.user_token = None
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

# Authentication functions
def authenticate_user(email: str, password: str) -> bool:
    """Authenticate user with API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            data={"username": email, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.user_token = data["access_token"]
            st.session_state.authenticated = True
            return True
        return False
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return False

def get_auth_headers():
    """Get authentication headers"""
    if st.session_state.user_token:
        return {"Authorization": f"Bearer {st.session_state.user_token}"}
    return {}

def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.user_token = None
    st.session_state.user_info = None
    st.rerun()

# Main application
def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🌱 GreenCast Agricultural Intelligence</h1>', unsafe_allow_html=True)
    
    # Authentication check
    if not st.session_state.authenticated:
        show_login_page()
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/2E8B57/FFFFFF?text=GreenCast", width=200)
        
        selected = option_menu(
            menu_title="Navigation",
            options=[
                "Dashboard",
                "Disease Detection", 
                "Yield Prediction",
                "Alert Center",
                "Field Logbook",
                "Analytics"
            ],
            icons=[
                "speedometer2",
                "bug",
                "bar-chart",
                "exclamation-triangle",
                "journal-text",
                "graph-up"
            ],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#2E8B57", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#2E8B57"},
            }
        )
        
        # User info and logout
        st.markdown("---")
        if st.session_state.user_info:
            st.write(f"👤 {st.session_state.user_info.get('full_name', 'User')}")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    
    # Page routing
    if selected == "Dashboard":
        show_dashboard()
    elif selected == "Disease Detection":
        show_disease_detection()
    elif selected == "Yield Prediction":
        show_yield_prediction()
    elif selected == "Alert Center":
        show_alert_center()
    elif selected == "Field Logbook":
        show_field_logbook()
    elif selected == "Analytics":
        show_analytics()

def show_login_page():
    """Show login page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login to GreenCast")
        
        with st.form("login_form"):
            email = st.text_input("📧 Email", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            col_login, col_register = st.columns(2)
            
            with col_login:
                login_clicked = st.form_submit_button("🚀 Login", use_container_width=True)
            
            with col_register:
                register_clicked = st.form_submit_button("📝 Register", use_container_width=True)
            
            if login_clicked:
                if email and password:
                    if authenticate_user(email, password):
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
                else:
                    st.error("⚠️ Please enter both email and password")
            
            if register_clicked:
                st.info("🚧 Registration feature coming soon!")

def show_dashboard():
    """Show main dashboard"""
    st.markdown("## 📊 Dashboard Overview")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔬 Disease Detections</h3>
            <h2>24</h2>
            <p>This month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🌾 Yield Predictions</h3>
            <h2>12</h2>
            <p>Active predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🚨 Active Alerts</h3>
            <h2>3</h2>
            <p>Require attention</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>📍 Fields</h3>
            <h2>5</h2>
            <p>Under monitoring</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Disease Detection Trends")
        
        # Sample data for disease trends
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        detections = [2, 1, 3, 0, 2, 4, 1, 2, 3, 1, 0, 2, 3, 4, 2, 1, 3, 2, 1, 4, 2, 3, 1, 2, 3, 1, 2, 4, 3, 2, 1]
        
        fig = px.line(
            x=dates, 
            y=detections,
            title="Daily Disease Detections",
            labels={'x': 'Date', 'y': 'Number of Detections'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Yield Prediction Accuracy")
        
        # Sample data for yield accuracy
        crops = ['Corn', 'Wheat', 'Soybean', 'Rice']
        accuracy = [92, 88, 85, 90]
        
        fig = px.bar(
            x=crops,
            y=accuracy,
            title="Model Accuracy by Crop Type",
            labels={'x': 'Crop Type', 'y': 'Accuracy (%)'},
            color=accuracy,
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.markdown("### 📋 Recent Activity")
    
    activity_data = [
        {"time": "2 hours ago", "activity": "Disease detected in Field A", "type": "disease", "severity": "high"},
        {"time": "4 hours ago", "activity": "Yield prediction completed for Field B", "type": "yield", "severity": "info"},
        {"time": "6 hours ago", "activity": "Pest risk alert for Field C", "type": "alert", "severity": "medium"},
        {"time": "1 day ago", "activity": "Field log entry added", "type": "log", "severity": "info"},
    ]
    
    for activity in activity_data:
        icon = "🔬" if activity["type"] == "disease" else "🌾" if activity["type"] == "yield" else "🚨" if activity["type"] == "alert" else "📝"
        st.markdown(f"**{icon} {activity['activity']}** - {activity['time']}")

if __name__ == "__main__":
    main()

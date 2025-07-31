"""
Complete Streamlit application for GreenCast Agricultural Intelligence Platform
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

# Import page modules
from pages.disease_detection import show_disease_detection
from pages.yield_prediction import show_yield_prediction
from pages.alert_center import show_alert_center
from pages.field_logbook import show_field_logbook

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
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton > button {
        background-color: #2E8B57;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #228B22;
        color: white;
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
    st.session_state.user_info = {
        'full_name': 'John Farmer',
        'email': 'john@example.com',
        'role': 'farmer'
    }

# Authentication functions
def authenticate_user(email: str, password: str) -> bool:
    """Authenticate user with API"""
    try:
        # Mock authentication for demo
        if email and password:
            st.session_state.user_token = "mock_token_123"
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
        # Logo/Brand
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #2E8B57; border-radius: 0.5rem; margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0;">🌱 GreenCast</h2>
            <p style="color: #e8f5e8; margin: 0; font-size: 0.9rem;">Agricultural Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
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
        
        # User profile card
        if st.session_state.user_info:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center;">
                    <div style="background-color: #2E8B57; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 0.5rem; font-weight: bold;">
                        {st.session_state.user_info['full_name'][0]}
                    </div>
                    <div>
                        <div style="font-weight: bold; font-size: 0.9rem;">{st.session_state.user_info['full_name']}</div>
                        <div style="font-size: 0.8rem; color: #666;">{st.session_state.user_info['role'].title()}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
        
        # Quick stats
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("🔬 Detections Today", "3", delta="1")
        st.metric("🚨 Active Alerts", "5", delta="-2")
        st.metric("🌾 Fields Monitored", "4", delta="0")
    
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
        st.markdown("""
        <div style="background-color: white; padding: 2rem; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-top: 2rem;">
            <h2 style="text-align: center; color: #2E8B57; margin-bottom: 2rem;">🔐 Welcome to GreenCast</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("#### Sign in to your account")
            
            email = st.text_input(
                "📧 Email Address", 
                placeholder="Enter your email",
                help="Use any email for demo purposes"
            )
            password = st.text_input(
                "🔒 Password", 
                type="password", 
                placeholder="Enter your password",
                help="Use any password for demo purposes"
            )
            
            col_login, col_register = st.columns(2)
            
            with col_login:
                login_clicked = st.form_submit_button("🚀 Sign In", use_container_width=True)
            
            with col_register:
                register_clicked = st.form_submit_button("📝 Register", use_container_width=True)
            
            if login_clicked:
                if email and password:
                    if authenticate_user(email, password):
                        st.success("✅ Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
                else:
                    st.error("⚠️ Please enter both email and password")
            
            if register_clicked:
                st.info("🚧 Registration feature coming soon! For now, use any email/password to login.")
        
        # Demo instructions
        st.markdown("""
        <div style="background-color: #e8f5e8; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
            <h4 style="color: #2E8B57; margin-top: 0;">🎯 Demo Instructions</h4>
            <p style="margin-bottom: 0;">
                This is a demo version. Enter any email and password to explore the platform.
                All data shown is simulated for demonstration purposes.
            </p>
        </div>
        """, unsafe_allow_html=True)

def show_dashboard():
    """Show main dashboard"""
    st.markdown("## 📊 Farm Dashboard Overview")
    
    # Welcome message
    if st.session_state.user_info:
        st.markdown(f"Welcome back, **{st.session_state.user_info['full_name']}**! Here's your farm status at a glance.")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E8B57; margin: 0;">🔬 Disease Detections</h3>
            <h2 style="margin: 0.5rem 0;">24</h2>
            <p style="margin: 0; color: #666;">This month (+3 from last week)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E8B57; margin: 0;">🌾 Yield Predictions</h3>
            <h2 style="margin: 0.5rem 0;">12</h2>
            <p style="margin: 0; color: #666;">Active predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E8B57; margin: 0;">🚨 Active Alerts</h3>
            <h2 style="margin: 0.5rem 0;">5</h2>
            <p style="margin: 0; color: #666;">2 high priority</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E8B57; margin: 0;">📍 Fields</h3>
            <h2 style="margin: 0.5rem 0;">4</h2>
            <p style="margin: 0; color: #666;">Under monitoring</p>
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
            title="Daily Disease Detections - January 2024",
            labels={'x': 'Date', 'y': 'Number of Detections'}
        )
        fig.update_traces(line_color="#2E8B57", marker_color="#2E8B57")
        fig.update_layout(height=350)
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
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity and alerts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Recent Activity")
        
        activity_data = [
            {"time": "2 hours ago", "activity": "Disease detected in Field A", "type": "disease", "severity": "high"},
            {"time": "4 hours ago", "activity": "Yield prediction completed for Field B", "type": "yield", "severity": "info"},
            {"time": "6 hours ago", "activity": "Pest risk alert for Field C", "type": "alert", "severity": "medium"},
            {"time": "1 day ago", "activity": "Field log entry added", "type": "log", "severity": "info"},
            {"time": "2 days ago", "activity": "Irrigation system maintenance", "type": "maintenance", "severity": "info"},
        ]
        
        for activity in activity_data:
            icon = "🔬" if activity["type"] == "disease" else "🌾" if activity["type"] == "yield" else "🚨" if activity["type"] == "alert" else "📝"
            st.markdown(f"**{icon} {activity['activity']}** - *{activity['time']}*")
    
    with col2:
        st.markdown("### 🚨 Priority Alerts")
        
        priority_alerts = [
            {"title": "High Fungal Risk", "field": "Field 1", "severity": "Critical", "time": "1 hour ago"},
            {"title": "Pest Activity Detected", "field": "Field 2", "severity": "High", "time": "3 hours ago"},
            {"title": "Soil Temperature Low", "field": "Field 3", "severity": "Medium", "time": "5 hours ago"},
        ]
        
        for alert in priority_alerts:
            severity_colors = {"Critical": "#f44336", "High": "#ff9800", "Medium": "#9c27b0", "Low": "#4caf50"}
            color = severity_colors.get(alert["severity"], "#2196f3")
            
            st.markdown(f"""
            <div style="
                background-color: {color}15;
                border-left: 4px solid {color};
                padding: 0.8rem;
                border-radius: 0.3rem;
                margin: 0.5rem 0;
            ">
                <strong>{alert['title']}</strong> - {alert['field']}<br>
                <small>{alert['severity']} • {alert['time']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Weather overview
    st.markdown("### 🌤️ Current Weather Conditions")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🌡️ Temperature", "24.5°C", delta="1.2°C")
    
    with col2:
        st.metric("💧 Humidity", "68%", delta="-5%")
    
    with col3:
        st.metric("🌧️ Rainfall", "2.3mm", delta="2.3mm")
    
    with col4:
        st.metric("💨 Wind Speed", "5.2 m/s", delta="0.8 m/s")
    
    with col5:
        st.metric("📊 Pressure", "1015 hPa", delta="2 hPa")

def show_analytics():
    """Show analytics page"""
    st.markdown("## 📈 Advanced Analytics")
    st.markdown("Comprehensive analysis of your farm data and performance metrics.")
    
    # Time period selector
    period = st.selectbox(
        "Analysis Period",
        options=["Last 7 days", "Last 30 days", "Last 3 months", "Last year"],
        index=1
    )
    
    # Performance overview
    st.markdown("### 🎯 Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔬 Detection Accuracy", "91.2%", delta="2.1%")
    
    with col2:
        st.metric("🌾 Yield Accuracy", "87.5%", delta="1.8%")
    
    with col3:
        st.metric("🚨 Alert Response Time", "2.3 hrs", delta="-0.5 hrs")
    
    with col4:
        st.metric("📊 System Uptime", "99.8%", delta="0.1%")
    
    # Detailed analytics charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Field Performance Comparison")
        
        fields = ["Field 1", "Field 2", "Field 3", "Field 4"]
        performance_scores = [92, 87, 89, 85]
        
        fig = px.bar(
            x=fields,
            y=performance_scores,
            title="Overall Field Performance Scores",
            color=performance_scores,
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🔍 Issue Resolution Trends")
        
        months = ["Oct", "Nov", "Dec", "Jan"]
        resolved = [45, 52, 48, 58]
        pending = [8, 6, 12, 7]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Resolved', x=months, y=resolved, marker_color='#2E8B57'))
        fig.add_trace(go.Bar(name='Pending', x=months, y=pending, marker_color='#ff9800'))
        
        fig.update_layout(
            title="Issue Resolution by Month",
            barmode='stack',
            xaxis_title="Month",
            yaxis_title="Number of Issues"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ROI Analysis
    st.markdown("### 💰 Return on Investment Analysis")
    
    roi_data = {
        "Category": ["Disease Prevention", "Yield Optimization", "Resource Efficiency", "Labor Savings"],
        "Investment ($)": [5000, 8000, 3000, 4000],
        "Savings ($)": [12000, 18000, 7500, 9000],
        "ROI (%)": [140, 125, 150, 125]
    }
    
    roi_df = pd.DataFrame(roi_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            roi_df,
            x="Category",
            y="ROI (%)",
            title="ROI by Category",
            color="ROI (%)",
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(roi_df, use_container_width=True)
    
    # Recommendations
    st.markdown("### 💡 AI-Powered Recommendations")
    
    recommendations = [
        {
            "priority": "High",
            "category": "Disease Management",
            "recommendation": "Increase fungicide application frequency in Field 1 based on recent weather patterns",
            "impact": "Reduce disease risk by 25%"
        },
        {
            "priority": "Medium", 
            "category": "Yield Optimization",
            "recommendation": "Adjust irrigation schedule for Field 3 to optimize water usage",
            "impact": "Improve yield by 8-12%"
        },
        {
            "priority": "Low",
            "category": "Resource Management",
            "recommendation": "Consider soil testing for Field 4 to optimize fertilizer application",
            "impact": "Reduce fertilizer costs by 15%"
        }
    ]
    
    for rec in recommendations:
        priority_colors = {"High": "#f44336", "Medium": "#ff9800", "Low": "#4caf50"}
        color = priority_colors.get(rec["priority"], "#2196f3")
        
        st.markdown(f"""
        <div style="
            background-color: {color}15;
            border-left: 4px solid {color};
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        ">
            <div style="display: flex; justify-content: between; align-items: center;">
                <strong style="color: {color};">{rec['priority']} Priority - {rec['category']}</strong>
            </div>
            <p style="margin: 0.5rem 0;">{rec['recommendation']}</p>
            <small style="color: #666;"><strong>Expected Impact:</strong> {rec['impact']}</small>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

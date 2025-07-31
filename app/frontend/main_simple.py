"""
Simplified Streamlit application for GreenCast Agricultural Intelligence Platform
No authentication required - direct access to features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from PIL import Image
import io
from datetime import datetime, timedelta
import random

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
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-info {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .feature-card {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

def show_dashboard():
    """Dashboard with overview metrics and charts"""
    st.markdown("## 📊 Farm Overview Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E8B57; margin: 0;">🌾 Total Fields</h3>
            <h2 style="margin: 0.5rem 0;">12</h2>
            <p style="color: #666; margin: 0;">Active monitoring</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E8B57; margin: 0;">🌡️ Avg Temperature</h3>
            <h2 style="margin: 0.5rem 0;">24°C</h2>
            <p style="color: #666; margin: 0;">Optimal range</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E8B57; margin: 0;">💧 Soil Moisture</h3>
            <h2 style="margin: 0.5rem 0;">68%</h2>
            <p style="color: #666; margin: 0;">Good levels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E8B57; margin: 0;">⚠️ Active Alerts</h3>
            <h2 style="margin: 0.5rem 0;">3</h2>
            <p style="color: #666; margin: 0;">Requires attention</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Weekly Temperature Trend")
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
        temps = [20 + random.uniform(-3, 8) for _ in range(len(dates))]
        
        fig = px.line(x=dates, y=temps, title="Temperature (°C)")
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💧 Soil Moisture Levels")
        fields = [f"Field {i+1}" for i in range(6)]
        moisture = [random.uniform(40, 80) for _ in range(6)]
        
        fig = px.bar(x=fields, y=moisture, title="Moisture (%)")
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

def show_disease_detection():
    """Disease detection page with image upload"""
    st.markdown("## 🔬 Plant Disease Detection")
    
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 AI-Powered Disease Detection</h3>
        <p>Upload an image of your plant to detect potential diseases using our trained CNN model.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a plant image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the plant leaf or affected area"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("### 🔍 Analysis Results")
            
            # Simulate disease detection results
            diseases = [
                ("Healthy", 0.75, "✅"),
                ("Leaf Spot", 0.15, "⚠️"),
                ("Rust", 0.08, "🟡"),
                ("Blight", 0.02, "🔴")
            ]
            
            for disease, confidence, icon in diseases:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; margin: 0.25rem 0; background-color: #f8f9fa; border-radius: 0.25rem;">
                    <span>{icon} {disease}</span>
                    <span><strong>{confidence:.1%}</strong></span>
                </div>
                """, unsafe_allow_html=True)
            
            if diseases[0][1] < 0.5:  # If not healthy
                st.markdown("""
                <div class="alert-warning">
                    <strong>⚠️ Disease Detected!</strong><br>
                    Recommended actions:<br>
                    • Apply appropriate fungicide<br>
                    • Improve air circulation<br>
                    • Monitor closely for spread
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-info">
                    <strong>✅ Plant appears healthy!</strong><br>
                    Continue regular monitoring and maintenance.
                </div>
                """, unsafe_allow_html=True)

def show_yield_prediction():
    """Yield prediction page"""
    st.markdown("## 📊 Crop Yield Prediction")
    
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 ML-Based Yield Forecasting</h3>
        <p>Predict crop yields using weather data, soil conditions, and historical patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌾 Crop Information")
        crop_type = st.selectbox("Crop Type", ["Wheat", "Corn", "Rice", "Soybeans", "Tomatoes"])
        field_size = st.number_input("Field Size (acres)", min_value=1, max_value=1000, value=50)
        planting_date = st.date_input("Planting Date", value=datetime.now() - timedelta(days=60))
    
    with col2:
        st.markdown("### 🌡️ Environmental Conditions")
        avg_temp = st.slider("Average Temperature (°C)", 15, 35, 24)
        rainfall = st.slider("Monthly Rainfall (mm)", 0, 200, 75)
        soil_ph = st.slider("Soil pH", 5.0, 8.0, 6.5)
    
    if st.button("🔮 Predict Yield", type="primary"):
        # Simulate yield prediction
        base_yield = {
            "Wheat": 3.2,
            "Corn": 9.8,
            "Rice": 4.5,
            "Soybeans": 2.8,
            "Tomatoes": 45.0
        }
        
        # Apply environmental factors
        temp_factor = 1.0 if 20 <= avg_temp <= 28 else 0.85
        rain_factor = 1.0 if 50 <= rainfall <= 120 else 0.9
        ph_factor = 1.0 if 6.0 <= soil_ph <= 7.0 else 0.95
        
        predicted_yield = base_yield[crop_type] * temp_factor * rain_factor * ph_factor * field_size
        confidence = random.uniform(0.82, 0.95)
        
        st.markdown("---")
        st.markdown("### 📈 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Yield", f"{predicted_yield:.1f} tons", f"+{random.uniform(5, 15):.1f}%")
        
        with col2:
            st.metric("Confidence Level", f"{confidence:.1%}", "High")
        
        with col3:
            st.metric("Expected Revenue", f"${predicted_yield * 250:.0f}", f"+${random.uniform(1000, 3000):.0f}")
        
        # Yield comparison chart
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        historical = [predicted_yield * random.uniform(0.8, 1.2) for _ in months]
        predicted = [predicted_yield] * len(months)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=historical, name="Historical Average", line=dict(color='lightblue')))
        fig.add_trace(go.Scatter(x=months, y=predicted, name="Predicted", line=dict(color='green', dash='dash')))
        fig.update_layout(title="Yield Comparison", yaxis_title="Yield (tons)", height=400)
        
        st.plotly_chart(fig, use_container_width=True)

def show_alert_center():
    """Alert center page"""
    st.markdown("## ⚠️ Agricultural Alert Center")
    
    st.markdown("""
    <div class="feature-card">
        <h3>🚨 Real-time Monitoring Alerts</h3>
        <p>Stay informed about critical conditions affecting your crops.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Alert filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alert_type = st.selectbox("Alert Type", ["All", "Weather", "Disease", "Pest", "Irrigation"])
    
    with col2:
        priority = st.selectbox("Priority", ["All", "Critical", "Warning", "Info"])
    
    with col3:
        time_range = st.selectbox("Time Range", ["Today", "This Week", "This Month"])
    
    st.markdown("---")
    
    # Sample alerts
    alerts = [
        {
            "type": "Critical",
            "title": "🌧️ Heavy Rainfall Warning",
            "message": "Excessive rainfall predicted for next 48 hours. Risk of waterlogging in Field 3 and Field 7.",
            "time": "2 hours ago",
            "action": "Ensure proper drainage, consider protective covers"
        },
        {
            "type": "Warning", 
            "title": "🦗 Pest Activity Detected",
            "message": "Increased aphid activity detected in tomato fields. Monitor closely for population growth.",
            "time": "6 hours ago",
            "action": "Apply organic pesticide, introduce beneficial insects"
        },
        {
            "type": "Info",
            "title": "🌡️ Temperature Optimal",
            "message": "Temperature conditions are optimal for crop growth across all monitored fields.",
            "time": "1 day ago",
            "action": "Continue current management practices"
        }
    ]
    
    for alert in alerts:
        if alert["type"] == "Critical":
            alert_class = "alert-critical"
            icon = "🔴"
        elif alert["type"] == "Warning":
            alert_class = "alert-warning"
            icon = "🟡"
        else:
            alert_class = "alert-info"
            icon = "🔵"
        
        st.markdown(f"""
        <div class="{alert_class}">
            <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
                <h4 style="margin: 0; color: #333;">{icon} {alert['title']}</h4>
                <small style="color: #666;">{alert['time']}</small>
            </div>
            <p style="margin: 0.5rem 0; color: #555;">{alert['message']}</p>
            <p style="margin: 0; font-weight: bold; color: #2E8B57;">Recommended Action: {alert['action']}</p>
        </div>
        """, unsafe_allow_html=True)

def show_field_logbook():
    """Field logbook page"""
    st.markdown("## 📝 Field Activity Logbook")
    
    st.markdown("""
    <div class="feature-card">
        <h3>📋 Farm Management Records</h3>
        <p>Track all field activities, treatments, and observations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add new entry
    with st.expander("➕ Add New Entry", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            field_name = st.selectbox("Field", [f"Field {i+1}" for i in range(12)])
            activity_type = st.selectbox("Activity Type", [
                "Planting", "Irrigation", "Fertilization", "Pest Control", 
                "Disease Treatment", "Harvesting", "Soil Testing", "Other"
            ])
            activity_date = st.date_input("Date", value=datetime.now())
        
        with col2:
            weather_conditions = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Windy"])
            notes = st.text_area("Notes", placeholder="Describe the activity, observations, or treatments applied...")
            
        if st.button("💾 Save Entry"):
            st.success("✅ Entry saved successfully!")
    
    st.markdown("---")
    
    # Recent entries
    st.markdown("### 📚 Recent Entries")
    
    # Sample log entries
    log_entries = [
        {
            "date": "2024-07-30",
            "field": "Field 3",
            "activity": "Fertilization",
            "weather": "Sunny",
            "notes": "Applied nitrogen fertilizer (20-10-10) at 150 lbs/acre. Soil moisture good."
        },
        {
            "date": "2024-07-29", 
            "field": "Field 1",
            "activity": "Pest Control",
            "weather": "Cloudy",
            "notes": "Spotted aphids on tomato plants. Applied organic neem oil spray."
        },
        {
            "date": "2024-07-28",
            "field": "Field 7",
            "activity": "Irrigation",
            "weather": "Sunny",
            "notes": "Drip irrigation system activated for 4 hours. Soil moisture increased to 65%."
        }
    ]
    
    for entry in log_entries:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #2E8B57;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <strong>{entry['field']} - {entry['activity']}</strong>
                <span style="color: #666; font-size: 0.9rem;">{entry['date']} | {entry['weather']}</span>
            </div>
            <p style="margin: 0; color: #555;">{entry['notes']}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🌱 GreenCast Agricultural Intelligence</h1>', unsafe_allow_html=True)
    
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
                "Field Logbook"
            ],
            icons=[
                "speedometer2",
                "bug",
                "bar-chart",
                "exclamation-triangle",
                "journal-text"
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
        
        # App info
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <h4 style="color: #2E8B57; margin: 0;">🚀 Demo Mode</h4>
            <p style="margin: 0.5rem 0; font-size: 0.9rem; color: #666;">
                All data is simulated for demonstration purposes.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Route to selected page
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

if __name__ == "__main__":
    main()

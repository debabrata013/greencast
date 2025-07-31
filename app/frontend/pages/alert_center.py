"""
Alert Center page for Streamlit frontend
"""

import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def show_alert_center():
    """Show alert center page"""
    
    st.markdown("## 🚨 Alert Center")
    st.markdown("Monitor and manage agricultural alerts with risk level indicators.")
    
    # Alert summary cards
    show_alert_summary()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔔 Active Alerts", "📅 Forecast Alerts", "📊 Alert Analytics", "⚙️ Settings"])
    
    with tab1:
        show_active_alerts()
    
    with tab2:
        show_forecast_alerts()
    
    with tab3:
        show_alert_analytics()
    
    with tab4:
        show_alert_settings()

def show_alert_summary():
    """Show alert summary cards"""
    
    st.markdown("### 📊 Alert Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div style="
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        ">
            <h3 style="color: #f44336; margin: 0;">🔴 Critical</h3>
            <h2 style="margin: 0.5rem 0;">2</h2>
            <p style="margin: 0; font-size: 0.9rem;">Immediate action required</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        ">
            <h3 style="color: #ff9800; margin: 0;">🟠 High</h3>
            <h2 style="margin: 0.5rem 0;">5</h2>
            <p style="margin: 0; font-size: 0.9rem;">Action needed soon</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background-color: #f3e5f5;
            border-left: 4px solid #9c27b0;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        ">
            <h3 style="color: #9c27b0; margin: 0;">🟡 Medium</h3>
            <h2 style="margin: 0.5rem 0;">8</h2>
            <p style="margin: 0; font-size: 0.9rem;">Monitor closely</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        ">
            <h3 style="color: #4caf50; margin: 0;">🟢 Low</h3>
            <h2 style="margin: 0.5rem 0;">3</h2>
            <p style="margin: 0; font-size: 0.9rem;">Informational</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div style="
            background-color: #f0f2f6;
            border-left: 4px solid #2196f3;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        ">
            <h3 style="color: #2196f3; margin: 0;">📊 Total</h3>
            <h2 style="margin: 0.5rem 0;">18</h2>
            <p style="margin: 0; font-size: 0.9rem;">Active alerts</p>
        </div>
        """, unsafe_allow_html=True)

def show_active_alerts():
    """Show active alerts"""
    
    st.markdown("### 🔔 Current Active Alerts")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        severity_filter = st.selectbox(
            "Filter by Severity",
            options=["All", "Critical", "High", "Medium", "Low"]
        )
    
    with col2:
        alert_type_filter = st.selectbox(
            "Filter by Type",
            options=["All", "Fungal Risk", "Pest Risk", "Soil Temperature", "Rainfall Anomaly"]
        )
    
    with col3:
        field_filter = st.selectbox(
            "Filter by Field",
            options=["All", "Field 1", "Field 2", "Field 3", "Field 4"]
        )
    
    # Mock active alerts data
    active_alerts = [
        {
            "id": "alert_001",
            "type": "Fungal Risk",
            "severity": "Critical",
            "title": "High Fungal Disease Risk",
            "message": "Temperature > 28°C and humidity > 80% for 3+ days in Field 1",
            "field": "Field 1",
            "crop": "Tomato",
            "created": "2024-01-15 08:30",
            "expires": "2024-01-16 20:00",
            "recommendations": [
                "Apply preventive fungicide spray immediately",
                "Improve ventilation around plants",
                "Reduce irrigation frequency"
            ],
            "is_read": False,
            "location": {"lat": 41.8781, "lon": -93.0977}
        },
        {
            "id": "alert_002",
            "type": "Pest Risk",
            "severity": "High",
            "title": "Pest Activity Risk Alert",
            "message": "Weather conditions favor pest activity in Field 2",
            "field": "Field 2",
            "crop": "Corn",
            "created": "2024-01-15 06:15",
            "expires": "2024-01-16 18:00",
            "recommendations": [
                "Monitor crops for pest activity",
                "Check and maintain pest traps",
                "Consider preventive treatments"
            ],
            "is_read": True,
            "location": {"lat": 41.8781, "lon": -93.0977}
        },
        {
            "id": "alert_003",
            "type": "Soil Temperature",
            "severity": "Medium",
            "title": "Soil Temperature Alert",
            "message": "Soil temperature (12°C) below optimal range for wheat in Field 3",
            "field": "Field 3",
            "crop": "Wheat",
            "created": "2024-01-15 05:45",
            "expires": "2024-01-17 12:00",
            "recommendations": [
                "Consider soil warming techniques",
                "Monitor seed germination",
                "Use mulching to retain heat"
            ],
            "is_read": False,
            "location": {"lat": 41.8781, "lon": -93.0977}
        },
        {
            "id": "alert_004",
            "type": "Rainfall Anomaly",
            "severity": "High",
            "title": "Excessive Rainfall Alert",
            "message": "Unusual high rainfall patterns detected (45mm in 24h)",
            "field": "Field 4",
            "crop": "Rice",
            "created": "2024-01-14 22:30",
            "expires": "2024-01-16 10:00",
            "recommendations": [
                "Check drainage systems",
                "Monitor for waterlogging",
                "Prevent soil erosion"
            ],
            "is_read": True,
            "location": {"lat": 41.8781, "lon": -93.0977}
        }
    ]
    
    # Apply filters
    filtered_alerts = active_alerts.copy()
    
    if severity_filter != "All":
        filtered_alerts = [a for a in filtered_alerts if a["severity"] == severity_filter]
    
    if alert_type_filter != "All":
        filtered_alerts = [a for a in filtered_alerts if a["type"] == alert_type_filter]
    
    if field_filter != "All":
        filtered_alerts = [a for a in filtered_alerts if a["field"] == field_filter]
    
    # Display alerts
    for alert in filtered_alerts:
        show_alert_card(alert)
    
    if not filtered_alerts:
        st.info("No alerts match the selected filters.")

def show_alert_card(alert):
    """Display individual alert card"""
    
    # Determine card styling based on severity
    severity_styles = {
        "Critical": {"bg": "#ffebee", "border": "#f44336", "icon": "🔴"},
        "High": {"bg": "#fff3e0", "border": "#ff9800", "icon": "🟠"},
        "Medium": {"bg": "#f3e5f5", "border": "#9c27b0", "icon": "🟡"},
        "Low": {"bg": "#e8f5e8", "border": "#4caf50", "icon": "🟢"}
    }
    
    style = severity_styles.get(alert["severity"], severity_styles["Medium"])
    read_indicator = "📖" if alert["is_read"] else "🔔"
    
    with st.container():
        st.markdown(f"""
        <div style="
            background-color: {style['bg']};
            border-left: 4px solid {style['border']};
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: {style['border']};">
                    {style['icon']} {alert['title']} {read_indicator}
                </h4>
                <small style="color: #666;">{alert['created']}</small>
            </div>
            <p style="margin: 0.5rem 0; font-size: 1rem;">
                <strong>Field:</strong> {alert['field']} ({alert['crop']})
            </p>
            <p style="margin: 0.5rem 0;">
                {alert['message']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons and recommendations
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(f"👁️ View Details", key=f"view_{alert['id']}"):
                show_alert_details(alert)
        
        with col2:
            if not alert["is_read"]:
                if st.button(f"✅ Mark Read", key=f"read_{alert['id']}"):
                    st.success("Alert marked as read!")
        
        with col3:
            if st.button(f"✔️ Acknowledge", key=f"ack_{alert['id']}"):
                st.success("Alert acknowledged!")
        
        with col4:
            if st.button(f"🗑️ Dismiss", key=f"dismiss_{alert['id']}"):
                st.success("Alert dismissed!")

def show_alert_details(alert):
    """Show detailed alert information in modal"""
    
    with st.expander(f"📋 Alert Details - {alert['title']}", expanded=True):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Alert Information:**")
            st.write(f"**Type:** {alert['type']}")
            st.write(f"**Severity:** {alert['severity']}")
            st.write(f"**Field:** {alert['field']}")
            st.write(f"**Crop:** {alert['crop']}")
            st.write(f"**Created:** {alert['created']}")
            st.write(f"**Expires:** {alert['expires']}")
        
        with col2:
            st.markdown("**Location:**")
            # Mock map display
            st.write(f"📍 Latitude: {alert['location']['lat']}")
            st.write(f"📍 Longitude: {alert['location']['lon']}")
            
            # Weather conditions (mock)
            st.markdown("**Current Conditions:**")
            st.write("🌡️ Temperature: 29.5°C")
            st.write("💧 Humidity: 85%")
            st.write("🌧️ Rainfall: 2.3mm")
        
        st.markdown("**Recommendations:**")
        for i, rec in enumerate(alert['recommendations'], 1):
            st.write(f"{i}. {rec}")

def show_forecast_alerts():
    """Show forecast alerts"""
    
    st.markdown("### 📅 7-Day Forecast Alerts")
    
    # Date range selector
    forecast_days = st.slider(
        "Forecast Days",
        min_value=1,
        max_value=14,
        value=7,
        help="Number of days to forecast alerts"
    )
    
    # Mock forecast alerts
    forecast_alerts = []
    
    for day in range(forecast_days):
        date = datetime.now() + timedelta(days=day + 1)
        
        # Generate mock alerts for each day
        if day % 2 == 0:  # Every other day has alerts
            forecast_alerts.append({
                "date": date.strftime("%Y-%m-%d"),
                "day_name": date.strftime("%A"),
                "alerts": [
                    {
                        "type": "Pest Risk",
                        "severity": "Medium",
                        "message": "Favorable conditions for pest activity",
                        "probability": 75
                    }
                ]
            })
        
        if day == 2:  # Day 3 has multiple alerts
            forecast_alerts[-1]["alerts"].append({
                "type": "Fungal Risk",
                "severity": "High",
                "message": "High humidity and temperature conditions",
                "probability": 85
            })
    
    # Display forecast timeline
    for forecast in forecast_alerts:
        st.markdown(f"#### 📅 {forecast['day_name']}, {forecast['date']}")
        
        if forecast["alerts"]:
            for alert in forecast["alerts"]:
                severity_colors = {
                    "Critical": "#f44336",
                    "High": "#ff9800",
                    "Medium": "#9c27b0",
                    "Low": "#4caf50"
                }
                
                color = severity_colors.get(alert["severity"], "#2196f3")
                
                st.markdown(f"""
                <div style="
                    background-color: {color}15;
                    border-left: 4px solid {color};
                    padding: 1rem;
                    border-radius: 0.5rem;
                    margin: 0.5rem 0;
                ">
                    <strong>{alert['type']} - {alert['severity']}</strong><br>
                    {alert['message']}<br>
                    <small>Probability: {alert['probability']}%</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No alerts forecasted for this day")
    
    # Forecast summary chart
    st.markdown("### 📊 Forecast Alert Summary")
    
    # Create mock data for chart
    dates = [(datetime.now() + timedelta(days=i)).strftime("%m-%d") for i in range(1, forecast_days + 1)]
    alert_counts = [len(f.get("alerts", [])) for f in forecast_alerts] + [0] * (forecast_days - len(forecast_alerts))
    
    fig = px.bar(
        x=dates,
        y=alert_counts,
        title="Forecasted Alerts by Day",
        labels={"x": "Date", "y": "Number of Alerts"},
        color=alert_counts,
        color_continuous_scale="Reds"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_alert_analytics():
    """Show alert analytics and trends"""
    
    st.markdown("### 📊 Alert Analytics")
    
    # Time period selector
    time_period = st.selectbox(
        "Analysis Period",
        options=["Last 7 days", "Last 30 days", "Last 3 months", "Last year"]
    )
    
    # Alert trends over time
    st.markdown("#### 📈 Alert Trends")
    
    # Mock trend data
    if time_period == "Last 7 days":
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
        alert_counts = [3, 2, 5, 1, 4, 2, 3]
    else:
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        alert_counts = np.random.poisson(2.5, len(dates))
    
    fig = px.line(
        x=dates,
        y=alert_counts,
        title="Daily Alert Count",
        labels={"x": "Date", "y": "Number of Alerts"}
    )
    fig.update_traces(line_color="#2E8B57", marker_color="#2E8B57")
    st.plotly_chart(fig, use_container_width=True)
    
    # Alert type distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Alert Type Distribution")
        
        alert_types = ["Fungal Risk", "Pest Risk", "Soil Temperature", "Rainfall Anomaly"]
        type_counts = [25, 18, 15, 12]
        
        fig = px.pie(
            values=type_counts,
            names=alert_types,
            title="Alert Types (Last 30 Days)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### ⚠️ Severity Distribution")
        
        severities = ["Critical", "High", "Medium", "Low"]
        severity_counts = [8, 22, 28, 12]
        colors = ["#f44336", "#ff9800", "#9c27b0", "#4caf50"]
        
        fig = px.bar(
            x=severities,
            y=severity_counts,
            title="Alert Severity Levels",
            color=severities,
            color_discrete_map=dict(zip(severities, colors))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Field-wise alert analysis
    st.markdown("#### 📍 Field-wise Alert Analysis")
    
    field_data = {
        "Field": ["Field 1", "Field 2", "Field 3", "Field 4"],
        "Total Alerts": [15, 12, 18, 10],
        "Critical": [3, 1, 4, 2],
        "High": [5, 4, 6, 3],
        "Medium": [5, 5, 6, 4],
        "Low": [2, 2, 2, 1]
    }
    
    df = pd.DataFrame(field_data)
    
    fig = px.bar(
        df,
        x="Field",
        y=["Critical", "High", "Medium", "Low"],
        title="Alert Distribution by Field",
        color_discrete_map={
            "Critical": "#f44336",
            "High": "#ff9800", 
            "Medium": "#9c27b0",
            "Low": "#4caf50"
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Response time analysis
    st.markdown("#### ⏱️ Alert Response Analysis")
    
    response_data = {
        "Metric": ["Average Response Time", "Alerts Acknowledged", "Alerts Resolved", "False Positives"],
        "Value": ["2.3 hours", "85%", "78%", "12%"],
        "Target": ["< 4 hours", "> 80%", "> 75%", "< 15%"],
        "Status": ["✅ Good", "✅ Good", "✅ Good", "✅ Good"]
    }
    
    response_df = pd.DataFrame(response_data)
    st.dataframe(response_df, use_container_width=True)

def show_alert_settings():
    """Show alert configuration settings"""
    
    st.markdown("### ⚙️ Alert Settings")
    
    # Notification preferences
    st.markdown("#### 📱 Notification Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_notifications = st.checkbox("📧 Email Notifications", value=True)
        sms_notifications = st.checkbox("📱 SMS Notifications", value=False)
        push_notifications = st.checkbox("🔔 Push Notifications", value=True)
    
    with col2:
        notification_frequency = st.selectbox(
            "Notification Frequency",
            options=["Immediate", "Hourly", "Daily", "Weekly"]
        )
        
        quiet_hours = st.checkbox("🌙 Quiet Hours (10 PM - 6 AM)", value=True)
    
    # Alert thresholds
    st.markdown("#### 🎚️ Alert Thresholds")
    
    st.markdown("**Fungal Risk Thresholds:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fungal_temp = st.number_input("Temperature (°C)", value=28.0, step=0.1)
    
    with col2:
        fungal_humidity = st.number_input("Humidity (%)", value=80.0, step=1.0)
    
    with col3:
        fungal_duration = st.number_input("Duration (days)", value=3, step=1)
    
    st.markdown("**Pest Risk Thresholds:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pest_temp_min = st.number_input("Min Temperature (°C)", value=20.0, step=0.1)
    
    with col2:
        pest_temp_max = st.number_input("Max Temperature (°C)", value=30.0, step=0.1)
    
    with col3:
        pest_humidity = st.number_input("Humidity Threshold (%)", value=70.0, step=1.0)
    
    # Field-specific settings
    st.markdown("#### 📍 Field-specific Settings")
    
    field_settings = st.selectbox(
        "Configure Field",
        options=["Field 1", "Field 2", "Field 3", "Field 4"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        field_alerts_enabled = st.checkbox(f"Enable alerts for {field_settings}", value=True)
        priority_field = st.checkbox(f"High priority field", value=False)
    
    with col2:
        crop_specific = st.checkbox("Use crop-specific thresholds", value=True)
        custom_location = st.checkbox("Custom weather location", value=False)
    
    # Save settings
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("✅ Alert settings saved successfully!")
        
        # Show summary of saved settings
        with st.expander("📋 Saved Settings Summary"):
            st.write("**Notifications:**")
            st.write(f"- Email: {'Enabled' if email_notifications else 'Disabled'}")
            st.write(f"- SMS: {'Enabled' if sms_notifications else 'Disabled'}")
            st.write(f"- Push: {'Enabled' if push_notifications else 'Disabled'}")
            st.write(f"- Frequency: {notification_frequency}")
            
            st.write("**Thresholds:**")
            st.write(f"- Fungal Risk: {fungal_temp}°C, {fungal_humidity}%, {fungal_duration} days")
            st.write(f"- Pest Risk: {pest_temp_min}-{pest_temp_max}°C, {pest_humidity}%")
    
    # Reset to defaults
    if st.button("🔄 Reset to Defaults"):
        st.info("Settings reset to default values")

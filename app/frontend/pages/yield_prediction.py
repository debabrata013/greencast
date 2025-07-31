"""
Yield Prediction page for Streamlit frontend
"""

import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def show_yield_prediction():
    """Show yield prediction page"""
    
    st.markdown("## 🌾 Crop Yield Prediction")
    st.markdown("Predict crop yields based on weather, soil, and management data.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 New Prediction", "📈 Prediction History", "🔍 Field Comparison"])
    
    with tab1:
        show_prediction_form()
    
    with tab2:
        show_prediction_history()
    
    with tab3:
        show_field_comparison()

def show_prediction_form():
    """Show yield prediction form"""
    
    st.markdown("### 📝 Enter Field Data")
    
    with st.form("yield_prediction_form"):
        # Field selection
        col1, col2 = st.columns(2)
        
        with col1:
            field_id = st.selectbox(
                "📍 Select Field",
                options=["field_1", "field_2", "field_3", "field_4"],
                format_func=lambda x: f"Field {x.split('_')[1]} - 5.2 hectares",
                help="Choose the field for yield prediction"
            )
        
        with col2:
            crop_type = st.selectbox(
                "🌱 Crop Type",
                options=["corn", "wheat", "rice", "soybean", "barley", "cotton"],
                help="Select the type of crop"
            )
        
        st.markdown("#### 🌤️ Weather Conditions")
        
        # Weather data inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temperature = st.number_input(
                "🌡️ Temperature (°C)",
                min_value=-10.0,
                max_value=50.0,
                value=25.0,
                step=0.1,
                help="Average temperature during growing season"
            )
        
        with col2:
            humidity = st.number_input(
                "💧 Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=1.0,
                help="Relative humidity percentage"
            )
        
        with col3:
            rainfall = st.number_input(
                "🌧️ Rainfall (mm)",
                min_value=0.0,
                max_value=2000.0,
                value=800.0,
                step=10.0,
                help="Total rainfall during growing season"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            wind_speed = st.number_input(
                "💨 Wind Speed (m/s)",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.1,
                help="Average wind speed"
            )
        
        with col2:
            pressure = st.number_input(
                "📊 Pressure (hPa)",
                min_value=900.0,
                max_value=1100.0,
                value=1013.25,
                step=0.1,
                help="Atmospheric pressure"
            )
        
        st.markdown("#### 🌱 Soil Conditions")
        
        # Soil data inputs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            soil_ph = st.number_input(
                "⚗️ Soil pH",
                min_value=3.0,
                max_value=10.0,
                value=6.5,
                step=0.1,
                help="Soil pH level"
            )
        
        with col2:
            soil_nitrogen = st.number_input(
                "🧪 Nitrogen (ppm)",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=1.0,
                help="Soil nitrogen content"
            )
        
        with col3:
            soil_phosphorus = st.number_input(
                "🧪 Phosphorus (ppm)",
                min_value=0.0,
                max_value=50.0,
                value=15.0,
                step=1.0,
                help="Soil phosphorus content"
            )
        
        with col4:
            soil_potassium = st.number_input(
                "🧪 Potassium (ppm)",
                min_value=0.0,
                max_value=500.0,
                value=200.0,
                step=10.0,
                help="Soil potassium content"
            )
        
        soil_moisture = st.slider(
            "💧 Soil Moisture (%)",
            min_value=0,
            max_value=100,
            value=45,
            help="Current soil moisture percentage"
        )
        
        st.markdown("#### 🚜 Management Practices")
        
        # Management data inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fertilizer_amount = st.number_input(
                "🌿 Fertilizer (kg/hectare)",
                min_value=0.0,
                max_value=500.0,
                value=150.0,
                step=10.0,
                help="Amount of fertilizer applied"
            )
        
        with col2:
            irrigation_hours = st.number_input(
                "💦 Irrigation (hours/week)",
                min_value=0.0,
                max_value=168.0,
                value=20.0,
                step=1.0,
                help="Weekly irrigation hours"
            )
        
        with col3:
            pesticide_applications = st.number_input(
                "🐛 Pesticide Applications",
                min_value=0,
                max_value=20,
                value=2,
                step=1,
                help="Number of pesticide applications"
            )
        
        # Date inputs
        col1, col2 = st.columns(2)
        
        with col1:
            planting_date = st.date_input(
                "🌱 Planting Date",
                value=datetime.now() - timedelta(days=60),
                help="Date when crop was planted"
            )
        
        with col2:
            expected_harvest = st.date_input(
                "🌾 Expected Harvest Date",
                value=datetime.now() + timedelta(days=30),
                help="Expected harvest date"
            )
        
        # Additional parameters
        elevation = st.number_input(
            "⛰️ Elevation (meters)",
            min_value=0.0,
            max_value=5000.0,
            value=300.0,
            step=10.0,
            help="Field elevation above sea level"
        )
        
        notes = st.text_area(
            "📝 Notes (Optional)",
            placeholder="Add any additional observations or information...",
            help="Optional notes about field conditions"
        )
        
        # Submit button
        submit_button = st.form_submit_button(
            "🔮 Predict Yield",
            use_container_width=True
        )
        
        if submit_button:
            predict_yield(
                field_id, crop_type, temperature, humidity, rainfall, wind_speed, pressure,
                soil_ph, soil_nitrogen, soil_phosphorus, soil_potassium, soil_moisture,
                fertilizer_amount, irrigation_hours, pesticide_applications,
                planting_date, expected_harvest, elevation, notes
            )

def predict_yield(field_id, crop_type, temperature, humidity, rainfall, wind_speed, pressure,
                 soil_ph, soil_nitrogen, soil_phosphorus, soil_potassium, soil_moisture,
                 fertilizer_amount, irrigation_hours, pesticide_applications,
                 planting_date, expected_harvest, elevation, notes):
    """Predict crop yield based on input parameters"""
    
    with st.spinner("🔮 Predicting crop yield..."):
        try:
            # Mock prediction calculation
            # In production, this would call the actual API
            
            # Base yield by crop type
            base_yields = {
                "corn": 8.0,
                "wheat": 4.0,
                "rice": 5.5,
                "soybean": 3.0,
                "barley": 4.5,
                "cotton": 1.8
            }
            
            base_yield = base_yields.get(crop_type, 5.0)
            
            # Apply modifiers based on conditions
            temp_modifier = 1.0 + (temperature - 25) * 0.02
            rain_modifier = 1.0 + (rainfall - 800) / 1000
            ph_modifier = 1.0 + (soil_ph - 6.5) * 0.1
            fert_modifier = 1.0 + (fertilizer_amount - 150) / 500
            
            # Calculate predicted yield
            predicted_yield = base_yield * temp_modifier * rain_modifier * ph_modifier * fert_modifier
            predicted_yield = max(0.5, predicted_yield)  # Minimum yield
            
            # Mock confidence interval
            uncertainty = predicted_yield * 0.15
            confidence_interval = {
                "lower": max(0, predicted_yield - uncertainty),
                "upper": predicted_yield + uncertainty,
                "uncertainty_percent": 15.0
            }
            
            # Mock feature importance
            feature_importance = {
                "temperature": 0.25,
                "rainfall": 0.20,
                "soil_ph": 0.15,
                "fertilizer_amount": 0.12,
                "soil_nitrogen": 0.10,
                "humidity": 0.08,
                "irrigation_hours": 0.06,
                "soil_phosphorus": 0.04
            }
            
            mock_result = {
                "predicted_yield": predicted_yield,
                "confidence_interval": confidence_interval,
                "feature_importance": feature_importance,
                "model_used": "XGBoost",
                "processing_time": 0.8
            }
            
            show_prediction_results(mock_result, crop_type)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

def show_prediction_results(result, crop_type):
    """Display yield prediction results"""
    
    st.success("✅ Yield prediction completed!")
    
    # Main result display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🌾 Predicted Yield",
            f"{result['predicted_yield']:.2f} tons/hectare",
            help="Estimated crop yield"
        )
    
    with col2:
        st.metric(
            "📊 Confidence Range",
            f"±{result['confidence_interval']['uncertainty_percent']:.0f}%",
            help="Prediction uncertainty range"
        )
    
    with col3:
        st.metric(
            "🤖 Model Used",
            result["model_used"],
            help="Machine learning model used for prediction"
        )
    
    # Confidence interval visualization
    st.markdown("### 📊 Prediction Confidence")
    
    lower = result['confidence_interval']['lower']
    upper = result['confidence_interval']['upper']
    predicted = result['predicted_yield']
    
    fig = go.Figure()
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=[lower, predicted, upper],
        y=[1, 1, 1],
        mode='markers+lines',
        marker=dict(size=[8, 12, 8], color=['orange', 'green', 'orange']),
        line=dict(color='gray', width=2),
        name='Confidence Range'
    ))
    
    fig.update_layout(
        title="Yield Prediction with Confidence Interval",
        xaxis_title="Yield (tons/hectare)",
        yaxis=dict(showticklabels=False, showgrid=False),
        height=200,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown("### 🎯 Feature Importance")
    
    features = list(result['feature_importance'].keys())
    importance = list(result['feature_importance'].values())
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Factors Affecting Yield Prediction",
        color=importance,
        color_continuous_scale="Greens"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    recommendations = generate_recommendations(result['predicted_yield'], crop_type, result['feature_importance'])
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Prediction", use_container_width=True):
            st.success("Prediction saved to history!")
    
    with col2:
        if st.button("📤 Export Report", use_container_width=True):
            st.info("Export functionality coming soon!")
    
    with col3:
        if st.button("🔄 New Prediction", use_container_width=True):
            st.rerun()

def generate_recommendations(predicted_yield, crop_type, feature_importance):
    """Generate recommendations based on prediction results"""
    
    recommendations = []
    
    # Yield-based recommendations
    if predicted_yield < 3.0:
        recommendations.append("Consider soil testing and nutrient supplementation to improve yield")
        recommendations.append("Evaluate irrigation system efficiency and water management practices")
    elif predicted_yield > 8.0:
        recommendations.append("Excellent yield potential - maintain current management practices")
        recommendations.append("Consider expanding similar practices to other fields")
    
    # Feature importance-based recommendations
    top_feature = max(feature_importance.keys(), key=lambda k: feature_importance[k])
    
    if top_feature == "temperature":
        recommendations.append("Temperature is the key factor - monitor weather forecasts closely")
    elif top_feature == "rainfall":
        recommendations.append("Rainfall significantly impacts yield - ensure proper drainage systems")
    elif top_feature == "soil_ph":
        recommendations.append("Soil pH is critical - consider lime application if needed")
    elif top_feature == "fertilizer_amount":
        recommendations.append("Fertilizer management is key - optimize application timing and rates")
    
    # Crop-specific recommendations
    crop_recommendations = {
        "corn": "Consider nitrogen side-dressing during vegetative growth",
        "wheat": "Monitor for fungal diseases during grain filling",
        "rice": "Maintain proper water levels throughout growing season",
        "soybean": "Ensure adequate phosphorus for nodulation",
        "barley": "Watch for lodging in high-yield potential areas",
        "cotton": "Monitor heat units for optimal harvest timing"
    }
    
    if crop_type in crop_recommendations:
        recommendations.append(crop_recommendations[crop_type])
    
    return recommendations

def show_prediction_history():
    """Show yield prediction history"""
    
    st.markdown("### 📈 Prediction History")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crop_filter = st.selectbox(
            "Filter by Crop",
            options=["All", "Corn", "Wheat", "Rice", "Soybean", "Cotton"]
        )
    
    with col2:
        field_filter = st.selectbox(
            "Filter by Field",
            options=["All", "Field 1", "Field 2", "Field 3", "Field 4"]
        )
    
    with col3:
        date_range = st.selectbox(
            "Date Range",
            options=["Last 30 days", "Last 3 months", "Last year", "All time"]
        )
    
    # Mock history data
    history_data = [
        {
            "date": "2024-01-15",
            "field": "Field 1",
            "crop": "Corn",
            "predicted_yield": 8.5,
            "actual_yield": 8.2,
            "accuracy": 96.5,
            "model": "XGBoost"
        },
        {
            "date": "2024-01-10",
            "field": "Field 2", 
            "crop": "Wheat",
            "predicted_yield": 4.2,
            "actual_yield": None,
            "accuracy": None,
            "model": "Random Forest"
        },
        {
            "date": "2024-01-05",
            "field": "Field 3",
            "crop": "Soybean",
            "predicted_yield": 3.1,
            "actual_yield": 2.9,
            "accuracy": 93.5,
            "model": "XGBoost"
        },
        {
            "date": "2023-12-20",
            "field": "Field 4",
            "crop": "Rice",
            "predicted_yield": 5.8,
            "actual_yield": 6.1,
            "accuracy": 94.8,
            "model": "XGBoost"
        }
    ]
    
    df = pd.DataFrame(history_data)
    
    # Apply filters
    if crop_filter != "All":
        df = df[df["crop"] == crop_filter]
    if field_filter != "All":
        df = df[df["field"] == field_filter]
    
    # Display table
    st.dataframe(df, use_container_width=True)
    
    # Summary metrics
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(df))
        
        with col2:
            avg_yield = df["predicted_yield"].mean()
            st.metric("Avg Predicted Yield", f"{avg_yield:.1f} t/ha")
        
        with col3:
            completed = df["actual_yield"].notna().sum()
            st.metric("Completed Harvests", completed)
        
        with col4:
            if completed > 0:
                avg_accuracy = df["accuracy"].dropna().mean()
                st.metric("Avg Accuracy", f"{avg_accuracy:.1f}%")
            else:
                st.metric("Avg Accuracy", "N/A")

def show_field_comparison():
    """Show field comparison analytics"""
    
    st.markdown("### 🔍 Field Comparison")
    
    # Mock field comparison data
    fields_data = {
        "Field 1": {"area": 5.2, "avg_yield": 8.1, "predictions": 12, "crop": "Corn"},
        "Field 2": {"area": 3.8, "avg_yield": 4.3, "predictions": 8, "crop": "Wheat"},
        "Field 3": {"area": 4.5, "avg_yield": 3.2, "predictions": 10, "crop": "Soybean"},
        "Field 4": {"area": 6.1, "avg_yield": 5.9, "predictions": 15, "crop": "Rice"}
    }
    
    # Field comparison chart
    fields = list(fields_data.keys())
    yields = [fields_data[f]["avg_yield"] for f in fields]
    areas = [fields_data[f]["area"] for f in fields]
    
    fig = px.scatter(
        x=areas,
        y=yields,
        size=[fields_data[f]["predictions"] for f in fields],
        color=fields,
        title="Field Performance Comparison",
        labels={"x": "Field Area (hectares)", "y": "Average Yield (tons/hectare)"},
        hover_data={"Field": fields}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Yield trends over time
    st.markdown("#### 📅 Yield Trends")
    
    # Mock time series data
    dates = pd.date_range(start='2023-06-01', end='2024-01-31', freq='M')
    
    fig = go.Figure()
    
    for field in fields:
        # Generate mock trend data
        base_yield = fields_data[field]["avg_yield"]
        trend_data = [base_yield + np.random.normal(0, 0.5) for _ in dates]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=trend_data,
            mode='lines+markers',
            name=field,
            line=dict(width=3)
        ))
    
    fig.update_layout(
        title="Yield Trends by Field",
        xaxis_title="Date",
        yaxis_title="Yield (tons/hectare)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Field statistics table
    st.markdown("#### 📊 Field Statistics")
    
    stats_df = pd.DataFrame(fields_data).T
    stats_df.index.name = "Field"
    stats_df = stats_df.reset_index()
    
    st.dataframe(stats_df, use_container_width=True)

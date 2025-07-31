"""
Disease Detection page for Streamlit frontend
"""

import streamlit as st
import requests
import json
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

def show_disease_detection():
    """Show disease detection page"""
    
    st.markdown("## 🔬 Plant Disease Detection")
    st.markdown("Upload plant images to detect diseases and get treatment recommendations.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📸 Upload Image", "📊 Detection History", "📈 Statistics"])
    
    with tab1:
        show_image_upload()
    
    with tab2:
        show_detection_history()
    
    with tab3:
        show_detection_statistics()

def show_image_upload():
    """Show image upload interface"""
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Plant Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a plant image",
            type=['jpg', 'jpeg', 'png', 'gif'],
            help="Upload a clear image of the plant showing symptoms"
        )
        
        # Form for additional details
        with st.form("disease_detection_form"):
            # Crop type selection
            crop_type = st.selectbox(
                "🌱 Crop Type",
                options=["corn", "wheat", "rice", "soybean", "tomato", "potato", "apple", "grape"],
                help="Select the type of crop"
            )
            
            # Field selection (mock data)
            field_id = st.selectbox(
                "📍 Field (Optional)",
                options=["", "field_1", "field_2", "field_3"],
                format_func=lambda x: "Select Field" if x == "" else f"Field {x.split('_')[1]}",
                help="Associate with a specific field"
            )
            
            # Location inputs
            col_lat, col_lon = st.columns(2)
            with col_lat:
                latitude = st.number_input(
                    "📍 Latitude",
                    value=41.8781,
                    format="%.6f",
                    help="GPS latitude coordinate"
                )
            with col_lon:
                longitude = st.number_input(
                    "📍 Longitude", 
                    value=-93.0977,
                    format="%.6f",
                    help="GPS longitude coordinate"
                )
            
            # Notes
            notes = st.text_area(
                "📝 Notes (Optional)",
                placeholder="Add any observations or additional information...",
                help="Optional notes about the plant condition"
            )
            
            # Submit button
            submit_button = st.form_submit_button(
                "🔍 Analyze Image",
                use_container_width=True
            )
            
            if submit_button and uploaded_file is not None:
                analyze_image(uploaded_file, crop_type, field_id, latitude, longitude, notes)
    
    with col2:
        st.markdown("### 🖼️ Image Preview")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.markdown("**Image Information:**")
            st.write(f"📁 Filename: {uploaded_file.name}")
            st.write(f"📏 Size: {image.size}")
            st.write(f"💾 File size: {len(uploaded_file.getvalue()) / 1024:.1f} KB")
        else:
            st.info("👆 Upload an image to see preview")
            
            # Show example images
            st.markdown("**Example Images:**")
            example_col1, example_col2 = st.columns(2)
            
            with example_col1:
                st.image("https://via.placeholder.com/200x150/90EE90/000000?text=Healthy+Plant", 
                        caption="Healthy Plant", width=150)
            
            with example_col2:
                st.image("https://via.placeholder.com/200x150/FFB6C1/000000?text=Diseased+Plant", 
                        caption="Diseased Plant", width=150)

def analyze_image(uploaded_file, crop_type, field_id, latitude, longitude, notes):
    """Analyze uploaded image for disease detection"""
    
    with st.spinner("🔍 Analyzing image for diseases..."):
        try:
            # Prepare form data
            files = {"image": uploaded_file.getvalue()}
            data = {
                "crop_type": crop_type,
                "latitude": latitude,
                "longitude": longitude
            }
            
            if field_id:
                data["field_id"] = field_id
            if notes:
                data["notes"] = notes
            
            # Mock API response for demonstration
            # In production, this would call the actual API
            mock_response = {
                "id": "detection_123",
                "predicted_disease": "Corn_(maize)___Northern_Leaf_Blight",
                "confidence_score": 0.87,
                "severity_level": "high",
                "treatment_recommendations": [
                    "Apply foliar fungicide immediately",
                    "Use resistant corn varieties in future plantings",
                    "Rotate crops to break disease cycle",
                    "Remove crop residue after harvest"
                ],
                "processing_time": 1.2,
                "created_at": datetime.now().isoformat()
            }
            
            # Display results
            show_detection_results(mock_response)
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

def show_detection_results(result):
    """Display disease detection results"""
    
    st.success("✅ Analysis completed!")
    
    # Main result card
    disease_name = result["predicted_disease"].replace("___", " - ").replace("_", " ")
    confidence = result["confidence_score"] * 100
    
    # Color coding based on severity
    severity_colors = {
        "critical": "#f44336",
        "high": "#ff9800", 
        "medium": "#9c27b0",
        "low": "#4caf50"
    }
    
    severity_color = severity_colors.get(result["severity_level"], "#2196f3")
    
    st.markdown(f"""
    <div style="
        background-color: {severity_color}15;
        border-left: 4px solid {severity_color};
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    ">
        <h3 style="color: {severity_color}; margin: 0;">🦠 {disease_name}</h3>
        <p style="font-size: 1.2rem; margin: 0.5rem 0;">
            <strong>Confidence:</strong> {confidence:.1f}%
        </p>
        <p style="margin: 0;">
            <strong>Severity:</strong> {result["severity_level"].title()}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Treatment recommendations
    st.markdown("### 💊 Treatment Recommendations")
    
    for i, recommendation in enumerate(result["treatment_recommendations"], 1):
        st.markdown(f"**{i}.** {recommendation}")
    
    # Additional info
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("⏱️ Processing Time", f"{result['processing_time']:.1f}s")
    
    with col2:
        st.metric("📅 Detection Date", datetime.fromisoformat(result['created_at']).strftime("%Y-%m-%d %H:%M"))
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Save Results", use_container_width=True):
            st.success("Results saved to your history!")
    
    with col2:
        if st.button("📤 Share Results", use_container_width=True):
            st.info("Sharing functionality coming soon!")
    
    with col3:
        if st.button("🔄 Analyze Another", use_container_width=True):
            st.rerun()

def show_detection_history():
    """Show disease detection history"""
    
    st.markdown("### 📊 Detection History")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crop_filter = st.selectbox(
            "Filter by Crop",
            options=["All", "Corn", "Wheat", "Rice", "Soybean", "Tomato"]
        )
    
    with col2:
        severity_filter = st.selectbox(
            "Filter by Severity",
            options=["All", "Critical", "High", "Medium", "Low"]
        )
    
    with col3:
        date_range = st.selectbox(
            "Date Range",
            options=["Last 7 days", "Last 30 days", "Last 3 months", "All time"]
        )
    
    # Mock history data
    history_data = [
        {
            "date": "2024-01-15",
            "crop": "Corn",
            "disease": "Northern Leaf Blight",
            "confidence": 87,
            "severity": "High",
            "field": "Field A"
        },
        {
            "date": "2024-01-14", 
            "crop": "Tomato",
            "disease": "Late Blight",
            "confidence": 92,
            "severity": "Critical",
            "field": "Field B"
        },
        {
            "date": "2024-01-13",
            "crop": "Wheat",
            "disease": "Healthy",
            "confidence": 95,
            "severity": "Low",
            "field": "Field C"
        },
        {
            "date": "2024-01-12",
            "crop": "Rice",
            "disease": "Bacterial Blight",
            "confidence": 78,
            "severity": "Medium",
            "field": "Field D"
        }
    ]
    
    # Display history table
    df = pd.DataFrame(history_data)
    
    # Apply filters
    if crop_filter != "All":
        df = df[df["crop"] == crop_filter]
    if severity_filter != "All":
        df = df[df["severity"] == severity_filter]
    
    # Style the dataframe
    def style_severity(val):
        colors = {
            "Critical": "background-color: #ffebee",
            "High": "background-color: #fff3e0", 
            "Medium": "background-color: #f3e5f5",
            "Low": "background-color: #e8f5e8"
        }
        return colors.get(val, "")
    
    styled_df = df.style.applymap(style_severity, subset=["severity"])
    st.dataframe(styled_df, use_container_width=True)
    
    # Summary statistics
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", len(df))
        
        with col2:
            avg_confidence = df["confidence"].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        with col3:
            disease_count = len(df[df["disease"] != "Healthy"])
            st.metric("Diseases Found", disease_count)
        
        with col4:
            healthy_count = len(df[df["disease"] == "Healthy"])
            st.metric("Healthy Plants", healthy_count)

def show_detection_statistics():
    """Show detection statistics and analytics"""
    
    st.markdown("### 📈 Detection Analytics")
    
    # Mock statistics data
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🦠 Disease Distribution")
        
        diseases = ["Northern Leaf Blight", "Late Blight", "Bacterial Spot", "Healthy", "Rust"]
        counts = [15, 12, 8, 25, 6]
        
        fig = px.pie(
            values=counts,
            names=diseases,
            title="Disease Types Detected",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🌱 Crop Analysis")
        
        crops = ["Corn", "Tomato", "Wheat", "Rice", "Soybean"]
        detections = [20, 15, 12, 10, 9]
        
        fig = px.bar(
            x=crops,
            y=detections,
            title="Detections by Crop Type",
            color=detections,
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Confidence score distribution
    st.markdown("#### 🎯 Confidence Score Distribution")
    
    confidence_ranges = ["90-100%", "80-89%", "70-79%", "60-69%", "Below 60%"]
    confidence_counts = [25, 18, 12, 8, 3]
    
    fig = px.bar(
        x=confidence_ranges,
        y=confidence_counts,
        title="Detection Confidence Distribution",
        color=confidence_counts,
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly trend
    st.markdown("#### 📅 Monthly Detection Trend")
    
    months = ["Oct", "Nov", "Dec", "Jan"]
    monthly_detections = [45, 52, 38, 66]
    
    fig = px.line(
        x=months,
        y=monthly_detections,
        title="Disease Detections Over Time",
        markers=True
    )
    fig.update_traces(line_color="#2E8B57", marker_color="#2E8B57")
    st.plotly_chart(fig, use_container_width=True)

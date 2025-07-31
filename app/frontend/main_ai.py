"""
AI-Integrated Streamlit application for GreenCast Agricultural Intelligence Platform
Real disease detection with trained CNN model
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
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AI model
try:
    from ml_models.disease_detector import predict_plant_disease, get_treatment_advice
    AI_MODEL_AVAILABLE = True
    print("✅ AI model loaded successfully!")
except ImportError as e:
    print(f"⚠️ AI model not available: {e}")
    AI_MODEL_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="GreenCast - AI Agricultural Intelligence",
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
    .disease-result {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2E8B57;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    .treatment-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b35;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def show_disease_detection():
    """AI-powered disease detection page with real model integration"""
    st.markdown("## 🔬 AI Plant Disease Detection")
    
    if not AI_MODEL_AVAILABLE:
        st.error("🚫 AI model is not available. Please check the model installation.")
        st.info("The model requires TensorFlow and trained weights to function properly.")
        return
    
    st.markdown("""
    <div style="background-color: #e8f5e9; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;">
        <h3 style="color: #2e7d32; margin: 0;">🤖 Real AI-Powered Disease Detection</h3>
        <p style="margin: 0.5rem 0 0 0; color: #1b5e20;">
            Upload a plant image to get instant disease detection using our trained CNN model.
            Supports 38+ plant diseases across multiple crops.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📸 Choose a plant image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the plant leaf or affected area for best results"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📷 Uploaded Image")
            st.image(image, caption="Plant Image for Analysis", use_column_width=True)
            
            # Image info
            st.markdown(f"""
            <div style="background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                <strong>Image Details:</strong><br>
                📏 Size: {image.size[0]} x {image.size[1]} pixels<br>
                🎨 Mode: {image.mode}<br>
                📁 Format: {uploaded_file.type}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🔍 AI Analysis Results")
            
            # Show loading spinner while processing
            with st.spinner("🤖 AI is analyzing your plant image..."):
                try:
                    # Convert image to bytes for model
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='PNG')
                    img_bytes = img_bytes.getvalue()
                    
                    # Get AI predictions
                    predictions = predict_plant_disease(img_bytes)
                    
                    if predictions:
                        st.success("✅ Analysis completed!")
                        
                        # Display top prediction prominently
                        top_prediction = predictions[0]
                        
                        # Determine alert level
                        if 'healthy' in top_prediction['disease'].lower():
                            alert_color = "#4caf50"
                            alert_icon = "✅"
                            alert_text = "Plant appears healthy!"
                        elif top_prediction['confidence'] > 0.7:
                            alert_color = "#f44336"
                            alert_icon = "🚨"
                            alert_text = "Disease detected with high confidence!"
                        elif top_prediction['confidence'] > 0.4:
                            alert_color = "#ff9800"
                            alert_icon = "⚠️"
                            alert_text = "Possible disease detected!"
                        else:
                            alert_color = "#2196f3"
                            alert_icon = "ℹ️"
                            alert_text = "Low confidence detection"
                        
                        # Main result card
                        st.markdown(f"""
                        <div style="background-color: {alert_color}15; border: 2px solid {alert_color}; border-radius: 0.5rem; padding: 1.5rem; margin: 1rem 0;">
                            <h3 style="color: {alert_color}; margin: 0;">{alert_icon} {alert_text}</h3>
                            <h2 style="margin: 0.5rem 0; color: #333;">{top_prediction['disease']}</h2>
                            <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                                <span style="margin-right: 1rem; font-weight: bold;">Confidence:</span>
                                <div style="flex: 1; background-color: #e0e0e0; border-radius: 10px; height: 20px;">
                                    <div style="width: {top_prediction['confidence']*100}%; height: 100%; background-color: {alert_color}; border-radius: 10px;"></div>
                                </div>
                                <span style="margin-left: 1rem; font-weight: bold;">{top_prediction['confidence']:.1%}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Detailed predictions
                        st.markdown("#### 📊 Detailed Analysis")
                        
                        for i, pred in enumerate(predictions[:5]):
                            confidence_color = "#4caf50" if pred['confidence'] > 0.5 else "#ff9800" if pred['confidence'] > 0.2 else "#9e9e9e"
                            
                            st.markdown(f"""
                            <div class="disease-result">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <strong style="color: #333;">{i+1}. {pred['disease']}</strong>
                                    <span style="background-color: {confidence_color}; color: white; padding: 0.25rem 0.5rem; border-radius: 1rem; font-size: 0.8rem;">
                                        {pred['confidence']:.1%}
                                    </span>
                                </div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: {pred['confidence']*100}%; background-color: {confidence_color};"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Treatment recommendations
                        if not 'healthy' in top_prediction['disease'].lower():
                            st.markdown("---")
                            st.markdown("### 💊 Treatment Recommendations")
                            
                            treatment = get_treatment_advice(top_prediction['raw_class'])
                            
                            urgency_colors = {
                                'Critical': '#f44336',
                                'High': '#ff5722', 
                                'Medium': '#ff9800',
                                'Low': '#4caf50',
                                'None': '#9e9e9e'
                            }
                            
                            urgency_color = urgency_colors.get(treatment['urgency'], '#9e9e9e')
                            
                            st.markdown(f"""
                            <div class="treatment-card">
                                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                                    <h4 style="margin: 0; color: #d32f2f;">🏥 Treatment Plan</h4>
                                    <span style="background-color: {urgency_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.8rem; margin-left: auto;">
                                        {treatment['urgency']} Priority
                                    </span>
                                </div>
                                
                                <div style="margin-bottom: 1rem;">
                                    <strong style="color: #1976d2;">💉 Immediate Treatment:</strong><br>
                                    <span style="color: #333;">{treatment['treatment']}</span>
                                </div>
                                
                                <div>
                                    <strong style="color: #388e3c;">🛡️ Prevention:</strong><br>
                                    <span style="color: #333;">{treatment['prevention']}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        else:
                            st.markdown("---")
                            st.markdown("""
                            <div style="background-color: #e8f5e9; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #4caf50;">
                                <h4 style="color: #2e7d32; margin: 0;">🌿 Plant Health Status: Excellent!</h4>
                                <p style="margin: 0.5rem 0 0 0; color: #1b5e20;">
                                    Your plant appears to be healthy. Continue with regular care and monitoring.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    else:
                        st.error("❌ Failed to analyze the image. Please try with a different image.")
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.info("Please try uploading a different image or check your internet connection.")
    
    else:
        # Show example images and instructions
        st.markdown("### 📋 How to Use")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
                <h4 style="color: #2e7d32;">📸 1. Upload Image</h4>
                <p style="color: #666; font-size: 0.9rem;">Take a clear photo of the affected plant part</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
                <h4 style="color: #2e7d32;">🤖 2. AI Analysis</h4>
                <p style="color: #666; font-size: 0.9rem;">Our CNN model analyzes the image instantly</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
                <h4 style="color: #2e7d32;">💊 3. Get Treatment</h4>
                <p style="color: #666; font-size: 0.9rem;">Receive specific treatment recommendations</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🌱 Supported Plants & Diseases")
        
        supported_plants = {
            "🍎 Apple": ["Apple Scab", "Black Rot", "Cedar Apple Rust"],
            "🌽 Corn": ["Cercospora Leaf Spot", "Common Rust", "Northern Leaf Blight"],
            "🍇 Grape": ["Black Rot", "Esca", "Leaf Blight"],
            "🍑 Cherry": ["Powdery Mildew"],
            "🥔 Potato": ["Early Blight", "Late Blight"],
            "🍅 Tomato": ["Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot"]
        }
        
        cols = st.columns(2)
        for i, (plant, diseases) in enumerate(supported_plants.items()):
            with cols[i % 2]:
                disease_list = ", ".join(diseases)
                st.markdown(f"""
                <div style="background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;">
                    <strong style="color: #2e7d32;">{plant}</strong><br>
                    <span style="color: #666; font-size: 0.9rem;">{disease_list}</span>
                </div>
                """, unsafe_allow_html=True)

def show_dashboard():
    """Simple dashboard overview"""
    st.markdown("## 📊 Farm Overview Dashboard")
    
    # AI Model Status
    model_status = "🟢 Online" if AI_MODEL_AVAILABLE else "🔴 Offline"
    model_color = "#4caf50" if AI_MODEL_AVAILABLE else "#f44336"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #2E8B57; margin: 0;">🤖 AI Model</h3>
            <h2 style="margin: 0.5rem 0; color: {model_color};">{model_status}</h2>
            <p style="color: #666; margin: 0;">Disease Detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E8B57; margin: 0;">🌾 Supported Plants</h3>
            <h2 style="margin: 0.5rem 0;">38+</h2>
            <p style="color: #666; margin: 0;">Disease classes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E8B57; margin: 0;">🎯 Accuracy</h3>
            <h2 style="margin: 0.5rem 0;">94%</h2>
            <p style="color: #666; margin: 0;">Model performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2E8B57; margin: 0;">⚡ Speed</h3>
            <h2 style="margin: 0.5rem 0;">&lt;3s</h2>
            <p style="color: #666; margin: 0;">Analysis time</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if AI_MODEL_AVAILABLE:
        st.success("🎉 AI Disease Detection is ready! Upload plant images to get instant analysis.")
    else:
        st.warning("⚠️ AI model is not loaded. Please check the installation and restart the application.")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    st.markdown("""
    1. **📸 Navigate to Disease Detection** - Click on "Disease Detection" in the sidebar
    2. **🖼️ Upload Plant Image** - Choose a clear photo of your plant
    3. **🤖 Get AI Analysis** - Receive instant disease detection results
    4. **💊 Follow Treatment** - Get specific treatment recommendations
    """)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🌱 GreenCast AI Agricultural Intelligence</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        # Logo/Brand
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #2E8B57; border-radius: 0.5rem; margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0;">🌱 GreenCast AI</h2>
            <p style="color: #e8f5e8; margin: 0; font-size: 0.9rem;">Agricultural Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title="Navigation",
            options=[
                "Dashboard",
                "Disease Detection"
            ],
            icons=[
                "speedometer2",
                "bug"
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
        
        # Model status
        st.markdown("---")
        
        if AI_MODEL_AVAILABLE:
            st.markdown("""
            <div style="background-color: #e8f5e9; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                <h4 style="color: #2e7d32; margin: 0;">🤖 AI Status</h4>
                <p style="margin: 0.5rem 0; font-size: 0.9rem; color: #1b5e20;">
                    <strong>✅ Model Loaded</strong><br>
                    Ready for disease detection
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #ffebee; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                <h4 style="color: #c62828; margin: 0;">🤖 AI Status</h4>
                <p style="margin: 0.5rem 0; font-size: 0.9rem; color: #b71c1c;">
                    <strong>❌ Model Offline</strong><br>
                    Check installation
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Route to selected page
    if selected == "Dashboard":
        show_dashboard()
    elif selected == "Disease Detection":
        show_disease_detection()

if __name__ == "__main__":
    main()

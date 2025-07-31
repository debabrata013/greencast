"""
Field Logbook page for Streamlit frontend
"""

import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image
import io

def show_field_logbook():
    """Show field logbook page"""
    
    st.markdown("## 📝 Field Logbook")
    st.markdown("Record field observations, activities, and maintain detailed logs with images and notes.")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["➕ New Entry", "📋 Log Entries", "📊 Activity Summary", "🔍 Search & Filter"])
    
    with tab1:
        show_new_entry_form()
    
    with tab2:
        show_log_entries()
    
    with tab3:
        show_activity_summary()
    
    with tab4:
        show_search_filter()

def show_new_entry_form():
    """Show new log entry form"""
    
    st.markdown("### ➕ Create New Log Entry")
    
    with st.form("new_log_entry"):
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            field_id = st.selectbox(
                "📍 Field",
                options=["field_1", "field_2", "field_3", "field_4"],
                format_func=lambda x: f"Field {x.split('_')[1]} - 5.2 hectares",
                help="Select the field for this log entry"
            )
        
        with col2:
            entry_type = st.selectbox(
                "📂 Entry Type",
                options=[
                    "Observation",
                    "Treatment",
                    "Planting",
                    "Harvest",
                    "Irrigation",
                    "Fertilization",
                    "Pest Control",
                    "Disease Management",
                    "Weather Event",
                    "Equipment Maintenance",
                    "Other"
                ],
                help="Type of log entry"
            )
        
        # Title and description
        title = st.text_input(
            "📝 Entry Title",
            placeholder="Brief description of the activity or observation...",
            help="Short, descriptive title for the log entry"
        )
        
        description = st.text_area(
            "📄 Detailed Description",
            placeholder="Provide detailed information about the activity, observations, conditions, etc...",
            height=150,
            help="Comprehensive description of the log entry"
        )
        
        # Location and conditions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            latitude = st.number_input(
                "📍 Latitude",
                value=41.8781,
                format="%.6f",
                help="GPS latitude coordinate"
            )
        
        with col2:
            longitude = st.number_input(
                "📍 Longitude",
                value=-93.0977,
                format="%.6f",
                help="GPS longitude coordinate"
            )
        
        with col3:
            is_important = st.checkbox(
                "⭐ Mark as Important",
                help="Flag this entry as important for easy reference"
            )
        
        # Weather conditions
        st.markdown("#### 🌤️ Weather Conditions (Optional)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temperature = st.number_input(
                "🌡️ Temperature (°C)",
                value=None,
                help="Temperature at time of entry"
            )
        
        with col2:
            humidity = st.number_input(
                "💧 Humidity (%)",
                value=None,
                min_value=0.0,
                max_value=100.0,
                help="Relative humidity"
            )
        
        with col3:
            rainfall = st.number_input(
                "🌧️ Rainfall (mm)",
                value=None,
                min_value=0.0,
                help="Rainfall amount"
            )
        
        with col4:
            wind_speed = st.number_input(
                "💨 Wind Speed (m/s)",
                value=None,
                min_value=0.0,
                help="Wind speed"
            )
        
        # Image uploads
        st.markdown("#### 📸 Images (Optional)")
        
        uploaded_images = st.file_uploader(
            "Upload Images",
            type=['jpg', 'jpeg', 'png', 'gif'],
            accept_multiple_files=True,
            help="Upload photos related to this log entry"
        )
        
        # Display uploaded images preview
        if uploaded_images:
            st.markdown("**Image Previews:**")
            cols = st.columns(min(len(uploaded_images), 4))
            
            for i, img in enumerate(uploaded_images):
                with cols[i % 4]:
                    image = Image.open(img)
                    st.image(image, caption=img.name, use_column_width=True)
        
        # Tags
        tags_input = st.text_input(
            "🏷️ Tags (Optional)",
            placeholder="Enter tags separated by commas (e.g., irrigation, corn, fertilizer)",
            help="Add tags to categorize and search entries easily"
        )
        
        # Submit button
        submit_button = st.form_submit_button(
            "💾 Save Log Entry",
            use_container_width=True
        )
        
        if submit_button:
            if title and description:
                save_log_entry(
                    field_id, entry_type, title, description,
                    latitude, longitude, is_important,
                    temperature, humidity, rainfall, wind_speed,
                    uploaded_images, tags_input
                )
            else:
                st.error("⚠️ Please provide both title and description")

def save_log_entry(field_id, entry_type, title, description, latitude, longitude, 
                  is_important, temperature, humidity, rainfall, wind_speed,
                  uploaded_images, tags_input):
    """Save log entry to database"""
    
    with st.spinner("💾 Saving log entry..."):
        try:
            # Process tags
            tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()] if tags_input else []
            
            # Mock save operation
            # In production, this would call the actual API
            
            entry_data = {
                "id": f"entry_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "field_id": field_id,
                "entry_type": entry_type,
                "title": title,
                "description": description,
                "location": {"latitude": latitude, "longitude": longitude},
                "is_important": is_important,
                "weather": {
                    "temperature": temperature,
                    "humidity": humidity,
                    "rainfall": rainfall,
                    "wind_speed": wind_speed
                } if any([temperature, humidity, rainfall, wind_speed]) else None,
                "images": [img.name for img in uploaded_images] if uploaded_images else [],
                "tags": tags,
                "created_at": datetime.now().isoformat()
            }
            
            st.success("✅ Log entry saved successfully!")
            
            # Show saved entry summary
            with st.expander("📋 Saved Entry Summary", expanded=True):
                st.write(f"**Title:** {title}")
                st.write(f"**Type:** {entry_type}")
                st.write(f"**Field:** {field_id}")
                st.write(f"**Important:** {'Yes' if is_important else 'No'}")
                if tags:
                    st.write(f"**Tags:** {', '.join(tags)}")
                if uploaded_images:
                    st.write(f"**Images:** {len(uploaded_images)} uploaded")
            
        except Exception as e:
            st.error(f"❌ Failed to save log entry: {str(e)}")

def show_log_entries():
    """Show existing log entries"""
    
    st.markdown("### 📋 Field Log Entries")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        field_filter = st.selectbox(
            "Filter by Field",
            options=["All", "Field 1", "Field 2", "Field 3", "Field 4"]
        )
    
    with col2:
        type_filter = st.selectbox(
            "Filter by Type",
            options=["All", "Observation", "Treatment", "Planting", "Harvest", "Irrigation", "Other"]
        )
    
    with col3:
        importance_filter = st.selectbox(
            "Filter by Importance",
            options=["All", "Important Only", "Regular Only"]
        )
    
    with col4:
        date_range = st.selectbox(
            "Date Range",
            options=["Last 7 days", "Last 30 days", "Last 3 months", "All time"]
        )
    
    # Mock log entries data
    log_entries = [
        {
            "id": "entry_001",
            "date": "2024-01-15",
            "time": "08:30",
            "field": "Field 1",
            "type": "Disease Management",
            "title": "Applied fungicide treatment",
            "description": "Applied copper-based fungicide to tomato plants showing early blight symptoms. Covered approximately 2 hectares in the north section.",
            "is_important": True,
            "tags": ["fungicide", "tomato", "disease", "treatment"],
            "images": 3,
            "weather": {"temp": 24.5, "humidity": 78, "rainfall": 0},
            "author": "John Farmer"
        },
        {
            "id": "entry_002",
            "date": "2024-01-14",
            "time": "16:45",
            "field": "Field 2",
            "type": "Observation",
            "title": "Pest activity noticed",
            "description": "Observed increased aphid activity on corn plants in the eastern section. Population seems moderate but monitoring closely.",
            "is_important": False,
            "tags": ["pest", "corn", "aphid", "monitoring"],
            "images": 2,
            "weather": {"temp": 26.8, "humidity": 65, "rainfall": 0},
            "author": "John Farmer"
        },
        {
            "id": "entry_003",
            "date": "2024-01-13",
            "time": "07:15",
            "field": "Field 3",
            "type": "Irrigation",
            "title": "Irrigation system maintenance",
            "description": "Cleaned and adjusted drip irrigation lines. Replaced 3 clogged emitters and checked water pressure throughout the system.",
            "is_important": False,
            "tags": ["irrigation", "maintenance", "drip", "water"],
            "images": 1,
            "weather": None,
            "author": "John Farmer"
        },
        {
            "id": "entry_004",
            "date": "2024-01-12",
            "time": "14:20",
            "field": "Field 4",
            "type": "Harvest",
            "title": "Rice harvest completed",
            "description": "Completed rice harvest for the season. Total yield: 28.5 tons from 5 hectares. Quality looks excellent with minimal disease damage.",
            "is_important": True,
            "tags": ["harvest", "rice", "yield", "quality"],
            "images": 5,
            "weather": {"temp": 28.2, "humidity": 72, "rainfall": 0},
            "author": "John Farmer"
        }
    ]
    
    # Apply filters
    filtered_entries = log_entries.copy()
    
    if field_filter != "All":
        filtered_entries = [e for e in filtered_entries if e["field"] == field_filter]
    
    if type_filter != "All":
        filtered_entries = [e for e in filtered_entries if e["type"] == type_filter]
    
    if importance_filter == "Important Only":
        filtered_entries = [e for e in filtered_entries if e["is_important"]]
    elif importance_filter == "Regular Only":
        filtered_entries = [e for e in filtered_entries if not e["is_important"]]
    
    # Display entries
    for entry in filtered_entries:
        show_log_entry_card(entry)
    
    if not filtered_entries:
        st.info("No log entries match the selected filters.")

def show_log_entry_card(entry):
    """Display individual log entry card"""
    
    importance_indicator = "⭐" if entry["is_important"] else ""
    image_indicator = f"📸 {entry['images']}" if entry["images"] > 0 else ""
    
    with st.container():
        # Entry header
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"### {importance_indicator} {entry['title']}")
        
        with col2:
            st.markdown(f"**{entry['date']}**")
            st.markdown(f"*{entry['time']}*")
        
        with col3:
            st.markdown(f"**{entry['field']}**")
            st.markdown(f"*{entry['type']}*")
        
        # Entry content
        st.markdown(entry['description'])
        
        # Tags
        if entry['tags']:
            tag_html = " ".join([f"<span style='background-color: #e1f5fe; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; margin-right: 4px;'>🏷️ {tag}</span>" for tag in entry['tags']])
            st.markdown(tag_html, unsafe_allow_html=True)
        
        # Weather and additional info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if entry['weather']:
                st.markdown(f"🌡️ {entry['weather']['temp']}°C | 💧 {entry['weather']['humidity']}%")
        
        with col2:
            if image_indicator:
                st.markdown(image_indicator)
        
        with col3:
            st.markdown(f"👤 {entry['author']}")
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(f"👁️ View Full", key=f"view_{entry['id']}"):
                show_entry_details(entry)
        
        with col2:
            if st.button(f"✏️ Edit", key=f"edit_{entry['id']}"):
                st.info("Edit functionality coming soon!")
        
        with col3:
            if st.button(f"📤 Share", key=f"share_{entry['id']}"):
                st.info("Share functionality coming soon!")
        
        with col4:
            if st.button(f"🗑️ Delete", key=f"delete_{entry['id']}"):
                st.warning("Delete functionality coming soon!")
        
        st.markdown("---")

def show_entry_details(entry):
    """Show detailed entry information"""
    
    with st.expander(f"📋 Entry Details - {entry['title']}", expanded=True):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Entry Information:**")
            st.write(f"**ID:** {entry['id']}")
            st.write(f"**Date:** {entry['date']} at {entry['time']}")
            st.write(f"**Field:** {entry['field']}")
            st.write(f"**Type:** {entry['type']}")
            st.write(f"**Important:** {'Yes' if entry['is_important'] else 'No'}")
            st.write(f"**Author:** {entry['author']}")
        
        with col2:
            if entry['weather']:
                st.markdown("**Weather Conditions:**")
                st.write(f"🌡️ Temperature: {entry['weather']['temp']}°C")
                st.write(f"💧 Humidity: {entry['weather']['humidity']}%")
                st.write(f"🌧️ Rainfall: {entry['weather']['rainfall']}mm")
        
        st.markdown("**Description:**")
        st.write(entry['description'])
        
        if entry['tags']:
            st.markdown("**Tags:**")
            st.write(", ".join(entry['tags']))
        
        if entry['images'] > 0:
            st.markdown(f"**Images:** {entry['images']} attached")

def show_activity_summary():
    """Show activity summary and analytics"""
    
    st.markdown("### 📊 Activity Summary")
    
    # Time period selector
    period = st.selectbox(
        "Analysis Period",
        options=["Last 7 days", "Last 30 days", "Last 3 months", "Last year"]
    )
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Total Entries", "47", delta="5")
    
    with col2:
        st.metric("⭐ Important Entries", "12", delta="2")
    
    with col3:
        st.metric("📸 Images Uploaded", "89", delta="15")
    
    with col4:
        st.metric("🏷️ Unique Tags", "28", delta="3")
    
    # Activity by type
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📂 Activity by Type")
        
        activity_types = ["Observation", "Treatment", "Irrigation", "Harvest", "Planting", "Other"]
        type_counts = [15, 12, 8, 6, 4, 2]
        
        fig = px.pie(
            values=type_counts,
            names=activity_types,
            title="Log Entries by Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📍 Activity by Field")
        
        fields = ["Field 1", "Field 2", "Field 3", "Field 4"]
        field_counts = [18, 12, 10, 7]
        
        fig = px.bar(
            x=fields,
            y=field_counts,
            title="Log Entries by Field",
            color=field_counts,
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Activity timeline
    st.markdown("#### 📅 Activity Timeline")
    
    # Mock timeline data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    daily_entries = np.random.poisson(1.5, len(dates))
    
    fig = px.line(
        x=dates,
        y=daily_entries,
        title="Daily Log Entries",
        labels={"x": "Date", "y": "Number of Entries"}
    )
    fig.update_traces(line_color="#2E8B57", marker_color="#2E8B57")
    st.plotly_chart(fig, use_container_width=True)
    
    # Tag cloud (mock)
    st.markdown("#### 🏷️ Popular Tags")
    
    popular_tags = [
        {"tag": "irrigation", "count": 15},
        {"tag": "corn", "count": 12},
        {"tag": "disease", "count": 10},
        {"tag": "harvest", "count": 8},
        {"tag": "fertilizer", "count": 7},
        {"tag": "pest", "count": 6},
        {"tag": "treatment", "count": 5},
        {"tag": "monitoring", "count": 4}
    ]
    
    tag_df = pd.DataFrame(popular_tags)
    
    fig = px.bar(
        tag_df,
        x="count",
        y="tag",
        orientation="h",
        title="Most Used Tags",
        color="count",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_search_filter():
    """Show advanced search and filter options"""
    
    st.markdown("### 🔍 Advanced Search & Filter")
    
    # Search form
    with st.form("advanced_search"):
        # Text search
        search_text = st.text_input(
            "🔍 Search Text",
            placeholder="Search in titles, descriptions, and tags...",
            help="Enter keywords to search across all log entries"
        )
        
        # Advanced filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            field_filter = st.multiselect(
                "📍 Fields",
                options=["Field 1", "Field 2", "Field 3", "Field 4"],
                help="Select specific fields"
            )
        
        with col2:
            type_filter = st.multiselect(
                "📂 Entry Types",
                options=["Observation", "Treatment", "Planting", "Harvest", "Irrigation", "Other"],
                help="Select entry types"
            )
        
        with col3:
            tag_filter = st.multiselect(
                "🏷️ Tags",
                options=["irrigation", "corn", "disease", "harvest", "fertilizer", "pest", "treatment"],
                help="Select tags"
            )
        
        # Date range
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "📅 Start Date",
                value=datetime.now() - timedelta(days=30)
            )
        
        with col2:
            end_date = st.date_input(
                "📅 End Date",
                value=datetime.now()
            )
        
        # Additional filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            importance_only = st.checkbox("⭐ Important entries only")
        
        with col2:
            with_images_only = st.checkbox("📸 Entries with images only")
        
        with col3:
            with_weather_only = st.checkbox("🌤️ Entries with weather data only")
        
        # Search button
        search_button = st.form_submit_button(
            "🔍 Search",
            use_container_width=True
        )
        
        if search_button:
            perform_search(
                search_text, field_filter, type_filter, tag_filter,
                start_date, end_date, importance_only, with_images_only, with_weather_only
            )

def perform_search(search_text, field_filter, type_filter, tag_filter,
                  start_date, end_date, importance_only, with_images_only, with_weather_only):
    """Perform search with given criteria"""
    
    with st.spinner("🔍 Searching log entries..."):
        # Mock search results
        search_results = [
            {
                "id": "entry_001",
                "date": "2024-01-15",
                "field": "Field 1",
                "type": "Disease Management",
                "title": "Applied fungicide treatment",
                "description": "Applied copper-based fungicide to tomato plants...",
                "tags": ["fungicide", "tomato", "disease"],
                "relevance": 95
            },
            {
                "id": "entry_002",
                "date": "2024-01-10",
                "field": "Field 2",
                "type": "Observation",
                "title": "Corn growth assessment",
                "description": "Measured corn plant height and assessed overall health...",
                "tags": ["corn", "growth", "assessment"],
                "relevance": 87
            }
        ]
        
        st.success(f"✅ Found {len(search_results)} matching entries")
        
        # Display search results
        for result in search_results:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{result['title']}**")
                    st.markdown(result['description'][:100] + "...")
                
                with col2:
                    st.markdown(f"**{result['date']}**")
                    st.markdown(f"*{result['field']}*")
                
                with col3:
                    st.markdown(f"**{result['relevance']}% match**")
                    st.markdown(f"*{result['type']}*")
                
                # Tags
                if result['tags']:
                    tag_html = " ".join([f"<span style='background-color: #e1f5fe; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; margin-right: 4px;'>🏷️ {tag}</span>" for tag in result['tags']])
                    st.markdown(tag_html, unsafe_allow_html=True)
                
                st.markdown("---")
        
        # Export search results
        if st.button("📤 Export Search Results"):
            st.info("Export functionality coming soon!")

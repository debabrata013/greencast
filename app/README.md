# 🌱 GreenCast Agricultural Intelligence Platform

A comprehensive AI-powered agricultural platform that combines disease detection, yield prediction, and intelligent alert systems to help farmers make data-driven decisions.

## 🎯 Project Overview

GreenCast is a full-stack agricultural intelligence platform built with modern technologies:

- **🔬 Disease Detection**: CNN-based plant disease classification using transfer learning
- **🌾 Yield Prediction**: ML models (Random Forest, XGBoost, LSTM) for crop yield forecasting  
- **🚨 Alert System**: Rule-based and ML-driven agricultural alerts with GPS-based weather forecasting
- **📝 Field Logbook**: Comprehensive field management with image uploads and activity tracking
- **📊 Analytics Dashboard**: Advanced analytics and performance insights

## 🏗️ Architecture

### Backend (FastAPI)
- **Framework**: FastAPI with async/await support
- **Database**: MongoDB with Beanie ODM
- **ML Integration**: TensorFlow, scikit-learn, XGBoost
- **Authentication**: JWT-based authentication
- **File Handling**: Image upload and processing
- **API Documentation**: Auto-generated OpenAPI/Swagger docs

### Frontend (Streamlit)
- **Framework**: Streamlit with custom components
- **Visualization**: Plotly for interactive charts
- **UI Components**: streamlit-option-menu for navigation
- **Image Processing**: PIL for image handling
- **Responsive Design**: Custom CSS for professional appearance

### ML Models
- **Disease Detection**: MobileNetV2/ResNet50 with transfer learning
- **Yield Prediction**: Ensemble of Random Forest, XGBoost, and LSTM
- **Alert System**: Rule-based + ML hybrid approach
- **Feature Engineering**: Comprehensive agricultural parameter processing

## 📁 Project Structure

```
app/
├── backend/                    # FastAPI backend
│   ├── api/v1/endpoints/      # API endpoints
│   │   ├── disease_detection.py
│   │   ├── yield_prediction.py
│   │   ├── alerts.py
│   │   └── field_logs.py
│   ├── core/                  # Core configuration
│   │   ├── config.py
│   │   └── database.py
│   ├── models/                # Database models
│   │   └── database.py
│   ├── services/              # Business logic
│   │   └── ml_service.py
│   └── main.py               # FastAPI application
├── frontend/                  # Streamlit frontend
│   ├── pages/                # Page components
│   │   ├── disease_detection.py
│   │   ├── yield_prediction.py
│   │   ├── alert_center.py
│   │   └── field_logbook.py
│   └── main_complete.py      # Main Streamlit app
├── ml_models/                # Trained ML models
├── uploads/                  # File uploads directory
├── logs/                     # Application logs
├── requirements.txt          # Python dependencies
├── start_application.py      # Startup script
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **MongoDB** installed and running
3. **Virtual environment** (recommended)

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd /Users/debabratapattnayak/web-dev/greencast/app
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start MongoDB:**
   ```bash
   # macOS with Homebrew
   brew services start mongodb-community
   
   # Linux
   sudo systemctl start mongod
   
   # Windows
   net start MongoDB
   ```

5. **Start the application:**
   ```bash
   python start_application.py
   ```

### Access the Application

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/v1/docs

### Demo Login
- **Email**: Any email address
- **Password**: Any password
- All data is simulated for demonstration purposes

## 🔧 Features

### 🔬 Disease Detection
- **Upload plant images** for instant disease identification
- **38+ disease classes** supported across multiple crops
- **Confidence scoring** with treatment recommendations
- **History tracking** with filtering and search
- **Statistics dashboard** with performance analytics

### 🌾 Yield Prediction
- **Multi-factor analysis** using weather, soil, and management data
- **Multiple ML models** with automatic best model selection
- **Confidence intervals** for prediction reliability
- **Feature importance** analysis
- **Field comparison** and trend analysis

### 🚨 Alert System
- **Real-time alerts** based on weather conditions
- **4 alert types**: Fungal risk, Pest risk, Soil temperature, Rainfall anomaly
- **7-day forecasting** with GPS-based weather data
- **Severity levels**: Critical, High, Medium, Low
- **Customizable thresholds** and notification preferences

### 📝 Field Logbook
- **Comprehensive logging** with images and notes
- **Activity tracking** across multiple entry types
- **Weather integration** for contextual information
- **Tag system** for easy categorization and search
- **Advanced filtering** and search capabilities

### 📊 Analytics Dashboard
- **Performance metrics** and KPI tracking
- **ROI analysis** with cost-benefit calculations
- **Trend analysis** across time periods
- **AI-powered recommendations** for optimization
- **Interactive visualizations** with Plotly

## 🛠️ API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User authentication
- `POST /api/v1/auth/register` - User registration

### Disease Detection
- `POST /api/v1/disease-detection/upload` - Upload image for analysis
- `GET /api/v1/disease-detection/history` - Get detection history
- `GET /api/v1/disease-detection/{id}` - Get specific detection

### Yield Prediction
- `POST /api/v1/yield-prediction/predict` - Create yield prediction
- `GET /api/v1/yield-prediction/history` - Get prediction history
- `PUT /api/v1/yield-prediction/{id}/actual-yield` - Update actual yield

### Alerts
- `POST /api/v1/alerts/generate` - Generate alerts for location
- `GET /api/v1/alerts/` - Get user alerts
- `PUT /api/v1/alerts/{id}/read` - Mark alert as read

### Field Logs
- `POST /api/v1/field-logs/` - Create log entry
- `GET /api/v1/field-logs/` - Get log entries
- `GET /api/v1/field-logs/{id}` - Get specific log entry

## 🔒 Security Features

- **JWT Authentication** with secure token handling
- **Input validation** using Pydantic models
- **File upload security** with type and size restrictions
- **CORS configuration** for cross-origin requests
- **Environment-based configuration** for sensitive data

## 📊 Database Schema

### Collections
- **users** - User accounts and profiles
- **fields** - Agricultural field information
- **disease_detections** - Disease detection results
- **yield_predictions** - Yield prediction data
- **alerts** - Alert notifications
- **field_log_entries** - Field activity logs

### Indexes
- Optimized queries with compound indexes
- Text search indexes for full-text search
- TTL indexes for automatic data cleanup

## 🧪 Testing

### Backend Testing
```bash
cd backend
pytest tests/ -v
```

### Frontend Testing
```bash
cd frontend
streamlit run main_complete.py
```

## 📈 Performance Optimization

### Backend
- **Async/await** for non-blocking operations
- **Connection pooling** for database efficiency
- **Caching strategies** for frequently accessed data
- **Background tasks** for heavy computations

### Frontend
- **Lazy loading** for large datasets
- **Caching** with st.cache_data
- **Optimized visualizations** with Plotly
- **Responsive design** for mobile compatibility

## 🔮 Future Enhancements

### Planned Features
- **Real-time notifications** with WebSocket support
- **Mobile app** with React Native
- **Satellite imagery integration** for field monitoring
- **IoT sensor integration** for real-time data
- **Multi-language support** for global users

### ML Model Improvements
- **Model versioning** and A/B testing
- **Federated learning** for privacy-preserving training
- **Edge deployment** for offline capabilities
- **Custom model training** for specific crops/regions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TensorFlow** and **scikit-learn** teams for ML frameworks
- **FastAPI** and **Streamlit** communities for excellent frameworks
- **MongoDB** for flexible document database
- **Plotly** for interactive visualizations
- **PlantVillage** dataset for disease detection training data

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the API documentation at `/api/v1/docs`
- Review the example code in the `examples/` directory

---

**🌱 GreenCast - Empowering Agriculture with AI Intelligence**

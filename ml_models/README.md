# 🤖 GreenCast ML Models - Jupyter Notebooks

This directory contains comprehensive Jupyter notebooks for agricultural machine learning models and alert systems.

## 📁 Directory Structure

```
ml_models/
├── Disease_Detection_CNN.ipynb           # Plant disease classification
├── Yield_Prediction_ML.ipynb            # Crop yield prediction
├── Alert_System_Forecasting.ipynb       # Agricultural alerts & forecasting
├── GreenCast_ML_Complete.ipynb          # Complete system overview
├── trained_models/                      # Saved model files
├── results/                             # Training results & visualizations
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd ml_models
pip install -r requirements.txt
```

### 2. Launch Jupyter
```bash
jupyter notebook
```

### 3. Run Notebooks in Order
1. **GreenCast_ML_Complete.ipynb** - Start here for system overview
2. **Disease_Detection_CNN.ipynb** - Train disease detection model
3. **Yield_Prediction_ML.ipynb** - Train yield prediction models
4. **Alert_System_Forecasting.ipynb** - Implement alert system

## 📊 Model Capabilities

### 🔬 Disease Detection
- **Technology**: CNN with Transfer Learning (MobileNetV2/ResNet50)
- **Classes**: 38+ plant disease types
- **Accuracy**: 85-95%
- **Input**: Plant images (224x224 RGB)
- **Output**: Disease class + confidence score

### 🌾 Yield Prediction
- **Technology**: Random Forest, XGBoost, LSTM
- **Performance**: R² = 0.85-0.90
- **Input**: Soil, weather, management data
- **Output**: Predicted yield (tons/hectare)
- **Features**: 13 agricultural parameters

### 🚨 Alert System
- **Technology**: Rule-based + ML classification
- **Alert Types**: 4 (Fungal, Pest, Soil Temp, Rainfall)
- **Forecasting**: 7-day weather-based alerts
- **Accuracy**: 99%+ alert detection
- **Integration**: GPS-based weather data

## 🎯 Alert Rules Implemented

### 1. Fungal Risk Alert 🍄
- **Trigger**: Temp > 28°C AND Humidity > 80% for 3+ days
- **Severity**: High
- **Action**: Apply fungicide, improve ventilation

### 2. Pest Risk Alert 🐛
- **Trigger**: Temp 20-30°C + Humidity >70% + Wind <2m/s
- **Severity**: Medium
- **Action**: Monitor crops, check pest traps

### 3. Soil Temperature Alert 🌡️
- **Trigger**: Outside optimal range for crop type
- **Severity**: Medium
- **Action**: Adjust irrigation, consider mulching

### 4. Rainfall Anomaly Alert 🌧️
- **Trigger**: ML-detected unusual patterns
- **Severity**: High
- **Action**: Check drainage, adjust irrigation

## 📈 Expected Performance

| Model | Metric | Performance | Training Time |
|-------|--------|-------------|---------------|
| Disease Detection | Accuracy | 85-95% | 2-4 hours |
| Yield Prediction | R² Score | 0.85-0.90 | 5-15 minutes |
| Alert System | Detection Rate | 99%+ | 1-2 minutes |
| Weather Forecast | Coverage | 7 days | Real-time |

## 🔧 Customization

### Adding New Disease Classes
1. Add images to processed dataset
2. Update `num_classes` in Disease Detection notebook
3. Retrain the model

### Adding New Alert Rules
1. Define rule logic in Alert System notebook
2. Set trigger conditions and thresholds
3. Add recommendations and severity levels

### Tuning Model Performance
1. Adjust hyperparameters in respective notebooks
2. Modify data augmentation strategies
3. Experiment with different architectures

## 📊 Monitoring & Evaluation

### Training Visualization
- Loss and accuracy curves
- Feature importance plots
- Confusion matrices
- Prediction vs actual plots

### Alert Dashboard
- Real-time alert status
- 7-day forecast visualization
- Historical alert patterns
- Performance metrics

## 🌐 Integration Ready

### API Deployment
Models are designed for easy API integration:
```python
# Load trained model
model = tf.keras.models.load_model('trained_models/disease_detection.h5')

# Make prediction
prediction = model.predict(image_array)
```

### Mobile App Integration
- Standardized input/output formats
- Confidence scoring for all predictions
- JSON-based alert notifications
- GPS-based location services

## 🔍 Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce batch size in training
2. **Poor Performance**: Check data quality and balance
3. **Slow Training**: Use GPU acceleration if available
4. **Alert Sensitivity**: Adjust threshold parameters

### Performance Optimization
- Use mixed precision training
- Implement data pipeline caching
- Enable GPU acceleration
- Optimize model architectures

## 📚 Documentation

Each notebook contains:
- Detailed explanations of algorithms
- Step-by-step implementation guides
- Performance evaluation metrics
- Visualization and analysis tools
- Deployment-ready code examples

## 🤝 Contributing

1. Follow existing notebook structure
2. Add comprehensive markdown documentation
3. Include performance benchmarks
4. Test all code cells before committing

## 📄 License

This project is part of the GreenCast agricultural AI platform.

---

**🎉 Ready to revolutionize agriculture with AI!**

For questions or support, refer to individual notebook documentation or create an issue in the project repository.

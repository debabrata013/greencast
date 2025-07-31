"""
Plant Disease Detection Model
Uses transfer learning with MobileNetV2 for real-time disease detection
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

class PlantDiseaseDetector:
    def __init__(self):
        """Initialize the disease detection model"""
        self.model = None
        self.class_names = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        self.build_model()
    
    def build_model(self):
        """Build the disease detection model using transfer learning"""
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        inputs = base_model.input
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(len(self.class_names), activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Disease detection model built successfully!")
    
    def preprocess_image(self, image_data):
        """Preprocess image for model prediction"""
        try:
            # Convert bytes to PIL Image
            if isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data))
            else:
                img = image_data
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to model input size
            img = img.resize((224, 224))
            
            # Convert to array and normalize
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            
            return img_array
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict_disease(self, image_data, top_k=5):
        """Predict disease from image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            if processed_image is None:
                return None
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[::-1][:top_k]
            
            results = []
            for i in top_indices:
                class_name = self.class_names[i]
                confidence = float(predictions[0][i])
                
                # Parse class name for better display
                if '___' in class_name:
                    plant, disease = class_name.split('___', 1)
                    plant = plant.replace('_', ' ').title()
                    disease = disease.replace('_', ' ').title()
                    
                    if disease.lower() == 'healthy':
                        display_name = f"{plant} - Healthy"
                        severity = "None"
                        color = "green"
                    else:
                        display_name = f"{plant} - {disease}"
                        if confidence > 0.7:
                            severity = "High"
                            color = "red"
                        elif confidence > 0.4:
                            severity = "Medium"
                            color = "orange"
                        else:
                            severity = "Low"
                            color = "yellow"
                else:
                    display_name = class_name.replace('_', ' ').title()
                    severity = "Unknown"
                    color = "gray"
                
                results.append({
                    'disease': display_name,
                    'confidence': confidence,
                    'severity': severity,
                    'color': color,
                    'raw_class': class_name
                })
            
            return results
            
        except Exception as e:
            print(f"Error predicting disease: {e}")
            return None
    
    def get_treatment_recommendations(self, disease_class):
        """Get treatment recommendations for detected disease"""
        treatments = {
            'Apple_scab': {
                'treatment': 'Apply fungicide (captan or mancozeb)',
                'prevention': 'Improve air circulation, avoid overhead watering',
                'urgency': 'Medium'
            },
            'Black_rot': {
                'treatment': 'Remove infected parts, apply copper-based fungicide',
                'prevention': 'Prune for better airflow, avoid wetting leaves',
                'urgency': 'High'
            },
            'Late_blight': {
                'treatment': 'Apply copper fungicide immediately',
                'prevention': 'Avoid overhead watering, ensure good drainage',
                'urgency': 'Critical'
            },
            'Early_blight': {
                'treatment': 'Apply fungicide, remove affected leaves',
                'prevention': 'Crop rotation, avoid overhead irrigation',
                'urgency': 'Medium'
            },
            'Powdery_mildew': {
                'treatment': 'Apply sulfur or potassium bicarbonate spray',
                'prevention': 'Improve air circulation, avoid overcrowding',
                'urgency': 'Medium'
            },
            'healthy': {
                'treatment': 'No treatment needed',
                'prevention': 'Continue regular monitoring and care',
                'urgency': 'None'
            }
        }
        
        # Extract disease key from class name
        for key in treatments.keys():
            if key.lower() in disease_class.lower():
                return treatments[key]
        
        # Default recommendation
        return {
            'treatment': 'Consult agricultural extension service',
            'prevention': 'Monitor plant health regularly',
            'urgency': 'Medium'
        }

# Global model instance
disease_detector = None

def get_disease_detector():
    """Get or create disease detector instance"""
    global disease_detector
    if disease_detector is None:
        disease_detector = PlantDiseaseDetector()
    return disease_detector

def predict_plant_disease(image_data):
    """Main function to predict plant disease from image"""
    detector = get_disease_detector()
    return detector.predict_disease(image_data)

def get_treatment_advice(disease_class):
    """Get treatment advice for detected disease"""
    detector = get_disease_detector()
    return detector.get_treatment_recommendations(disease_class)

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ============================================================================
# STEP 1: LOAD TRAINED MODEL AND PREPROCESSING OBJECTS
# ============================================================================

print("=" * 80)
print("INITIALIZING AIRBNB PRICE PREDICTION APPLICATION")
print("=" * 80)

# Define paths for model and preprocessing objects
MODEL_PATH = 'models/random_forest_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
PCA_PATH = 'models/pca_model.pkl'
ENCODER_PATHS = {
    'neighbourhood': 'models/encoder_neighbourhood.pkl',
    'room_type': 'models/encoder_room_type.pkl',
    'interaction': 'models/encoder_interaction.pkl'
}

# Load the trained Random Forest model
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Loaded Random Forest Model")
except FileNotFoundError:
    print(f"❌ Model not found at {MODEL_PATH}")
    model = None

# Load the StandardScaler
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Loaded StandardScaler")
except FileNotFoundError:
    print(f"❌ Scaler not found at {SCALER_PATH}")
    scaler = None

# Load the PCA model
try:
    with open(PCA_PATH, 'rb') as f:
        pca = pickle.load(f)
    print("✅ Loaded PCA Model")
except FileNotFoundError:
    print(f"❌ PCA model not found at {PCA_PATH}")
    pca = None

# Load LabelEncoders
encoders = {}
for encoder_name, encoder_path in ENCODER_PATHS.items():
    try:
        with open(encoder_path, 'rb') as f:
            encoders[encoder_name] = pickle.load(f)
        print(f"✅ Loaded {encoder_name} LabelEncoder")
    except FileNotFoundError:
        print(f"❌ {encoder_name} encoder not found at {encoder_path}")
        encoders[encoder_name] = None

# ============================================================================
# STEP 2: DEFINE REFERENCE DATA
# ============================================================================

NEIGHBOURHOODS = [
    'Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'
]

ROOM_TYPES = [
    'Entire Home/Apt', 'Private Room', 'Shared Room', 'Hotel Room'
]

MONTHS = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

print("=" * 80)

# ============================================================================
# STEP 3: HELPER FUNCTIONS
# ============================================================================

def validate_input(data):
    """
    Validate user input data
    """
    errors = []
    
    # Neighbourhood validation
    if 'neighbourhood' not in data or data['neighbourhood'] not in NEIGHBOURHOODS:
        errors.append("Invalid neighbourhood selected")
    
    # Room Type validation
    if 'roomType' not in data or data['roomType'] not in ROOM_TYPES:
        errors.append("Invalid room type selected")
    
    # Zipcode validation
    if 'zipcode' not in data or not str(data['zipcode']).strip():
        errors.append("Zipcode is required")
    
    # Beds validation
    try:
        beds = float(data.get('beds', 0))
        if beds < 0 or beds > 20:
            errors.append("Beds must be between 0 and 20")
    except (ValueError, TypeError):
        errors.append("Beds must be a valid number")
    
    # Bedrooms validation
    try:
        bedrooms = float(data.get('bedrooms', 0))
        if bedrooms < 0 or bedrooms > 15:
            errors.append("Bedrooms must be between 0 and 15")
    except (ValueError, TypeError):
        errors.append("Bedrooms must be a valid number")
    
    # Accommodates validation
    try:
        accommodates = float(data.get('accommodates', 1))
        if accommodates < 1 or accommodates > 30:
            errors.append("Accommodates must be between 1 and 30")
    except (ValueError, TypeError):
        errors.append("Accommodates must be a valid number")
    
    # Minimum Nights validation
    try:
        minimum_nights = float(data.get('minimumNights', 1))
        if minimum_nights < 1 or minimum_nights > 365:
            errors.append("Minimum nights must be between 1 and 365")
    except (ValueError, TypeError):
        errors.append("Minimum nights must be a valid number")
    
    # Number of Reviews validation
    try:
        reviews = float(data.get('numberOfReviews', 0))
        if reviews < 0 or reviews > 500:
            errors.append("Number of reviews must be between 0 and 500")
    except (ValueError, TypeError):
        errors.append("Number of reviews must be a valid number")
    
    # Reviews per Month validation
    try:
        reviews_per_month = float(data.get('reviewsPerMonth', 0))
        if reviews_per_month < 0 or reviews_per_month > 30:
            errors.append("Reviews per month must be between 0 and 30")
    except (ValueError, TypeError):
        errors.append("Reviews per month must be a valid number")
    
    # Review Scores Rating validation
    try:
        rating = float(data.get('reviewScoresRating', 0))
        if rating < 1 or rating > 5:
            errors.append("Review scores rating must be between 1 and 5")
    except (ValueError, TypeError):
        errors.append("Review scores rating must be a valid number")
    
    # Host Year validation
    try:
        host_year = int(data.get('hostYear', 2020))
        if host_year < 2008 or host_year > 2026:
            errors.append("Host year must be between 2008 and 2026")
    except (ValueError, TypeError):
        errors.append("Host year must be a valid number")
    
    # Host Month validation
    try:
        host_month = int(data.get('hostMonth', 1))
        if host_month < 1 or host_month > 12:
            errors.append("Host month must be between 1 and 12")
    except (ValueError, TypeError):
        errors.append("Host month must be a valid number")
    
    # Host Response Rate validation (optional)
    if data.get('hostResponseRate'):
        try:
            response_rate = float(data.get('hostResponseRate', 0))
            if response_rate < 0 or response_rate > 100:
                errors.append("Host response rate must be between 0 and 100")
        except (ValueError, TypeError):
            errors.append("Host response rate must be a valid number")
    
    return errors

def preprocess_input(data):
    """
    Preprocess user input for model prediction
    """
    try:
        # Extract numerical features
        numerical_features = {
            'Beds': float(data.get('beds', 0)),
            'Number Of Reviews': float(data.get('numberOfReviews', 0)),
            'Review Scores Rating': float(data.get('reviewScoresRating', 4.5)),
            'Host_Year': int(data.get('hostYear', 2020)),
            'Host_Month': int(data.get('hostMonth', 1))
        }
        
        # Extract categorical features
        neighbourhood = data.get('neighbourhood', 'Manhattan')
        room_type = data.get('roomType', 'Entire Home/Apt')
        
        # Create interaction feature
        interaction = neighbourhood + '_' + room_type
        
        # Encode categorical features
        if encoders['neighbourhood'] is not None:
            try:
                neighbourhood_encoded = encoders['neighbourhood'].transform([neighbourhood])[0]
            except ValueError:
                print(f"⚠️ Unknown neighbourhood: {neighbourhood}, using default")
                neighbourhood_encoded = 0
        else:
            neighbourhood_encoded = 0
        
        if encoders['room_type'] is not None:
            try:
                room_type_encoded = encoders['room_type'].transform([room_type])[0]
            except ValueError:
                print(f"⚠️ Unknown room type: {room_type}, using default")
                room_type_encoded = 0
        else:
            room_type_encoded = 0
        
        # Create numerical interaction
        neighbourhood_roomtype_interaction = neighbourhood_encoded * room_type_encoded
        
        # Try to encode categorical interaction
        try:
            if encoders['interaction'] is not None:
                interaction_encoded = encoders['interaction'].transform([interaction])[0]
            else:
                interaction_encoded = neighbourhood_roomtype_interaction
        except ValueError:
            print(f"⚠️ Unknown interaction: {interaction}, using multiplicative value")
            interaction_encoded = neighbourhood_roomtype_interaction
        
        # Prepare input features for model
        input_features = {
            'Beds': numerical_features['Beds'],
            'Number Of Reviews': numerical_features['Number Of Reviews'],
            'Review Scores Rating': numerical_features['Review Scores Rating'],
            'Host_Year': numerical_features['Host_Year'],
            'Host_Month': numerical_features['Host_Month'],
            'Neighbourhood_Encoded': neighbourhood_encoded,
            'RoomType_Encoded': room_type_encoded,
            'Neighbourhood_RoomType_Interaction': neighbourhood_roomtype_interaction,
            'Neighbourhood_RoomType_Encoded': interaction_encoded
        }
        
        return input_features, neighbourhood, room_type, interaction
    
    except Exception as e:
        print(f"❌ Error in preprocessing: {str(e)}")
        raise

def make_prediction(input_features):
    """
    Make price prediction using the trained model
    """
    try:
        if model is None:
            raise Exception("Model not loaded")
        
        # Create DataFrame with input features
        feature_df = pd.DataFrame([input_features])
        
        # Ensure features are in correct order
        required_features = [
            'Beds', 'Number Of Reviews', 'Review Scores Rating', 
            'Host_Year', 'Host_Month', 'Neighbourhood_Encoded', 
            'RoomType_Encoded', 'Neighbourhood_RoomType_Interaction', 
            'Neighbourhood_RoomType_Encoded'
        ]
        
        feature_df = feature_df[required_features]
        
        # Make prediction
        predicted_price = model.predict(feature_df)[0]
        
        # Ensure prediction is positive
        predicted_price = max(predicted_price, 0)
        
        return float(predicted_price)
    
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        raise

def calculate_price_range(predicted_price, confidence_interval=0.15):
    """
    Calculate price range based on predicted price
    confidence_interval: percentage (0.15 = 15%)
    """
    margin = predicted_price * confidence_interval
    return {
        'min': float(predicted_price - margin),
        'predicted': float(predicted_price),
        'max': float(predicted_price + margin)
    }

# ============================================================================
# STEP 4: FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    """
    Serve the home page with the input form
    """
    return render_template('index.html')

@app.route('/api/neighbourhoods', methods=['GET'])
def get_neighbourhoods():
    """
    API endpoint to get list of neighbourhoods
    """
    return jsonify({
        'neighbourhoods': NEIGHBOURHOODS,
        'count': len(NEIGHBOURHOODS)
    })

@app.route('/api/room-types', methods=['GET'])
def get_room_types():
    """
    API endpoint to get list of room types
    """
    return jsonify({
        'roomTypes': ROOM_TYPES,
        'count': len(ROOM_TYPES)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for price prediction
    Accepts JSON data with property details
    Returns predicted price and price range
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        print("\n" + "=" * 80)
        print("NEW PREDICTION REQUEST RECEIVED")
        print("=" * 80)
        print(f"\nInput Data:")
        for key, value in data.items():
            print(f"  {key}: {value}")
        
        # Validate input
        errors = validate_input(data)
        if errors:
            print(f"\n❌ Validation Errors:")
            for error in errors:
                print(f"  - {error}")
            return jsonify({
                'success': False,
                'error': 'Validation failed',
                'details': errors
            }), 400
        
        # Preprocess input
        input_features, neighbourhood, room_type, interaction = preprocess_input(data)
        
        print(f"\n✅ Input validated and preprocessed")
        print(f"\nProcessed Features:")
        for key, value in input_features.items():
            print(f"  {key}: {value:.4f}")
        
        # Make prediction
        predicted_price = make_prediction(input_features)
        
        # Calculate price range
        price_range = calculate_price_range(predicted_price)
        
        print(f"\n✅ Prediction Successful!")
        print(f"  Predicted Price: ${predicted_price:.2f}")
        print(f"  Price Range: ${price_range['min']:.2f} - ${price_range['max']:.2f}")
        print("=" * 80 + "\n")
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'price': predicted_price,
                'currency': 'USD',
                'priceRange': {
                    'min': price_range['min'],
                    'predicted': price_range['predicted'],
                    'max': price_range['max']
                }
            },
            'property': {
                'neighbourhood': neighbourhood,
                'roomType': room_type,
                'interaction': interaction,
                'beds': float(data.get('beds', 0)),
                'bedrooms': float(data.get('bedrooms', 0)),
                'accommodates': float(data.get('accommodates', 0)),
                'reviews': float(data.get('numberOfReviews', 0)),
                'rating': float(data.get('reviewScoresRating', 0))
            },
            'message': 'Prediction successful',
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("=" * 80 + "\n")
        return jsonify({
            'success': False,
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'pca_loaded': pca is not None,
        'encoders_loaded': all(v is not None for v in encoders.values())
    }), 200

@app.route('/api/info', methods=['GET'])
def info():
    """
    Get application information
    """
    return jsonify({
        'app': 'Airbnb Price Predictor',
        'version': '1.0.0',
        'model': 'Random Forest Regressor',
        'features': {
            'numerical': ['Beds', 'Number Of Reviews', 'Review Scores Rating', 'Host_Year', 'Host_Month'],
            'categorical': ['Neighbourhood', 'Room Type'],
            'interaction': ['Neighbourhood_RoomType']
        },
        'neighbourhoods': NEIGHBOURHOODS,
        'roomTypes': ROOM_TYPES
    }), 200

# ============================================================================
# STEP 5: ERROR HANDLERS
# ============================================================================

@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    return jsonify({
        'success': False,
        'error': 'Bad Request',
        'details': str(error)
    }), 400

@app.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    return jsonify({
        'success': False,
        'error': 'Not Found',
        'details': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def server_error(error):
    """Handle server errors"""
    return jsonify({
        'success': False,
        'error': 'Internal Server Error',
        'details': str(error)
    }), 500

# ============================================================================
# STEP 6: RUN APPLICATION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 STARTING AIRBNB PRICE PREDICTION APPLICATION")
    print("=" * 80)
    print("\n📊 Application Configuration:")
    print(f"  - Flask Debug Mode: True")
    print(f"  - Port: 5000")
    print(f"  - Host: localhost")
    print(f"\n📡 Available Endpoints:")
    print(f"  - GET  /                  → Home page with input form")
    print(f"  - POST /api/predict        → Predict price")
    print(f"  - GET  /api/neighbourhoods → Get neighbourhoods list")
    print(f"  - GET  /api/room-types     → Get room types list")
    print(f"  - GET  /api/health         → Health check")
    print(f"  - GET  /api/info           → Application info")
    print(f"\n🌐 Access the app at: http://localhost:5000")
    print("=" * 80 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='localhost', port=5000)
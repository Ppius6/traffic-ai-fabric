# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "e4450e3f-6c73-431d-9946-fe0f891228f6",
# META       "default_lakehouse_name": "LH",
# META       "default_lakehouse_workspace_id": "d5b6a633-7a45-4c85-82b6-89bc1d124fcf",
# META       "known_lakehouses": [
# META         {
# META           "id": "e4450e3f-6c73-431d-9946-fe0f891228f6"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

%pip install azure-eventhub==5.11.4

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd
import numpy as np
import requests
import time
import json
import uuid
import joblib
import os
import mlflow
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
from azure.eventhub import EventHubProducerClient, EventData

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# CONFIGURATION
CONFIG = {
    "event_interval_seconds": 8,
    "monitored_locations": 12,
    "run_duration_minutes": 999,  # Run indefinitely
    "weather_refresh_seconds": 300,
    "enable_ai_predictions": True,
    "verbose_logging": True
}


# EVENTSTREAM CONNECTION 

EVENTSTREAM_CONFIG = {
    "connection_string": """
    Endpoint= [removed]
    """.replace("\n", "").replace(" ", ""),
    "target_table": "EnhancedTrafficEvents" 
}

# Initialize producer
try:
    producer = EventHubProducerClient.from_connection_string(
        conn_str=EVENTSTREAM_CONFIG["connection_string"]
    )
    print("✅ Connected to NairobiTrafficStream -> EnhancedTrafficEvents table")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    producer = None

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# LOAD TRAINED ML MODEL

print("=" * 80)
print("🤖 Loading Trained ML Model from Fabric MLflow")
print("=" * 80 + "\n")

# Initialize variables with default values
best_model = None
label_encoders = {}
feature_names = []

try:
    # Try loading from MLflow 
    import mlflow
    model_uri = "models:/nairobi_accident_risk_model/Production"
    best_model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Successfully loaded model from MLflow!")
    
except Exception as e:
    print(f"⚠️  MLflow load failed: {e}")
    try:
        # Try loading from Files relative path
        best_model = joblib.load("Files/ml_models/nairobi_accident_risk_model_final.pkl")
        label_encoders = joblib.load("Files/ml_models/label_encoders.pkl")
        with open("Files/ml_models/feature_names.json", 'r') as f:
            feature_names = json.load(f)
        print("✅ Successfully loaded model from Files directory!")
        
    except Exception as e2:
        print(f"⚠️  Files load failed: {e2}")
        print("   Trying absolute Fabric paths...")
        try:
            # Try Fabric absolute paths
            best_model = joblib.load("/lakehouse/default/Files/ml_models/nairobi_accident_risk_model_final.pkl")
            label_encoders = joblib.load("/lakehouse/default/Files/ml_models/label_encoders.pkl")
            with open("/lakehouse/default/Files/ml_models/feature_names.json", 'r') as f:
                feature_names = json.load(f)
            print("✅ Successfully loaded model from absolute lakehouse path!")
            
        except Exception as e3:
            print(f"❌ All model loading attempts failed: {e3}")
            print("   Continuing with rule-based predictions")

# Display model status
if best_model is not None:
    print(f"   Model type: {type(best_model).__name__}")
    print(f"   Features required: {len(feature_names)}")
    print(f"   Label encoders: {list(label_encoders.keys())}")
else:
    print("   Using rule-based predictions (no ML model loaded)")

print()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# LOAD ROAD NETWORK FROM LAKEHOUSE

print("=" * 80)
print("📍 Loading Road Network from Lakehouse")
print("=" * 80 + "\n")

try:
    # Load from your actual nairobi_accidents_historical table
    df_accidents = spark.read.table("LH.nairobi_accidents_historical").toPandas()
    print(f"✅ Loaded accident data: {len(df_accidents)} records")
    
    # Check what columns are actually available
    available_columns = df_accidents.columns.tolist()
    print(f"📋 Available columns: {available_columns}")
    
    # Create road network from accident data
    if 'location' in available_columns and 'road_name' in available_columns:
        # First convert severity to numeric scores
        severity_map = {'Fatal': 1.0, 'Serious': 0.7, 'Minor': 0.3}
        df_accidents['severity_score'] = df_accidents['severity'].map(severity_map).fillna(0.5)
        
        # Group by location and calculate risk metrics
        road_locations = df_accidents.groupby(['location', 'road_name']).agg({
            'latitude': 'first',
            'longitude': 'first',
            'severity_score': 'mean',  # Now we can calculate mean of numeric scores
            'risk_score': 'mean'  # Also use the existing risk_score column
        }).reset_index()
        
        # Calculate base risk from accident frequency and severity
        location_counts = df_accidents['location'].value_counts()
        road_locations['accident_count'] = road_locations['location'].map(location_counts)
        
        # Use both severity and existing risk_score for better risk calculation
        road_locations['base_risk'] = (
            road_locations['accident_count'] / road_locations['accident_count'].max() * 0.3 +
            road_locations['severity_score'] * 0.4 +
            road_locations['risk_score'] * 0.3
        ).clip(0.3, 0.95)
        
        # Rename columns to match expected format
        NAIROBI_LOCATIONS = []
        for _, row in road_locations.iterrows():
            location_dict = {
                'location_name': row['location'],
                'road_name': row['road_name'], 
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'base_risk': float(row['base_risk'])
            }
            NAIROBI_LOCATIONS.append(location_dict)
        
        print(f"✅ Extracted {len(NAIROBI_LOCATIONS)} unique road locations from historical data")
        
        # Show the highest risk locations found
        print(f"\n📍 Top 5 Highest Risk Locations from Historical Data:")
        sorted_locations = sorted(NAIROBI_LOCATIONS, key=lambda x: x['base_risk'], reverse=True)
        for i, location in enumerate(sorted_locations[:5], 1):
            print(f"   {i:2d}. {location['location_name'][:35]:35s} - Risk: {location['base_risk']:.3f}")
        
    else:
        print("⚠️  Expected columns not found, using fallback locations")
        # Fallback to hardcoded locations
        NAIROBI_LOCATIONS = [
            {
                "location_name": "Mombasa Road Junction",
                "road_name": "Mombasa Road",
                "latitude": -1.3192,
                "longitude": 36.8517,
                "base_risk": 0.85
            },
            {
                "location_name": "Thika Road Superhighway", 
                "road_name": "Thika Road",
                "latitude": -1.2326,
                "longitude": 36.8847,
                "base_risk": 0.78
            },
        ]

except Exception as e:
    print(f"❌ Failed to load road network: {e}")
    print("   Using fallback location data...")
    
    # Fallback locations
    NAIROBI_LOCATIONS = [
        {
            "location_name": "Mombasa Road Junction",
            "road_name": "Mombasa Road", 
            "latitude": -1.3192,
            "longitude": 36.8517,
            "base_risk": 0.85
        },
        {
            "location_name": "Thika Road Superhighway",
            "road_name": "Thika Road",
            "latitude": -1.2326,
            "longitude": 36.8847,
            "base_risk": 0.78
        },
        {
            "location_name": "Waiyaki Way Westlands",
            "road_name": "Waiyaki Way",
            "latitude": -1.2635,
            "longitude": 36.8078,
            "base_risk": 0.72
        }
    ]

    
# Select high-risk locations for monitoring
monitored_count = min(CONFIG['monitored_locations'], len(NAIROBI_LOCATIONS))
if len(NAIROBI_LOCATIONS) > monitored_count:
    # Sort by base_risk and take top locations
    sorted_locations = sorted(NAIROBI_LOCATIONS, key=lambda x: x['base_risk'], reverse=True)
    MONITORED_LOCATIONS = sorted_locations[:monitored_count]
else:
    MONITORED_LOCATIONS = NAIROBI_LOCATIONS

print(f"✅ Selected {len(MONITORED_LOCATIONS)} high-risk locations for monitoring")

# Display top monitored locations
print("\n📍 Top Monitored Locations:")
for i, location in enumerate(MONITORED_LOCATIONS[:10], 1):
    print(f"   {i:2d}. {location['location_name'][:30]:30s} ({location['road_name'][:20]:20s}) - Risk: {location['base_risk']:.2f}")

print()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# WEATHER API FUNCTION

weather_cache = None
weather_cache_time = None

def get_current_weather():
    """Get real weather for Nairobi with caching"""
    global weather_cache, weather_cache_time
    
    # Check cache
    if weather_cache and weather_cache_time:
        cache_age = (datetime.now() - weather_cache_time).total_seconds()
        if cache_age < CONFIG['weather_refresh_seconds']:
            return weather_cache
    
    try:
        # Open-Meteo API for Nairobi
        api_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": -1.2921,
            "longitude": 36.8219,
            "current": "temperature_2m,precipitation,cloud_cover,wind_speed_10m",
            "timezone": "Africa/Nairobi"
        }
        
        response = requests.get(api_url, params=params, timeout=10)
        data = response.json()
        
        current = data['current']
        temp = current['temperature_2m']
        precip = current['precipitation']
        
        # Determine conditions
        if precip > 5:
            weather = "Heavy Rain"
        elif precip > 0.5:
            weather = "Rain"
        elif precip > 0:
            weather = "Light Rain"
        else:
            weather = "Clear"
        
        weather_data = {
            "weather": weather,
            "temperature": temp,
            "precipitation": precip
        }
        
        # Update cache
        weather_cache = weather_data
        weather_cache_time = datetime.now()
        
        return weather_data
        
    except Exception as e:
        print(f"⚠️ Weather API error: {e}")
        # Fallback simulation
        return {
            "weather": np.random.choice(["Clear", "Light Rain", "Rain"], p=[0.7, 0.2, 0.1]),
            "temperature": np.random.uniform(18, 28),
            "precipitation": np.random.uniform(0, 2)
        }

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ML PREDICTION ENGINE using our trained model

def prepare_features_for_ml_model(event_data):
    """
    Prepare features for the trained ML model
    Must match the 20 features your model expects
    """
    features = {}
    
    # Time features
    timestamp = pd.to_datetime(event_data['timestamp'])
    features['hour'] = timestamp.hour
    features['month'] = timestamp.month
    
    # Cyclical time features
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    
    # Day of week encoding
    day_of_week = timestamp.strftime('%A')
    if 'day_of_week' in label_encoders:
        try:
            features['day_of_week_encoded'] = label_encoders['day_of_week'].transform([day_of_week])[0]
        except:
            features['day_of_week_encoded'] = 0
    else:
        features['day_of_week_encoded'] = 0
    
    features['day_sin'] = np.sin(2 * np.pi * features['day_of_week_encoded'] / 7)
    features['day_cos'] = np.cos(2 * np.pi * features['day_of_week_encoded'] / 7)
    
    # Time context
    is_weekend = day_of_week in ['Saturday', 'Sunday']
    is_rush_hour = (7 <= features['hour'] <= 9) or (17 <= features['hour'] <= 19)
    
    features['is_weekend'] = 1 if is_weekend else 0
    features['is_rush_hour'] = 1 if is_rush_hour else 0
    features['is_holiday'] = 0
    
    # Location features
    features['latitude'] = float(event_data['latitude'])
    features['longitude'] = float(event_data['longitude'])
    features['base_location_risk'] = float(event_data['base_risk'])
    
    # Road features (with defaults)
    features['speed_limit'] = 60  # Default speed limit
    features['lanes'] = 4  # Default lanes
    
    # Road type encoding
    road_type = 'main_road'  # Default road type
    if 'road_type' in label_encoders:
        try:
            features['road_type_encoded'] = label_encoders['road_type'].transform([road_type])[0]
        except:
            features['road_type_encoded'] = 0
    else:
        features['road_type_encoded'] = 0
    
    # Weather encoding
    weather = event_data['weather']
    if 'weather' in label_encoders:
        try:
            features['weather_encoded'] = label_encoders['weather'].transform([weather])[0]
        except:
            features['weather_encoded'] = 0
    else:
        features['weather_encoded'] = 0
    
    # Weather features
    features['temperature'] = float(event_data['temperature'])
    features['precipitation'] = float(event_data['precipitation'])
    features['cloudcover'] = 50  # Default cloud cover
    features['windspeed'] = 10  # Default wind speed
    
    # Road surface encoding
    road_surface = "Wet" if "Rain" in weather else "Dry"
    if 'road_surface' in label_encoders:
        try:
            features['road_surface_encoded'] = label_encoders['road_surface'].transform([road_surface])[0]
        except:
            features['road_surface_encoded'] = 0
    else:
        features['road_surface_encoded'] = 0
    
    # Traffic features
    features['traffic_density'] = int(event_data['traffic_density'])
    features['average_speed'] = 40  # Will be calculated later
    
    # Risk factors
    features['time_risk_factor'] = 1.5 if is_rush_hour else 1.0
    features['weather_risk_factor'] = 1.8 if 'Rain' in weather else 1.0
    features['day_risk_factor'] = 0.7 if is_weekend else 1.0
    
    # Create feature array in the order expected by the model
    if best_model and len(feature_names) > 0:
        feature_array = []
        for fname in feature_names:
            feature_array.append(features.get(fname, 0))
        return np.array([feature_array])
    else:
        return None

def generate_ai_prediction(traffic_density, weather, hour, location_risk):
    """
    Generate comprehensive AI predictions using trained model or fallback
    """
    global best_model

    # Prepare data for ML model
    event_data = {
        'timestamp': datetime.now(),
        'latitude': -1.2921,  # Default Nairobi coordinates
        'longitude': 36.8219,
        'base_risk': location_risk,
        'weather': weather,
        'temperature': 25.0,  # Default temperature
        'precipitation': 1.0 if "Rain" in weather else 0.0,
        'traffic_density': traffic_density
    }
    
    # Try ML prediction first
    if best_model is not None:
        try:
            # Prepare features for ML model
            X = prepare_features_for_ml_model(event_data)
            
            if X is not None:
                # Get ML prediction
                risk_level = best_model.predict(X)[0]
                risk_probs = best_model.predict_proba(X)[0]
                
                # Get class probabilities
                class_probs = {cls: float(prob) for cls, prob in zip(best_model.classes_, risk_probs)}
                
                # Calculate overall risk score
                risk_score_map = {'Low': 0.3, 'Medium': 0.6, 'High': 0.9}
                risk_score = sum(risk_score_map.get(cls, 0.5) * prob for cls, prob in class_probs.items())
                
                # Confidence is the probability of the predicted class
                confidence = class_probs[risk_level]
                
                # Use ML results
                prob_high = class_probs.get('High', 0.1)
                prob_medium = class_probs.get('Medium', 0.3)
                prob_low = class_probs.get('Low', 0.6)
                
                prediction_method = "ML"
                
            else:
                raise Exception("Feature preparation failed")
                
        except Exception as e:
            print(f"⚠️ ML prediction failed: {e}, using fallback")
            best_model = None  # Disable ML for subsequent calls
    
    # Fallback rule-based prediction
    if best_model is None:
        # Calculate base risk factors
        weather_risk = 1.5 if "Rain" in weather else 1.0
        time_risk = 1.3 if (7 <= hour <= 9) or (17 <= hour <= 19) else 1.0
        density_risk = traffic_density / 100.0
        location_risk_factor = location_risk
        
        # Combined risk score (0-1)
        risk_score = min(1.0, (density_risk * weather_risk * time_risk * location_risk_factor))
        
        # Determine risk level
        if risk_score >= 0.75:
            risk_level = "High"
            confidence = np.random.uniform(0.85, 0.95)
        elif risk_score >= 0.45:
            risk_level = "Medium" 
            confidence = np.random.uniform(0.70, 0.85)
        else:
            risk_level = "Low"
            confidence = np.random.uniform(0.60, 0.80)
        
        # Calculate probability distribution
        if risk_level == "High":
            prob_high = np.random.uniform(0.7, 0.9)
            prob_medium = np.random.uniform(0.1, 0.3)
            prob_low = 1.0 - prob_high - prob_medium
        elif risk_level == "Medium":
            prob_medium = np.random.uniform(0.5, 0.7)
            prob_high = np.random.uniform(0.1, 0.3)
            prob_low = 1.0 - prob_high - prob_medium
        else:
            prob_low = np.random.uniform(0.6, 0.8)
            prob_high = np.random.uniform(0.05, 0.15)
            prob_medium = 1.0 - prob_high - prob_low
        
        prediction_method = "RuleBased"
    
    # Generate primary factors
    factors = []
    if traffic_density >= 80:
        factors.append("High traffic density")
    if "Rain" in weather:
        factors.append("Adverse weather conditions")
    if (7 <= hour <= 9) or (17 <= hour <= 19):
        factors.append("Rush hour traffic")
    if location_risk >= 0.7:
        factors.append("High-risk location")
    
    primary_factors = ", ".join(factors) if factors else "Normal traffic conditions"
    
    # Generate AI explanation
    if prediction_method == "ML":
        explanations = [
            f"ML model prediction: {risk_level} risk with {confidence:.0%} confidence",
            f"Traffic density at {traffic_density}% with {weather.lower()} conditions",
            f"Decision Tree model trained on historical Nairobi accident data"
        ]
    else:
        explanations = [
            f"Traffic density at {traffic_density}% with {weather.lower()} conditions",
            f"Risk assessment based on traffic patterns and weather impact",
            f"Location risk factor: {location_risk:.2f}, Time factor: {time_risk:.1f}"
        ]
    
    ai_explanation = ". ".join(explanations[:2])
    
    # Determine if alert should be triggered
    alert_triggered = (risk_level == "High" and confidence > 0.8) or (traffic_density >= 90)
    
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_probability_high": prob_high,
        "risk_probability_medium": prob_medium, 
        "risk_probability_low": prob_low,
        "confidence": confidence,
        "primary_factors": primary_factors,
        "ai_explanation": ai_explanation,
        "alert_triggered": alert_triggered,
        "prediction_method": prediction_method
    }

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# TRAFFIC EVENT GENERATOR

def generate_enhanced_traffic_event():
    """
    Generate traffic event with complete AI predictions
    Matches your Eventhouse table schema exactly
    """
    
    # Select random location from monitored locations
    location = np.random.choice(MONITORED_LOCATIONS)
    
    # Get current time info
    now = datetime.now()
    hour = now.hour
    
    # Get weather
    weather_data = get_current_weather()
    
    # Generate traffic density based on time and weather
    base_density = np.random.randint(20, 70)
    
    # Time adjustments
    if (7 <= hour <= 9) or (17 <= hour <= 19):  # Rush hours
        base_density = min(100, base_density + np.random.randint(20, 40))
    
    # Weather adjustments
    if "Rain" in weather_data["weather"]:
        base_density = min(100, base_density + np.random.randint(10, 25))
    
    traffic_density = int(np.clip(base_density, 0, 100))
    
    # Generate AI predictions
    ai_prediction = generate_ai_prediction(
        traffic_density, 
        weather_data["weather"], 
        hour, 
        location["base_risk"]
    )
    
    # Build complete event matching your schema
    event = {
        # Core identifiers
        "PredictionId": str(uuid.uuid4()),
        "Timestamp": now.isoformat(),
        "EventId": str(uuid.uuid4()),
        
        # Location data
        "LocationName": location["location_name"],
        "RoadName": location["road_name"],
        "Latitude": location["latitude"],
        "Longitude": location["longitude"],
        
        # AI Predictions (matching your schema)
        "RiskLevel": ai_prediction["risk_level"],
        "RiskScore": round(ai_prediction["risk_score"], 4),
        "RiskProbabilityHigh": round(ai_prediction["risk_probability_high"], 4),
        "RiskProbabilityMedium": round(ai_prediction["risk_probability_medium"], 4),
        "RiskProbabilityLow": round(ai_prediction["risk_probability_low"], 4),
        "Confidence": round(ai_prediction["confidence"], 4),
        "PrimaryFactors": ai_prediction["primary_factors"],
        "AIExplanation": ai_prediction["ai_explanation"],
        
        # Traffic metrics
        "TrafficDensity": traffic_density,
        "Weather": weather_data["weather"],
        "Hour": hour,
        
        # Alert system
        "AlertTriggered": ai_prediction["alert_triggered"],
        "AlertSentTime": now.isoformat() if ai_prediction["alert_triggered"] else None
    }
    
    return event

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# SEND TO EVENTSTREAM

def send_event_to_stream(event):
    """Send event to your Eventstream"""
    if producer is None:
        return False
    
    try:
        event_json = json.dumps(event, default=str)
        event_data = EventData(event_json)
        
        batch = producer.create_batch()
        batch.add(event_data)
        producer.send_batch(batch)
        
        return True
        
    except Exception as e:
        print(f"⚠️ Send error: {str(e)[:50]}")
        return False

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# MAIN SIMULATION LOOP

print("🚀 Starting Traffic Generator")
print("=" * 60 + "\n")

stats = {
    "events_generated": 0,
    "events_sent": 0,
    "high_risk_events": 0,
    "alerts_triggered": 0
}

start_time = datetime.now()

try:
    while True:
        # Generate enhanced event
        event = generate_enhanced_traffic_event()
        stats["events_generated"] += 1
        
        # Track statistics
        if event["RiskLevel"] == "High":
            stats["high_risk_events"] += 1
        if event["AlertTriggered"]:
            stats["alerts_triggered"] += 1
        
        # Send to Eventstream
        if send_event_to_stream(event):
            stats["events_sent"] += 1
        
        # Display progress
        if stats["events_generated"] % 5 == 0:
            elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
            
            print(f"📊 Status: {stats['events_sent']}/{stats['events_generated']} sent | "
                  f"🔴 {stats['high_risk_events']} high-risk | "
                  f"🚨 {stats['alerts_triggered']} alerts | "
                  f"⏱️ {elapsed_minutes:.1f}m runtime")
            
            # Show latest event details
            risk_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            print(f"   Latest: {event['LocationName']} | "
                  f"{risk_emoji[event['RiskLevel']]} {event['RiskLevel']} | "
                  f"Density: {event['TrafficDensity']}% | "
                  f"Weather: {event['Weather']}")
            
            if event["AlertTriggered"]:
                print(f"   🚨 ALERT: {event['PrimaryFactors']}")
            
            print()
        
        # Wait before next event
        time.sleep(CONFIG["event_interval_seconds"])

except KeyboardInterrupt:
    print("\n Generator stopped by user")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Final summary
elapsed = datetime.now() - start_time
print(f"\n✅ Final Stats:")
print(f"   Events: {stats['events_sent']}/{stats['events_generated']}")
print(f"   High Risk: {stats['high_risk_events']}")
print(f"   Alerts: {stats['alerts_triggered']}")
print(f"   Runtime: {elapsed.total_seconds()/60:.1f} minutes")
print("=" * 60)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

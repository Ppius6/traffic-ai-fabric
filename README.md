# traffic-ai-fabric

## Real-Time Traffic Intelligence System for Nairobi

A comprehensive AI-powered traffic safety system built on Microsoft Fabric that predicts accident risk in real-time using machine learning and live data streams.

## Overview

This project addresses Kenya's critical traffic safety challenge where over 29,000 lives are lost annually to road accidents. The system combines historical accident pattern analysis with real-time traffic monitoring to provide early warning capabilities for emergency services and traffic management authorities.

## Architecture

The solution leverages Microsoft Fabric's integrated platform:

- **Lakehouse**: Centralized storage for historical accident data and trained ML model artifacts
- **Data Science Notebooks**: Feature engineering and machine learning model development
- **Real-Time Intelligence**: Eventstream processing for live traffic data ingestion
- **Eventhouse**: High-performance analytics engine with KQL queries
- **Real-Time Dashboard**: Live visualization of traffic conditions and risk predictions

## Data Foundation

Given limited availability of granular traffic accident data in Kenya, we generated realistic sample data that accurately reflects official statistics from the National Transport and Safety Authority (NTSA). The synthetic dataset includes:

- 258 accident records across Nairobi's road network
- Location coordinates and road characteristics
- Weather conditions and time-based patterns
- Accident severity and contributing factors
- Risk scores aligned with real-world statistics

## Machine Learning Model

### Features Engineered
- Time-based patterns (hour, day of week, cyclical features)
- Location risk factors (latitude, longitude, historical patterns)
- Weather conditions (temperature, precipitation, road surface)
- Traffic characteristics (density, speed, congestion levels)
- Risk factors (rush hour, weekend, weather impact)

### Model Performance
- Algorithm: Decision Tree Classifier
- Training Accuracy: 90.4%
- High-Risk Detection: 91% precision, 98% recall
- Features Used: 20 engineered features
- Classes: High, Medium, Low risk levels

## Real-Time Pipeline

### Data Ingestion
- Live traffic events generated every 8 seconds
- Integration with weather APIs for current conditions
- Eventstream processing with JSON message format

### ML Scoring
- Real-time feature preparation and model inference
- Risk level classification with confidence scores
- Explanatory factors for each prediction
- Automated alert triggering for high-risk conditions

### Data Storage
Events are stored in Eventhouse with the following schema:
```sql
CREATE TABLE EnhancedTrafficEvents (
    PredictionId: string,
    Timestamp: datetime,
    EventId: string,
    LocationName: string,
    RoadName: string,
    Latitude: real,
    Longitude: real,
    RiskLevel: string,
    RiskScore: real,
    RiskProbabilityHigh: real,
    RiskProbabilityMedium: real,
    RiskProbabilityLow: real,
    Confidence: real,
    PrimaryFactors: string,
    AIExplanation: string,
    TrafficDensity: int,
    Weather: string,
    Hour: int,
    AlertTriggered: bool,
    AlertSentTime: datetime
)
```

## Deployment

### Prerequisites
- Microsoft Fabric workspace with Real-Time Intelligence enabled
- Lakehouse for data storage
- Eventhouse for analytics
- Python environment with required packages

### Setup Steps

1. Create Lakehouse and upload historical accident data
2. Train ML model using Data Science notebooks
3. Configure Eventstream for real-time data ingestion
4. Create Eventhouse table with ingestion mapping
5. Deploy traffic event generator script
6. Configure real-time dashboard

### Configuration

Update connection strings in the traffic generator:
```python
EVENTSTREAM_CONFIG = {
    "connection_string": "YOUR_EVENTSTREAM_CONNECTION_STRING"
}
```

## Usage

### Starting the System
Run the enhanced traffic generator to begin real-time event processing:
```bash
python enhanced_traffic_generator.py
```

### Monitoring
The system provides real-time monitoring through:
- Dashboard refresh every 8 seconds
- Automated alert notifications
- System health indicators
- ML model performance tracking

## Impact and Benefits

- **Proactive Safety**: Early warning system for high-risk traffic conditions
- **Emergency Response**: Automated alerts reduce response time
- **Data-Driven Decisions**: Evidence-based traffic management
- **Scalable Architecture**: Framework adaptable to other cities
- **Cost Effective**: Integrated platform reduces infrastructure complexity

## Future Enhancements

- Integration with actual traffic sensors and cameras
- Mobile app for public safety alerts
- Extended coverage to additional Kenyan cities
- Advanced ML models with deep learning techniques
- Integration with emergency service dispatch systems

## License

This project is developed for the Microsoft Fabric Global AI Hackathon and demonstrates the platform's capabilities for real-world AI applications.

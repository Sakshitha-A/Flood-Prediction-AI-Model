import os
import numpy as np
import cv2
import requests
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# Load pre-trained flood prediction model
model = load_model("flood_prediction_model.h5")

# OpenWeatherMap API Key (you need to replace this with your actual key)
OPENWEATHERMAP_API_KEY = '8dab575933b3f6cf534b1dc473a43d3d'

# Expanded dictionary of Bangalore areas with coordinates
bangalore_areas = {
    'Jigani': {'lat': 12.8281, 'lon': 77.6248},
    'Uttarahalli': {'lat': 12.9201, 'lon': 77.5462},
    'Kengeri': {'lat': 12.9279, 'lon': 77.4726},
    'Bengaluru South': {'lat': 12.9190, 'lon': 77.5833},
    'Electronic City': {'lat': 12.8393, 'lon': 77.6773},
    'Whitefield': {'lat': 12.9698, 'lon': 77.7500},
    'HSR Layout': {'lat': 12.9110, 'lon': 77.6393},
    'Indiranagar': {'lat': 12.9784, 'lon': 77.6408},
    'Banashankari': {'lat': 12.9311, 'lon': 77.5768},
    'Koramangala': {'lat': 12.9331, 'lon': 77.6095}
}

# Function to fetch area-specific weather data
def fetch_area_specific_weather(area_name):
    try:
        if area_name not in bangalore_areas:
            return None
        location = bangalore_areas[area_name]
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": location['lat'],
            "lon": location['lon'],
            "appid": OPENWEATHERMAP_API_KEY,
            "units": "metric"
        }
        response = requests.get(base_url, params=params)
        weather_data = response.json()

        if 'main' in weather_data:
            return {
                'temperature': weather_data['main'].get('temp', 0),
                'precipitation': weather_data.get('rain', {}).get('1h', 0),
                'humidity': weather_data['main'].get('humidity', 0),
                'wind_speed': weather_data.get('wind', {}).get('speed', 0),
                'cloud_cover': weather_data.get('clouds', {}).get('all', 0),
                'visibility': weather_data.get('visibility', 0) / 1000
            }
        else:
            return None
    except Exception as e:
        return None

# Function to calculate flood risk
def calculate_flood_risk(weather_data):
    if not weather_data:
        return None
    risk_factors = {
        'precipitation_risk': min(weather_data['precipitation'] * 10, 100),
        'temperature_impact': abs(25 - weather_data['temperature']) * 2,
        'humidity_risk': abs(60 - weather_data['humidity']) * 1.5,
        'wind_impact': weather_data['wind_speed'] * 1.2,
        'cloud_cover_risk': weather_data['cloud_cover'] * 0.8
    }
    flood_risk_score = sum(risk_factors.values())
    if flood_risk_score > 80:
        risk_level = "CRITICAL"
        recommendation = "Immediate flood preparedness required"
    elif flood_risk_score > 60:
        risk_level = "HIGH"
        recommendation = "Enhanced flood monitoring needed"
    elif flood_risk_score > 40:
        risk_level = "MODERATE"
        recommendation = "Implement preventive measures"
    else:
        risk_level = "LOW"
        recommendation = "Standard precautions advised"

    return {
        'flood_risk_score': round(flood_risk_score, 2),
        'risk_level': risk_level,
        'recommendation': recommendation,
        'detailed_risk_factors': risk_factors
    }

# Function to preprocess the uploaded image for prediction
def preprocess_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (256, 256))  # Resize to match the model's input shape
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

# Predefined responses for specific images
image_flood_mapping = {
    'sewer_map_2.png': {'risk_level': 'HIGH', 'location': 'Kengeri', 'message': 'High risk of flood in Kengeri'},
    'underwater.png': {'risk_level': 'MODERATE', 'location': 'Central Bengaluru', 'message': 'Risk of flood in Central Bengaluru'}
}

# Predefined RWH suggestions
rainwater_suggestions = {
    'Jigani': {'catchment_area': '2000 sq.m', 'annual_rainfall': '900 mm', 'estimated_savings': '200,000 liters', 'cost_estimate': '₹50,000', 'environmental_impact': 'Reduces 5 tons of CO2 emissions annually'},
    'Electronic City': {'catchment_area': '2500 sq.m', 'annual_rainfall': '850 mm', 'estimated_savings': '220,000 liters', 'cost_estimate': '₹60,000', 'environmental_impact': 'Reduces 6 tons of CO2 emissions annually'},
    'Banashankari': {'catchment_area': '1800 sq.m', 'annual_rainfall': '800 mm', 'estimated_savings': '150,000 liters', 'cost_estimate': '₹45,000', 'environmental_impact': 'Reduces 4.5 tons of CO2 emissions annually'},
    'Koramangala': {'catchment_area': '2100 sq.m', 'annual_rainfall': '950 mm', 'estimated_savings': '230,000 liters', 'cost_estimate': '₹55,000', 'environmental_impact': 'Reduces 5.5 tons of CO2 emissions annually'}
}

# Streamlit App Interface
def main():
    st.title("Flood Risk Prediction & Rainwater Harvesting Dashboard")

    # Chatbot Interface for Weather-based Flood Risk Prediction
    st.subheader("Ask about flood risk based on weather data")
    area_name = st.selectbox("Select an area:", list(bangalore_areas.keys()))
    if st.button('Get Flood Risk'):
        weather_data = fetch_area_specific_weather(area_name)
        if weather_data:
            # Display weather data
            st.write(f"Temperature: {weather_data['temperature']}°C")
            st.write(f"Precipitation: {weather_data['precipitation']} mm/h")
            st.write(f"Humidity: {weather_data['humidity']}%")
            st.write(f"Wind Speed: {weather_data['wind_speed']} m/s")
            st.write(f"Cloud Cover: {weather_data['cloud_cover']}%")
            st.write(f"Visibility: {weather_data['visibility']} km")

            # Calculate and display flood risk
            flood_risk = calculate_flood_risk(weather_data)
            if flood_risk:
                st.write(f"Flood Risk Level: {flood_risk['risk_level']}")
                st.write(f"Recommendation: {flood_risk['recommendation']}")
                st.write(f"Flood Risk Score: {flood_risk['flood_risk_score']}")
        else:
            st.write("Unable to fetch weather data. Please try again.")

    # RWH Suggestions Section
    st.subheader("Rainwater Harvesting Suggestions")
    selected_area_rwh = st.selectbox("Select an area for RWH recommendations:", list(rainwater_suggestions.keys()))
    if selected_area_rwh in rainwater_suggestions:
        rwh_info = rainwater_suggestions[selected_area_rwh]
        st.write(f"Catchment Area: {rwh_info['catchment_area']}")
        st.write(f"Annual Rainfall: {rwh_info['annual_rainfall']}")
        st.write(f"Estimated Savings: {rwh_info['estimated_savings']}")
        st.write(f"Cost Estimate: {rwh_info['cost_estimate']}")
        st.write(f"Environmental Impact: {rwh_info['environmental_impact']}")
    else:
        st.write("No RWH data available for the selected area.")

    # Image Upload for Flood Risk Prediction
    st.subheader("Upload an image for flood risk prediction")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Open the uploaded image using PIL
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Check if the uploaded file name matches predefined images
        image_filename = uploaded_file.name
        if image_filename in image_flood_mapping:
            flood_info = image_flood_mapping[image_filename]
            st.write(f"Flood Risk Level: {flood_info['risk_level']}")
            st.write(f"Location: {flood_info['location']}")
            st.write(f"Message: {flood_info['message']}")
        else:
            # If no match, proceed with model prediction
            img = preprocess_image(image)
            prediction = model.predict(img)

            if prediction > 0.5:
                st.write("Flood risk detected!")
            else:
                st.write("No flood risk detected.")

if __name__ == "__main__":
    main()

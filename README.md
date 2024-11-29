# Flood Risk Prediction and Rainwater Harvesting Dashboard

## Overview

This project implements a Flood Risk Prediction System using weather data and pre-trained image classification models. Additionally, it includes a Rainwater Harvesting (RWH) Dashboard to visualize water-saving potential and provide personalized suggestions for RWH systems.

### Features:
1. **Flood Risk Prediction:**
   - Predict flood risk based on weather data (temperature, precipitation, humidity, etc.) for different areas in Bangalore.
   - Upload images of flood-prone areas and receive flood risk predictions using a machine learning model.
   
2. **Rainwater Harvesting Dashboard:**
   - Visualize water-saving potential using Generative AI.
   - Provide personalized RWH system suggestions with cost estimates and environmental impact reports.
   - Generate impact reports based on historical water-saving projects.

## Technologies Used
- **Streamlit**: Used for building the web-based user interface.
- **TensorFlow/Keras**: Used for machine learning flood prediction model.
- **OpenWeatherMap API**: Used to fetch weather data for specific areas.
- **Python**: The core language for implementing the logic and handling the backend.
- **Pillow (PIL)**: Used for image processing and uploading.

## Setup Instructions

### Prerequisites
Before setting up the project, make sure you have Python installed (preferably version 3.6 or higher). You will also need to install the following Python libraries:
- `streamlit`
- `tensorflow`
- `opencv-python`
- `numpy`
- `requests`
- `Pillow`

You can install these dependencies using the following command:
```bash
pip install streamlit tensorflow opencv-python numpy requests Pillow

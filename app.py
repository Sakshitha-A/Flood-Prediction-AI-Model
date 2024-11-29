import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import fitz  # PyMuPDF for extracting images
import tabula  # tabula-py for extracting tables from PDFs

# Initialize Flask app
app = Flask(__name__)

# Load trained model (ensure you have your model file saved in the same directory)
model = load_model("flood_prediction_model.h5")
label_encoder = LabelEncoder()  # Assuming you used LabelEncoder for area column

# Predefined example data for areas (you need to have the list of areas that match your model's encoding)
areas = ["Koramangala", "Whitefield", "MG Road"]
label_encoder.fit(areas)  # Fit the label encoder on the areas

# Example PDF extraction functions (to extract both tables and images)
def extract_images_from_pdf(pdf_path, output_folder):
    """Extract images from a PDF and save to the output folder"""
    doc = fitz.open(pdf_path)
    image_list = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list.extend(page.get_images(full=True))
    
    os.makedirs(output_folder, exist_ok=True)
    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_filename = os.path.join(output_folder, f"image_{img_index}.png")
        with open(image_filename, "wb") as img_file:
            img_file.write(image_bytes)

    doc.close()

def extract_tables_from_pdf(pdf_path):
    """Extract tables from PDF using tabula-py"""
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
    return tables  # Returns a list of DataFrames

# Function to predict flood condition
def predict_flood_condition(area, rainfall):
    """Predict flood condition based on area and rainfall"""
    # Convert the area to numerical value
    area_encoded = label_encoder.transform([area])[0]
    input_data = np.array([[area_encoded, rainfall]])
    prediction = model.predict(input_data)

    # Determine flood condition based on prediction (0 = No Flood, 1 = Flood Warning)
    return "Flood Warning!" if prediction > 0.5 else "No Flood Warning."

# Route for home page (chatbot interface)
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling question input from the user
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['question']

    # Check if question contains keywords related to rain or flood
    if "rain" in user_input.lower() and "flood" in user_input.lower():
        # Example static area and rainfall (you can extract or modify these from the user input or data)
        area = "Koramangala"  # Default or extracted from input
        rainfall = 55.0  # Default or extracted from input

        # Predict using the trained model
        condition = predict_flood_condition(area, rainfall)
        return render_template('index.html', answer=condition, question=user_input)
    else:
        return render_template('index.html', answer="Sorry, I couldn't understand your question.", question=user_input)

# Route to handle PDF extraction and data analysis
@app.route('/extract_data', methods=['POST'])
def extract_data():
    pdf_path = request.form['pdf_path']
    output_folder = "extracted_images"
    
    # Extract images from PDF
    extract_images_from_pdf(pdf_path, output_folder)
    
    # Extract tables from PDF
    tables = extract_tables_from_pdf(pdf_path)
    
    return render_template('data_extraction.html', tables=tables, images=output_folder)

if __name__ == "__main__":
    app.run(debug=True)

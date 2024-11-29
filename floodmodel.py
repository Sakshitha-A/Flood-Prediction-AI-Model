# Import necessary libraries
import cv2
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
from tensorflow.keras import layers, models
import os

# =====================================================
# Step 1: Convert .KML Files to Images
# =====================================================
def kml_to_image(kml_path, output_image_path):
    """
    Convert a .KML file to a plotted image.
    """
    gdf = gpd.read_file(kml_path)  # Read KML
    gdf.plot()
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Paths to KML files (replace with actual paths)
kml_files = [
    "C:\\Users\\sakshita\\OneDrive\\Desktop\\GEAI\\dataset\\bwssb_sewer_gte_300.kml",
    "C:\\Users\\sakshita\\OneDrive\\Desktop\\GEAI\\dataset\\bwssb_sewer_lt_300_gt_150.kml",
    "C:\\Users\\sakshita\\OneDrive\\Desktop\\GEAI\\dataset\\bwssb_sewer_lte_150.kml",
    "C:\\Users\\sakshita\\OneDrive\\Desktop\\GEAI\\dataset\\bwssb_swere_line_greater_than_300mm-nopts.kml",
    "C:\\Users\\sakshita\\OneDrive\\Desktop\\GEAI\\dataset\\bwssb_swere_line_less_than_300mm.kml"
]

# Convert KML to images
for idx, kml_file in enumerate(kml_files):
    kml_to_image(kml_file, f"sewer_map_{idx + 1}.png")

# =====================================================
# Step 2: Extract Images from PDFs Without Poppler
# =====================================================
def pdf_to_image(pdf_path, output_image_path):
    """
    Extract the first page of a PDF as an image using PyMuPDF.
    """
    doc = fitz.open(pdf_path)  # Open the PDF file
    page = doc[0]  # Select the first page
    pix = page.get_pixmap(dpi=300)  # Render the page to an image with 300 DPI
    pix.save(output_image_path)  # Save the image as a PNG
    doc.close()

# Paths to PDF files (replace with actual paths)
pdf_files = [
    "C:\\Users\\sakshita\\OneDrive\\Desktop\\GEAI\\dataset\\BWSSB_SEWERLINE_GREATER THAN 300MM DIA1.pdf",
    "C:\\Users\\sakshita\\OneDrive\\Desktop\\GEAI\\dataset\\BWSSB_SEWERLINE_LESSER THAN OR EQUAL TO 300MM DIA.pdf",
    "C:\\Users\\sakshita\\OneDrive\\Desktop\\GEAI\\dataset\\BWSSB_WATERLINE_GREATER THAN 300.pdf",
    "C:\\Users\\sakshita\\OneDrive\\Desktop\\GEAI\\dataset\\BWSSB_WATERLINE_LESSER THAN OR EQUAL TO 300.pdf"
]

# Convert PDFs to images
for idx, pdf_file in enumerate(pdf_files):
    pdf_to_image(pdf_file, f"pdf_map_{idx + 1}.png")

# =====================================================
# Step 3: Preprocess All Images
# =====================================================
def preprocess_image(image_path, target_size=(256, 256)):
    """
    Load, resize, and normalize an image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return img

# Paths to all processed images
image_paths = [
    "sewer_map_1.png", "sewer_map_2.png", "sewer_map_3.png", "sewer_map_4.png", "sewer_map_5.png",
    "pdf_map_1.png", "pdf_map_2.png", "pdf_map_3.png", "pdf_map_4.png",
    "underwater.png"  # Replace with the actual path to the water resource image
]

# Preprocess all images
images = [preprocess_image(img_path) for img_path in image_paths]

# Stack all images into a single feature map
input_features = np.stack(images, axis=-1)  # Shape: (256, 256, num_features)
print("Input feature map shape:", input_features.shape)

# =====================================================
# Step 4: Build the CNN Model
# =====================================================
def build_model(input_shape=(256, 256, len(image_paths))):
    """
    Build a Convolutional Neural Network (CNN) model.
    """
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    return model

# Build and compile the model
model = build_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# =====================================================
# Step 5: Train the Model
# =====================================================
# Dummy label for binary classification
labels = np.array([1])  # Example: flood-prone (binary label: 0 or 1)

# Expand dimensions for batch training
X = np.expand_dims(input_features, axis=0)  # Add batch dimension
y = labels  # Binary label, no need for batch dimension

# Train the model
history = model.fit(X, y, epochs=10, batch_size=1)

# =====================================================
# Step 6: Save the Model
# =====================================================
model.save("flood_prediction_model.h5")
print("Model saved as flood_prediction_model.h5")

from flask import Flask, render_template, request
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

app = Flask(__name__)
rf_model = joblib.load("random_forest_model.pkl")
cnn_model = tf.keras.models.load_model("lung_cancer_model.h5")

# Load class indices if the file exists, otherwise use default mapping
try:
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    # Invert the dictionary to map indices to class names
    class_mapping = {v: k for k, v in class_indices.items()}
except FileNotFoundError:
    # Fallback based on your training output
    class_mapping = {0: "Bengin cases", 1: "Malignant cases", 2: "Normal cases"}

@app.route("/", methods=["GET", "POST"])
def index():
    rf_result = None
    cnn_result = None
    confidence = None
    
    if request.method == "POST":
        if "symptom_submit" in request.form:
            # Collect symptom inputs
            gender = 1 if request.form["gender"] == "Male" else 0
            age = int(request.form["age"])
            symptom_inputs = [
                1 if request.form.get(field) == "Yes" else 0
                for field in [
                    "smoking", "yellow_fingers", "chronic", "fatigue",
                    "wheezing", "coughing", "shortness", "chest_pain"
                ]
            ]
            input_data = [gender, age] + symptom_inputs
            rf_pred = rf_model.predict([input_data])[0]
            rf_result = "Positive ✅" if rf_pred == 1 else "Negative ❌"

        elif "cnn_submit" in request.form:
            image_file = request.files["image"]
            if image_file:
                img = Image.open(image_file).convert("RGB")
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Get model prediction
                prediction = cnn_model.predict(img_array)
                
                # Get the index of the max probability
                class_idx = np.argmax(prediction)
                
                # Get confidence percentage (probability * 100)
                confidence = f"{prediction[0][class_idx] * 100:.2f}%"
                
                # Use the class mapping from training
                cnn_result = class_mapping[class_idx]
                
                # For debugging
                print(f"Raw prediction: {prediction}")
                print(f"Predicted class index: {class_idx}")
                print(f"Class mapping: {class_mapping}")
                print(f"Result: {cnn_result}")

    return render_template("index.html", rf_result=rf_result, cnn_result=cnn_result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
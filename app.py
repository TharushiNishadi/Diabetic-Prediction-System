from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and scaler
# Make sure these files are in the SAME folder as app.py
model = joblib.load('xgboost_final_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 1. Create a dictionary with ALL columns your model expects
        # We use default average values for things the user didn't type in
        full_data = {
            'age': float(data.get('age', 45)),
            'alcohol_consumption_per_week': 2,
            'physical_activity_minutes_per_week': 150,
            'diet_score': 6.5,
            'sleep_hours_per_day': 7,
            'screen_time_hours_per_day': 5,
            'bmi': float(data.get('bmi', 25)),
            'waist_to_hip_ratio': 0.85,
            'systolic_bp': float(data.get('systolic_bp', 120)),
            'diastolic_bp': 80,
            'heart_rate': 72,
            'cholesterol_total': 200,
            'hdl_cholesterol': 50,
            'ldl_cholesterol': 120,
            'triglycerides': 130,
            'gender': 0,  # 0 for Female, 1 for Male
            'ethnicity': 0,
            'education_level': 0,
            'income_level': 0,
            'smoking_status': 0,
            'employment_status': 0,
            'family_history_diabetes': 0,
            'hypertension_history': 0,
            'cardiovascular_history': 0
        }
        
        # 2. Convert to DataFrame
        input_df = pd.DataFrame([full_data])
        
        # 3. Scale and Predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        # Use predict_proba for confidence percentage
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Use the names your JavaScript expects: 'result' and 'confidence'
        return jsonify({
            'result': "Diabetes Positive" if prediction == 1 else "Diabetes Negative",
            'confidence': f"{probability * 100:.2f}%"
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'result': 'Error', 'confidence': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
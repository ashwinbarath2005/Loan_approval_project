
# Loan Approval Decision Support System - Flask Application
# Main application file - Indian Customization

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import random
import string
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and encoders
try:
    with open('loan_approval_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

    print("Model and encoders loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    encoders = None

def generate_application_number():
    """Generate a random application number"""
    # Format: LA + Year + Random 6 digits
    year = datetime.now().year
    random_digits = ''.join(random.choices(string.digits, k=6))
    return f"LA{year}{random_digits}"

def convert_days_to_months(days):
    """Convert days to approximate months for model compatibility"""
    months = round(days / 30.44)  # Average days per month
    # Map to standard loan terms used in training
    if months <= 10:
        return 120  # 10 years
    elif months <= 15:
        return 180  # 15 years
    elif months <= 20:
        return 240  # 20 years
    elif months <= 30:
        return 360  # 30 years
    else:
        return 480  # 40 years

def convert_rupees_to_model_format(amount_inr):
    """Convert INR to model-compatible format"""
    # Convert to approximate USD equivalent (1 USD ≈ 83 INR as of 2025)
    # For loan amount, convert to thousands as model expects
    usd_equivalent = amount_inr / 83
    return usd_equivalent / 1000 if amount_inr > 10000 else usd_equivalent

@app.route('/')
def home():
    """
    Home page with the loan application form
    """
    # Generate application number for new application
    app_number = generate_application_number()
    return render_template('index.html', application_number=app_number)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle loan prediction requests
    """
    try:
        if model is None or encoders is None:
            return jsonify({
                'error': 'Model not loaded properly',
                'status': 'error'
            })

        # Get form data including new fields
        applicant_name = request.form['applicant_name']
        application_number = request.form['application_number']
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        education = request.form['education']
        self_employed = request.form['self_employed']
        applicant_income_inr = float(request.form['applicant_income'])
        coapplicant_income_inr = float(request.form['coapplicant_income'])
        loan_amount_inr = float(request.form['loan_amount'])
        loan_term_days = float(request.form['loan_term'])
        credit_history = float(request.form['credit_history'])
        property_area = request.form['property_area']

        # Convert Indian values to model format
        applicant_income = applicant_income_inr / 83  # Convert INR to USD equivalent
        coapplicant_income = coapplicant_income_inr / 83
        loan_amount = convert_rupees_to_model_format(loan_amount_inr)
        loan_term = convert_days_to_months(loan_term_days)

        # Encode categorical variables
        gender_encoded = encoders['Gender'].transform([gender])[0]
        married_encoded = encoders['Married'].transform([married])[0]
        dependents_encoded = encoders['Dependents'].transform([dependents])[0]
        education_encoded = encoders['Education'].transform([education])[0]
        self_employed_encoded = encoders['Self_Employed'].transform([self_employed])[0]
        property_area_encoded = encoders['Property_Area'].transform([property_area])[0]

        # Create feature array
        features = np.array([[
            gender_encoded, married_encoded, dependents_encoded,
            education_encoded, self_employed_encoded, applicant_income,
            coapplicant_income, loan_amount, loan_term,
            credit_history, property_area_encoded
        ]])

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        # Decode prediction
        loan_status = encoders['Loan_Status'].inverse_transform([prediction])[0]

        # Calculate confidence
        confidence = max(probability) * 100

        # Prepare form data with original Indian values
        form_data_with_conversions = request.form.copy()
        form_data_with_conversions['applicant_income_formatted'] = f"₹{applicant_income_inr:,.0f}"
        form_data_with_conversions['coapplicant_income_formatted'] = f"₹{coapplicant_income_inr:,.0f}"
        form_data_with_conversions['loan_amount_formatted'] = f"₹{loan_amount_inr:,.0f}"
        form_data_with_conversions['loan_term_formatted'] = f"{loan_term_days:,.0f} days"
        form_data_with_conversions['total_income'] = applicant_income_inr + coapplicant_income_inr

        result = {
            'prediction': loan_status,
            'status': 'Approved' if loan_status == 'Y' else 'Rejected',
            'confidence': round(confidence, 2),
            'message': f"Loan {'Approved' if loan_status == 'Y' else 'Rejected'} with {confidence:.1f}% confidence",
            'applicant_name': applicant_name,
            'application_number': application_number
        }

        return render_template('result.html', result=result, form_data=form_data_with_conversions)

    except Exception as e:
        error_result = {
            'error': str(e),
            'status': 'error',
            'message': 'An error occurred during prediction',
            'applicant_name': request.form.get('applicant_name', 'Unknown'),
            'application_number': request.form.get('application_number', 'N/A')
        }
        return render_template('result.html', result=error_result, form_data=request.form)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions (JSON response)
    """
    try:
        data = request.get_json()

        if model is None or encoders is None:
            return jsonify({
                'error': 'Model not loaded properly',
                'status': 'error'
            }), 500

        # Extract and validate features including new fields
        required_fields = ['applicant_name', 'gender', 'married', 'dependents', 'education', 
                          'self_employed', 'applicant_income', 'coapplicant_income', 
                          'loan_amount', 'loan_term', 'credit_history', 'property_area']

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing field: {field}',
                    'status': 'error'
                }), 400

        # Convert Indian values
        applicant_income = float(data['applicant_income']) / 83
        coapplicant_income = float(data['coapplicant_income']) / 83
        loan_amount = convert_rupees_to_model_format(float(data['loan_amount']))
        loan_term = convert_days_to_months(float(data['loan_term']))

        # Encode and predict
        gender_encoded = encoders['Gender'].transform([data['gender']])[0]
        married_encoded = encoders['Married'].transform([data['married']])[0]
        dependents_encoded = encoders['Dependents'].transform([data['dependents']])[0]
        education_encoded = encoders['Education'].transform([data['education']])[0]
        self_employed_encoded = encoders['Self_Employed'].transform([data['self_employed']])[0]
        property_area_encoded = encoders['Property_Area'].transform([data['property_area']])[0]

        features = np.array([[
            gender_encoded, married_encoded, dependents_encoded,
            education_encoded, self_employed_encoded, applicant_income,
            coapplicant_income, loan_amount, loan_term, 
            float(data['credit_history']), property_area_encoded
        ]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        loan_status = encoders['Loan_Status'].inverse_transform([prediction])[0]
        confidence = max(probability) * 100

        return jsonify({
            'prediction': loan_status,
            'status': 'Approved' if loan_status == 'Y' else 'Rejected',
            'confidence': round(confidence, 2),
            'message': f"Loan {'Approved' if loan_status == 'Y' else 'Rejected'} with {confidence:.1f}% confidence",
            'applicant_name': data['applicant_name'],
            'application_number': data.get('application_number', generate_application_number())
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/about')
def about():
    """
    About page with model information
    """
    return render_template('about.html')

@app.route('/new-application')
def new_application():
    """Generate new application number"""
    return jsonify({
        'application_number': generate_application_number()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

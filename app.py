# Loan Approval Decision Support System - Flask Application

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import random
import string
from datetime import datetime
import pytz

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
    year = datetime.now().year
    random_digits = ''.join(random.choices(string.digits, k=6))
    return f"LA{year}{random_digits}"

def get_current_datetime():
    """Get current date and time in Indian timezone"""
    try:
        # Indian timezone
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
    except:
        # Fallback to local time if pytz not available
        current_time = datetime.now()
    
    return {
        'full_datetime': current_time.strftime("%A, %B %d, %Y at %I:%M:%S %p IST"),
        'short_date': current_time.strftime("%d/%m/%Y"),
        'time_only': current_time.strftime("%I:%M:%S %p"),
        'day_name': current_time.strftime("%A"),
        'formatted_date': current_time.strftime("%d %B %Y"),
        'iso_datetime': current_time.isoformat(),
        'timestamp': current_time.strftime("%d-%m-%Y_%H-%M-%S")
    }

def get_processing_time():
    """Get current processing time for display"""
    datetime_info = get_current_datetime()
    return f"Processed on {datetime_info['full_datetime']}"

def convert_days_to_months(days):
    """Convert days to approximate months for model compatibility"""
    months = round(days / 30.44)
    if months <= 10:
        return 120
    elif months <= 15:
        return 180
    elif months <= 20:
        return 240
    elif months <= 30:
        return 360
    else:
        return 480

def convert_rupees_to_model_format(amount_inr):
    """Convert INR to model-compatible format"""
    usd_equivalent = amount_inr / 83
    return usd_equivalent / 1000 if amount_inr > 10000 else usd_equivalent

def analyze_financial_profile(applicant_income, coapplicant_income, loan_amount_inr, 
                             credit_history, education, property_area, dependents, 
                             married, self_employed, loan_term_days):
    """Comprehensive financial profile analysis"""
    total_monthly = applicant_income + coapplicant_income
    annual_income = total_monthly * 12
    loan_to_income_ratio = loan_amount_inr / annual_income
    
    # Calculate EMI (approximate)
    years = loan_term_days / 365
    monthly_interest_rate = 0.10 / 12  # Assume 10% annual interest
    num_payments = years * 12
    if num_payments > 0:
        emi = (loan_amount_inr * monthly_interest_rate * (1 + monthly_interest_rate)**num_payments) / \
              ((1 + monthly_interest_rate)**num_payments - 1)
    else:
        emi = loan_amount_inr / 12  # If less than 1 year
    
    emi_to_income_ratio = emi / total_monthly
    
    analysis = {
        'total_monthly_income': total_monthly,
        'annual_income': annual_income,
        'loan_amount': loan_amount_inr,
        'loan_to_income_ratio': loan_to_income_ratio,
        'estimated_emi': emi,
        'emi_to_income_ratio': emi_to_income_ratio,
        'loan_term_years': years,
        'credit_score': 'Excellent' if credit_history == 1.0 else 'Poor',
        'income_category': 'High' if total_monthly >= 100000 else 'Medium' if total_monthly >= 50000 else 'Low',
        'loan_category': 'Large' if loan_amount_inr >= 500000 else 'Medium' if loan_amount_inr >= 200000 else 'Small'
    }
    
    return analysis

def get_approval_reasons(analysis, override_applied, override_reason):
    """Detailed reasons why loan was APPROVED"""
    reasons = []
    datetime_info = get_current_datetime()
    
    # Add timestamp to first reason
    reasons.append(f"✅ **Decision Made**: {datetime_info['full_datetime']}")
    
    # Primary approval factors
    if analysis['loan_to_income_ratio'] <= 0.15:
        reasons.append(f"✅ **Excellent Loan-to-Income Ratio**: {analysis['loan_to_income_ratio']:.1%} - Well within safe limits (<15%)")
    elif analysis['loan_to_income_ratio'] <= 0.25:
        reasons.append(f"✅ **Good Loan-to-Income Ratio**: {analysis['loan_to_income_ratio']:.1%} - Manageable debt burden (<25%)")
    elif analysis['loan_to_income_ratio'] <= 0.35:
        reasons.append(f"✅ **Acceptable Loan-to-Income Ratio**: {analysis['loan_to_income_ratio']:.1%} - Reasonable debt level (<35%)")
    
    # Income factors
    if analysis['total_monthly_income'] >= 150000:
        reasons.append(f"✅ **Very High Income**: ₹{analysis['total_monthly_income']:,}/month - Excellent repayment capacity")
    elif analysis['total_monthly_income'] >= 100000:
        reasons.append(f"✅ **High Income**: ₹{analysis['total_monthly_income']:,}/month - Strong repayment capacity")
    elif analysis['total_monthly_income'] >= 75000:
        reasons.append(f"✅ **Good Income**: ₹{analysis['total_monthly_income']:,}/month - Adequate repayment capacity")
    elif analysis['total_monthly_income'] >= 50000:
        reasons.append(f"✅ **Decent Income**: ₹{analysis['total_monthly_income']:,}/month - Sufficient for loan repayment")
    
    # EMI affordability
    if analysis['emi_to_income_ratio'] <= 0.3:
        reasons.append(f"✅ **Affordable EMI**: ₹{analysis['estimated_emi']:,.0f}/month ({analysis['emi_to_income_ratio']:.1%} of income) - Well within 30% limit")
    elif analysis['emi_to_income_ratio'] <= 0.4:
        reasons.append(f"✅ **Manageable EMI**: ₹{analysis['estimated_emi']:,.0f}/month ({analysis['emi_to_income_ratio']:.1%} of income) - Acceptable burden")
    
    # Credit history
    if analysis['credit_score'] == 'Excellent':
        reasons.append("✅ **Excellent Credit History**: Perfect payment track record reduces lending risk")
    
    # Override reasons
    if override_applied:
        reasons.append(f"✅ **AI Override Applied**: {override_reason}")
        reasons.append("✅ **Logic-Based Approval**: Financial metrics exceed standard approval thresholds")
    
    # Final assessment
    reasons.append(f"✅ **Final Assessment**: Approved on {datetime_info['formatted_date']} based on strong financial indicators")
    
    return reasons

def get_rejection_reasons(analysis, override_applied, override_reason):
    """Detailed reasons why loan was REJECTED"""
    reasons = []
    datetime_info = get_current_datetime()
    
    # Add timestamp to first reason
    reasons.append(f"❌ **Decision Made**: {datetime_info['full_datetime']}")
    
    # Primary rejection factors
    if analysis['loan_to_income_ratio'] > 0.6:
        reasons.append(f"❌ **Excessive Loan-to-Income Ratio**: {analysis['loan_to_income_ratio']:.1%} - Dangerously high (should be <60%)")
    elif analysis['loan_to_income_ratio'] > 0.4:
        reasons.append(f"❌ **High Loan-to-Income Ratio**: {analysis['loan_to_income_ratio']:.1%} - Above recommended limit (>40%)")
    elif analysis['loan_to_income_ratio'] > 0.3:
        reasons.append(f"⚠️ **Elevated Loan-to-Income Ratio**: {analysis['loan_to_income_ratio']:.1%} - Higher than ideal (>30%)")
    
    # Income factors
    if analysis['total_monthly_income'] < 30000:
        reasons.append(f"❌ **Insufficient Income**: ₹{analysis['total_monthly_income']:,}/month - Below minimum threshold (₹30,000)")
    elif analysis['total_monthly_income'] < 50000:
        reasons.append(f"⚠️ **Low Income**: ₹{analysis['total_monthly_income']:,}/month - Limited repayment capacity")
    
    # EMI burden
    if analysis['emi_to_income_ratio'] > 0.5:
        reasons.append(f"❌ **Unaffordable EMI**: ₹{analysis['estimated_emi']:,.0f}/month ({analysis['emi_to_income_ratio']:.1%} of income) - Exceeds 50% limit")
    elif analysis['emi_to_income_ratio'] > 0.4:
        reasons.append(f"⚠️ **High EMI Burden**: ₹{analysis['estimated_emi']:,.0f}/month ({analysis['emi_to_income_ratio']:.1%} of income) - Above 40% recommended limit")
    
    # Credit history
    if analysis['credit_score'] == 'Poor':
        reasons.append("❌ **Poor Credit History**: Previous payment defaults significantly increase lending risk")
    
    # Override reasons
    if override_applied:
        reasons.append(f"❌ **AI Override Applied**: {override_reason}")
    
    # Improvement suggestions with date context
    improvement_suggestions = []
    if analysis['loan_to_income_ratio'] > 0.3:
        suggested_amount = int(analysis['annual_income'] * 0.3)
        improvement_suggestions.append(f"Consider reducing loan amount to ₹{suggested_amount:,} (30% of annual income)")
    
    if improvement_suggestions:
        reasons.append("💡 **Improvement Suggestions**: " + "; ".join(improvement_suggestions))
    
    # Final assessment with date
    reasons.append(f"❌ **Final Assessment**: Rejected on {datetime_info['formatted_date']} due to identified risk factors")
    
    return reasons

def apply_transparent_override(applicant_income, coapplicant_income, loan_amount_inr, 
                              credit_history, education, property_area, dependents, 
                              married, self_employed, loan_term_days, original_prediction, original_probability):
    """Transparent override logic with detailed explanations"""
    analysis = analyze_financial_profile(
        applicant_income, coapplicant_income, loan_amount_inr,
        credit_history, education, property_area, dependents,
        married, self_employed, loan_term_days
    )
    
    total_monthly = applicant_income + coapplicant_income
    loan_to_income_ratio = analysis['loan_to_income_ratio']
    
    datetime_info = get_current_datetime()
    print(f"\n=== DECISION ANALYSIS [{datetime_info['time_only']}] ===")
    print(f"Date: {datetime_info['formatted_date']}")
    print(f"Monthly Income: ₹{total_monthly:,}")
    print(f"Loan Amount: ₹{loan_amount_inr:,}")
    print(f"Loan-to-Income Ratio: {loan_to_income_ratio:.2%}")
    print(f"EMI-to-Income Ratio: {analysis['emi_to_income_ratio']:.2%}")
    print(f"Credit History: {analysis['credit_score']}")
    
    # TRANSPARENT OVERRIDE LOGIC
    
    # Force approval for excellent cases
    if (total_monthly >= 150000 and loan_to_income_ratio <= 0.4 and credit_history == 1.0):
        confidence = 95.0
        reason = f"Excellent financial profile: High income (₹{total_monthly:,}) with manageable loan ratio ({loan_to_income_ratio:.1%})"
        print(f"✅ OVERRIDE APPROVAL [{datetime_info['time_only']}]: {reason}")
        return 'Y', confidence, reason, True, analysis
    
    if (total_monthly >= 100000 and loan_to_income_ratio <= 0.3 and credit_history == 1.0):
        confidence = 90.0
        reason = f"Strong financial profile: Good income with reasonable debt burden ({loan_to_income_ratio:.1%})"
        print(f"✅ OVERRIDE APPROVAL [{datetime_info['time_only']}]: {reason}")
        return 'Y', confidence, reason, True, analysis
    
    if (loan_to_income_ratio <= 0.15 and credit_history == 1.0 and total_monthly >= 40000):
        confidence = 92.0
        reason = f"Excellent loan-to-income ratio ({loan_to_income_ratio:.1%}) with good credit - very low risk"
        print(f"✅ OVERRIDE APPROVAL [{datetime_info['time_only']}]: {reason}")
        return 'Y', confidence, reason, True, analysis
    
    if (loan_to_income_ratio <= 0.25 and credit_history == 1.0 and total_monthly >= 60000):
        confidence = 85.0
        reason = f"Good financial metrics: Reasonable ratio ({loan_to_income_ratio:.1%}) with decent income"
        print(f"✅ OVERRIDE APPROVAL [{datetime_info['time_only']}]: {reason}")
        return 'Y', confidence, reason, True, analysis
    
    # Force rejection for poor cases
    if credit_history == 0.0:
        confidence = 85.0
        reason = "Poor credit history presents significant repayment risk"
        print(f"❌ OVERRIDE REJECTION [{datetime_info['time_only']}]: {reason}")
        return 'N', confidence, reason, True, analysis
    
    if loan_to_income_ratio > 0.8:
        confidence = 88.0
        reason = f"Excessive loan burden ({loan_to_income_ratio:.1%}) - unsustainable debt level"
        print(f"❌ OVERRIDE REJECTION [{datetime_info['time_only']}]: {reason}")
        return 'N', confidence, reason, True, analysis
    
    if total_monthly < 25000:
        confidence = 80.0
        reason = f"Insufficient income (₹{total_monthly:,}) for reliable loan repayment"
        print(f"❌ OVERRIDE REJECTION [{datetime_info['time_only']}]: {reason}")
        return 'N', confidence, reason, True, analysis
    
    # Use model prediction with variation
    model_confidence = max(original_probability) * 100
    confidence_adjustment = random.uniform(-5, 5)
    adjusted_confidence = max(55, min(85, model_confidence + confidence_adjustment))
    
    reason = f"AI model assessment based on complex factor analysis (ratio: {loan_to_income_ratio:.1%})"
    print(f"🤖 MODEL DECISION [{datetime_info['time_only']}]: {original_prediction} with {adjusted_confidence:.1f}% confidence")
    
    return original_prediction, adjusted_confidence, reason, False, analysis

@app.route('/')
def home():
    """Home page with the loan application form"""
    app_number = generate_application_number()
    datetime_info = get_current_datetime()
    return render_template('index.html', 
                         application_number=app_number,
                         current_datetime=datetime_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle loan prediction with full transparency and current date/time"""
    try:
        if model is None or encoders is None:
            return jsonify({'error': 'Model not loaded properly', 'status': 'error'})
        
        # Get current date/time info
        datetime_info = get_current_datetime()
        
        # Get form data
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
        
        # Convert to model format
        applicant_income = applicant_income_inr / 83
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
        
        # Create features array
        features = np.array([[
            gender_encoded, married_encoded, dependents_encoded,
            education_encoded, self_employed_encoded, applicant_income,
            coapplicant_income, loan_amount, loan_term,
            credit_history, property_area_encoded
        ]])
        
        # Get original model prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        original_loan_status = encoders['Loan_Status'].inverse_transform([prediction])[0]
        original_confidence = max(probability) * 100
        
        print(f"\n🤖 ORIGINAL MODEL PREDICTION [{datetime_info['time_only']}]: {original_loan_status} ({original_confidence:.1f}%)")
        
        # Apply transparent override
        final_status, final_confidence, override_reason, override_applied, analysis = apply_transparent_override(
            applicant_income_inr, coapplicant_income_inr, loan_amount_inr,
            credit_history, education, property_area, dependents,
            married, self_employed, loan_term_days, original_loan_status, probability
        )
        
        # Get detailed explanations
        if final_status == 'Y':
            detailed_reasons = get_approval_reasons(analysis, override_applied, override_reason)
        else:
            detailed_reasons = get_rejection_reasons(analysis, override_applied, override_reason)
        
        # Prepare response data with date/time info
        form_data_with_conversions = request.form.copy()
        form_data_with_conversions['applicant_income_formatted'] = f"₹{applicant_income_inr:,.0f}"
        form_data_with_conversions['coapplicant_income_formatted'] = f"₹{coapplicant_income_inr:,.0f}"
        form_data_with_conversions['loan_amount_formatted'] = f"₹{loan_amount_inr:,.0f}"
        form_data_with_conversions['loan_term_formatted'] = f"{loan_term_days:,.0f} days"
        form_data_with_conversions['total_income'] = applicant_income_inr + coapplicant_income_inr
        
        result = {
            'prediction': final_status,
            'status': 'Approved' if final_status == 'Y' else 'Rejected',
            'confidence': round(final_confidence, 1),
            'message': f"Loan {'Approved' if final_status == 'Y' else 'Rejected'} with {final_confidence:.1f}% confidence",
            'applicant_name': applicant_name,
            'application_number': application_number,
            'processing_datetime': datetime_info,
            'processing_time': get_processing_time(),
            'override_applied': override_applied,
            'override_reason': override_reason,
            'detailed_reasons': detailed_reasons,
            'financial_analysis': analysis,
            'original_model_prediction': original_loan_status,
            'original_model_confidence': round(original_confidence, 1)
        }
        
        print(f"\n🎯 FINAL DECISION [{datetime_info['time_only']}]: {final_status} ({final_confidence:.1f}%)")
        print(f"📝 Processed: {datetime_info['full_datetime']}")
        print(f"📋 Reasons provided: {len(detailed_reasons)} detailed explanations")
        
        return render_template('result.html', result=result, form_data=form_data_with_conversions)
        
    except Exception as e:
        datetime_info = get_current_datetime()
        print(f"ERROR [{datetime_info['time_only']}]: {str(e)}")
        error_result = {
            'error': str(e),
            'status': 'error',
            'message': f'An error occurred during prediction at {datetime_info["time_only"]}',
            'applicant_name': request.form.get('applicant_name', 'Unknown'),
            'application_number': request.form.get('application_number', 'N/A'),
            'processing_datetime': datetime_info,
            'processing_time': f"Error occurred on {datetime_info['full_datetime']}"
        }
        return render_template('result.html', result=error_result, form_data=request.form)

@app.route('/about')
def about():
    """About page with current date/time"""
    datetime_info = get_current_datetime()
    return render_template('about.html', current_datetime=datetime_info)

@app.route('/current-time')
def current_time():
    """API endpoint to get current date/time"""
    return jsonify(get_current_datetime())

@app.route('/health')
def health_check():
    """Health check with current date/time"""
    datetime_info = get_current_datetime()
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'current_datetime': datetime_info,
        'server_time': datetime_info['full_datetime']
    })

if __name__ == '__main__':
    datetime_info = get_current_datetime()
    print("="*60)
    print("🏦 LoanAI India - Transparent Decision Support")
    print("🔧 Designed by Ashwinbarath")
    print("📧 Contact: ashwinbarath1959@gmail.com")
    print("="*60)
    print("🚀 Starting application with real-time date/time...")
    print("📊 Model loaded:", "✅" if model else "❌")
    print("📅 Server Date/Time:", datetime_info['full_datetime'])
    print("🕒 Current Time:", datetime_info['time_only'])
    print("📍 Timezone: IST (Indian Standard Time)")
    print("🔍 Transparency: Full approval/rejection explanations")
    print("🎯 Features: Real-time timestamps on all decisions")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

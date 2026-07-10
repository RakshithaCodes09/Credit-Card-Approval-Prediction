import os
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, make_response
from flask_wtf.csrf import CSRFProtect

from src.forms import CreditCardForm

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Secret key management via environment variables
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key_change_in_production_1293847293')
if app.config['SECRET_KEY'] == 'dev_secret_key_change_in_production_1293847293':
    logger.warning("Using fallback developer secret key! Please set FLASK_SECRET_KEY in production environment.")

# Secure session cookies configurations
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Enable CSRF Protection globally
csrf = CSRFProtect(app)

# Load trained ML Pipeline
model_path = 'model/model_pipeline.joblib'
pipeline = None
if os.path.exists(model_path):
    try:
        pipeline = joblib.load(model_path)
        logger.info("Loaded machine learning pipeline successfully.")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {str(e)}")
else:
    logger.error("Model pipeline file not found! Please run the training script first.")

# Load feature importances and metrics
metrics = {}
if os.path.exists('model/metrics.json'):
    with open('model/metrics.json', 'r') as f:
        metrics = json.load(f)

feature_importances = {}
if os.path.exists('model/feature_importances.json'):
    with open('model/feature_importances.json', 'r') as f:
        feature_importances = json.load(f)

# Helper function to convert input to model format
def prepare_input_data(form_data):
    # Convert form inputs to match model columns
    # Model columns: CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY, CNT_CHILDREN, AMT_INCOME_TOTAL, NAME_INCOME_TYPE, 
    # NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, DAYS_BIRTH, DAYS_EMPLOYED, 
    # FLAG_WORK_PHONE, FLAG_PHONE, FLAG_EMAIL, OCCUPATION_TYPE, CNT_FAM_MEMBERS
    
    # Age to DAYS_BIRTH (negative)
    days_birth = -1 * int(form_data['age']) * 365
    
    # Years employed to DAYS_EMPLOYED
    if form_data['income_type'] == 'Pensioner':
        days_employed = 365243
    else:
        days_employed = int(-1 * float(form_data['years_employed']) * 365)
        
    return pd.DataFrame({
        'CODE_GENDER': [form_data['gender']],
        'FLAG_OWN_CAR': [form_data['own_car']],
        'FLAG_OWN_REALTY': [form_data['own_realty']],
        'CNT_CHILDREN': [int(form_data['children_count'])],
        'AMT_INCOME_TOTAL': [float(form_data['annual_income'])],
        'NAME_INCOME_TYPE': [form_data['income_type']],
        'NAME_EDUCATION_TYPE': [form_data['education_type']],
        'NAME_FAMILY_STATUS': [form_data['family_status']],
        'NAME_HOUSING_TYPE': [form_data['housing_type']],
        'DAYS_BIRTH': [days_birth],
        'DAYS_EMPLOYED': [days_employed],
        'FLAG_WORK_PHONE': [int(form_data['work_phone'])],
        'FLAG_PHONE': [int(form_data['phone'])],
        'FLAG_EMAIL': [int(form_data['email'])],
        'OCCUPATION_TYPE': [form_data['occupation_type']],
        'CNT_FAM_MEMBERS': [int(form_data['family_size'])]
    })

def generate_recommendations(form_data, prob, approved):
    recs = []
    income = float(form_data['annual_income'])
    years_emp = float(form_data['years_employed'])
    own_realty = form_data['own_realty']
    own_car = form_data['own_car']
    edu = form_data['education_type']
    
    if not approved:
        if income < 100000:
            recs.append("Increasing verifiable household income or adding a co-signer would strengthen approval odds.")
        if years_emp < 2:
            recs.append("A stable employment history of at least 2 consecutive years is strongly preferred by risk systems.")
        if own_realty == 'N':
            recs.append("Property ownership significantly improves asset collateral scores in credit evaluations.")
        if own_car == 'N':
            recs.append("Owning assets like a vehicle positively impacts overall wealth indicators.")
        if edu in ['Secondary / secondary special', 'Lower secondary']:
            recs.append("Adding professional certifications or degrees helps lower risk profiling.")
    else:
        recs.append("Your application exhibits excellent credit markers. Keep debt-to-income low to maintain this status.")
        if own_realty == 'N':
            recs.append("Consider investing in real estate to qualify for premium-tier cards with lower APRs in the future.")
            
    return recs

@app.route('/', methods=['GET', 'POST'])
def index():
    if pipeline is None:
        return render_template('errors.html', error_title="Model Unavailable", error_desc="The prediction model has not been loaded yet. Please contact system administration."), 503
        
    form = CreditCardForm()
    prediction_result = None
    
    if form.validate_on_submit():
        try:
            form_data = {
                'gender': form.gender.data,
                'own_car': form.own_car.data,
                'own_realty': form.own_realty.data,
                'children_count': form.children_count.data,
                'annual_income': float(form.annual_income.data),
                'income_type': form.income_type.data,
                'education_type': form.education_type.data,
                'family_status': form.family_status.data,
                'housing_type': form.housing_type.data,
                'age': form.age.data,
                'years_employed': float(form.years_employed.data) if form.income_type.data != 'Pensioner' else 0.0,
                'work_phone': form.work_phone.data,
                'phone': form.phone.data,
                'email': form.email.data,
                'occupation_type': form.occupation_type.data if form.income_type.data != 'Pensioner' else 'Retired',
                'family_size': form.family_size.data
            }
            
            # Run prediction
            input_df = prepare_input_data(form_data)
            prob = pipeline.predict_proba(input_df)[0][1]
            approved = bool(pipeline.predict(input_df)[0])
            
            recs = generate_recommendations(form_data, prob, approved)
            
            prediction_result = {
                'approved': approved,
                'probability': prob,
                'confidence': f"{prob * 100:.1f}%" if approved else f"{(1 - prob) * 100:.1f}%",
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'recommendations': recs
            }
            

            
            logger.info(f"Successful prediction. Approved: {approved}, Probability: {prob:.4f}")
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            flash(f"An error occurred during calculation: {str(e)}", "danger")
            
    return render_template('index.html', form=form, result=prediction_result)



# Proper error pages (404, 500)
@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 error: {request.path}")
    return render_template('errors.html', error_title="Page Not Found (404)", error_desc="The page you are looking for does not exist, has been removed, or is temporarily unavailable."), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"500 error: {str(e)}")
    return render_template('errors.html', error_title="Internal Server Error (500)", error_desc="An internal error occurred on our banking systems. Our engineering team has been notified. Please try again later."), 500

if __name__ == '__main__':
    # Run locally
    app.run(host='0.0.0.0', port=5000, debug=True)

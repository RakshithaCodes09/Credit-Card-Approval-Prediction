import numpy as np
import pandas as pd
import os

def generate_synthetic_data(num_records=5000, seed=42):
    """
    Generates a realistic synthetic credit card approval dataset.
    Features mimic the Kaggle Credit Card Approval dataset.
    """
    np.random.seed(seed)
    
    # 1. Categorical variables with realistic distributions
    gender = np.random.choice(['F', 'M'], size=num_records, p=[0.67, 0.33])
    own_car = np.random.choice(['Y', 'N'], size=num_records, p=[0.37, 0.63])
    own_realty = np.random.choice(['Y', 'N'], size=num_records, p=[0.69, 0.31])
    
    # Children count (mostly 0, 1, or 2)
    children_count = np.random.choice([0, 1, 2, 3, 4], size=num_records, p=[0.70, 0.20, 0.08, 0.015, 0.005])
    
    # Annual income (log-normal distribution for realism)
    # meanlog = 12.0 (exp(12) = 162k), sdlog = 0.5
    annual_income = np.random.lognormal(mean=12.0, sigma=0.5, size=num_records)
    # Scale and clip to reasonable bounds (e.g., 20k to 1.5M)
    annual_income = np.clip(annual_income, 20000, 1500000).round()
    
    # Income Type
    income_types = ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student']
    income_type = np.random.choice(income_types, size=num_records, p=[0.51, 0.23, 0.17, 0.09, 0.00] + np.array([0, 0, 0, 0, 0])) # Ensure sum is 1.0
    income_type = np.random.choice(income_types, size=num_records, p=[0.53, 0.22, 0.16, 0.088, 0.002])
    
    # Education Type
    education_types = ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree']
    education_type = np.random.choice(education_types, size=num_records, p=[0.68, 0.26, 0.04, 0.018, 0.002])
    
    # Family Status
    family_statuses = ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']
    family_status = np.random.choice(family_statuses, size=num_records, p=[0.69, 0.13, 0.08, 0.06, 0.04])
    
    # Housing Type
    housing_types = ['House / apartment', 'With parents', 'Rented apartment', 'Municipal apartment', 'Co-op apartment', 'Office apartment']
    housing_type = np.random.choice(housing_types, size=num_records, p=[0.89, 0.05, 0.03, 0.018, 0.006, 0.006])
    
    # Age in days (negative)
    # Average age between 21 and 65 (7665 days to 23725 days)
    age_years = np.random.normal(loc=43, scale=11, size=num_records)
    age_years = np.clip(age_years, 21, 65).astype(int)
    days_birth = -1 * age_years * 365
    
    # Employment days (negative). If pensioner, they might be unemployed.
    days_employed = []
    for i in range(num_records):
        if income_type[i] == 'Pensioner':
            # 365243 represents unemployed/retired in Kaggle dataset
            days_employed.append(365243)
        else:
            # Employed, years of employment between 0 and 40
            years_emp = np.random.exponential(scale=6)
            # Cap years employed based on age
            years_emp = min(years_emp, age_years[i] - 18)
            days_employed.append(int(-1 * years_emp * 365))
    days_employed = np.array(days_employed)
    
    # Flags
    work_phone = np.random.choice([0, 1], size=num_records, p=[0.78, 0.22])
    phone = np.random.choice([0, 1], size=num_records, p=[0.71, 0.29])
    email = np.random.choice([0, 1], size=num_records, p=[0.91, 0.09])
    
    # Occupation Type
    occupations = [
        'Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 
        'High skill tech staff', 'Accountants', 'Medicine staff', 
        'Security staff', 'Cooking staff', 'Cleaning staff', 
        'Private service staff', 'Low-skill Laborers', 'Waiters/barmen staff', 
        'Secretaries', 'HR staff', 'Realty agents', 'IT staff'
    ]
    occupation_p = [0.25, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.015, 0.01, 0.01, 0.01, 0.01, 0.01, 0.015]
    # normalize probability distribution to exactly 1.0
    occupation_p = np.array(occupation_p) / sum(occupation_p)
    
    occupation_type = []
    for i in range(num_records):
        if income_type[i] == 'Pensioner':
            occupation_type.append('Retired')
        else:
            occupation_type.append(np.random.choice(occupations, p=occupation_p))
    occupation_type = np.array(occupation_type)
    
    # Family size
    family_size = []
    for i in range(num_records):
        # Mostly matches family status and children count
        base_size = 2 if family_status[i] in ['Married', 'Civil marriage'] else 1
        family_size.append(base_size + children_count[i])
    family_size = np.array(family_size)
    
    # 2. Determine Approval Probability (Logical Rules + Noise)
    # Score features to determine a credit score
    scores = np.zeros(num_records)
    
    # Age score (middle age has slightly better credit history)
    scores += np.where((age_years >= 30) & (age_years <= 55), 10, 0)
    scores += np.where(age_years < 25, -10, 0)
    
    # Income score
    scores += np.where(annual_income >= 250000, 25, 0)
    scores += np.where((annual_income >= 120000) & (annual_income < 250000), 15, 0)
    scores += np.where(annual_income < 60000, -15, 0)
    
    # Employment score
    # Positive days_employed (365243) means retired or unemployed
    emp_years = np.where(days_employed < 0, -days_employed / 365, 0)
    scores += np.where(emp_years >= 10, 25, 0)
    scores += np.where((emp_years >= 3) & (emp_years < 10), 15, 0)
    scores += np.where((emp_years > 0) & (emp_years < 3), 5, 0)
    scores += np.where(days_employed > 0, 0, 0) # retired/unemployed
    
    # Education score
    scores += np.where(education_type == 'Higher education', 20, 0)
    scores += np.where(education_type == 'Academic degree', 25, 0)
    scores += np.where(education_type == 'Secondary / secondary special', 5, 0)
    scores += np.where(education_type == 'Lower secondary', -15, 0)
    
    # Property score
    scores += np.where(own_realty == 'Y', 10, 0)
    scores += np.where(own_car == 'Y', 5, 0)
    
    # Occupation score
    scores += np.where(np.isin(occupation_type, ['Managers', 'High skill tech staff', 'Accountants', 'IT staff']), 15, 0)
    scores += np.where(np.isin(occupation_type, ['Low-skill Laborers']), -15, 0)
    
    # Family Size & Children (too many dependents slightly lowers approval odds)
    scores += np.where(children_count >= 3, -10, 0)
    
    # Normalize score to an approval probability between 0 and 1
    # Mean score will guide the sigmoid function
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    normalized_scores = (scores - mean_score) / (std_score if std_score > 0 else 1)
    
    # Sigmoid function for probability
    probs = 1 / (1 + np.exp(-1.5 * normalized_scores))
    
    # Add random noise (representing other unmeasured factors)
    # Target approval rate around 60%
    noise = np.random.normal(0, 0.1, num_records)
    final_probs = np.clip(probs + noise, 0, 1)
    
    # Binary target
    approved = np.where(final_probs >= 0.5, 1, 0)
    
    df = pd.DataFrame({
        'CODE_GENDER': gender,
        'FLAG_OWN_CAR': own_car,
        'FLAG_OWN_REALTY': own_realty,
        'CNT_CHILDREN': children_count,
        'AMT_INCOME_TOTAL': annual_income,
        'NAME_INCOME_TYPE': income_type,
        'NAME_EDUCATION_TYPE': education_type,
        'NAME_FAMILY_STATUS': family_status,
        'NAME_HOUSING_TYPE': housing_type,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'FLAG_MOBIL': 1, # Everyone has a mobile in modern datasets
        'FLAG_WORK_PHONE': work_phone,
        'FLAG_PHONE': phone,
        'FLAG_EMAIL': email,
        'OCCUPATION_TYPE': occupation_type,
        'CNT_FAM_MEMBERS': family_size,
        'APPROVED': approved
    })
    
    return df

if __name__ == '__main__':
    # Save a training and evaluation dataset
    os.makedirs('data', exist_ok=True)
    df = generate_synthetic_data(num_records=10000)
    df.to_csv('data/credit_record.csv', index=False)
    print(f"Generated synthetic data: {df.shape[0]} records, target approval rate: {df['APPROVED'].mean():.2%}")

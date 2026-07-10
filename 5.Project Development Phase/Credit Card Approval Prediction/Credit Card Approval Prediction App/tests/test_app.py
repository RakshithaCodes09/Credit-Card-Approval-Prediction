import unittest
import os
import tempfile
import pandas as pd
from app import app
from src.data_generator import generate_synthetic_data

class CreditAppTestCase(unittest.TestCase):
    def setUp(self):
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False # Disable CSRF for easy form posting in unit tests
        app.config['SECRET_KEY'] = 'test_secret_key'
        self.client = app.test_client()
        
    def test_homepage_loads(self):
        """Test that the homepage loads successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Credit Card Approval Prediction', response.data)
        
    def test_dashboard_loads(self):
        """Test that the dashboard loads successfully with model metrics."""
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Model Performance Dashboard', response.data)
        self.assertIn(b'Model Accuracy', response.data)
        
    def test_history_loads(self):
        """Test that prediction history page loads."""
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction History', response.data)

    def test_prediction_post(self):
        """Test sending valid data through prediction form."""
        form_payload = {
            'gender': 'F',
            'own_car': 'Y',
            'own_realty': 'Y',
            'children_count': '1',
            'annual_income': '150000.0',
            'income_type': 'Working',
            'education_type': 'Higher education',
            'family_status': 'Married',
            'housing_type': 'House / apartment',
            'age': '35',
            'years_employed': '5.5',
            'work_phone': '0',
            'phone': '1',
            'email': '1',
            'occupation_type': 'Managers',
            'family_size': '3'
        }
        response = self.client.post('/', data=form_payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Decision Output', response.data)
        self.assertIn(b'Approval Probability', response.data)
        
    def test_invalid_prediction_post(self):
        """Test validation rules on prediction form."""
        form_payload = {
            'gender': 'F',
            'own_car': 'Y',
            'own_realty': 'Y',
            'children_count': '5',
            'annual_income': '150000.0',
            'income_type': 'Working',
            'education_type': 'Higher education',
            'family_status': 'Married',
            'housing_type': 'House / apartment',
            'age': '20',
            'years_employed': '10', # Invalid: years employed cannot be greater than age - 15 (20 - 15 = 5)
            'work_phone': '0',
            'phone': '1',
            'email': '1',
            'occupation_type': 'Managers',
            'family_size': '2' # Invalid: family size must be at least children_count + 1 (5 + 1 = 6)
        }
        response = self.client.post('/', data=form_payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        # Verify validation errors are present on the returned page
        self.assertIn(b'Family size must be at least 6', response.data)
        self.assertIn(b'Years employed cannot exceed Age minus 15', response.data)

    def test_batch_prediction_csv(self):
        """Test uploading a CSV file for batch predictions."""
        # Generate some synthetic data to write to a temp file
        df = generate_synthetic_data(num_records=10)
        df_to_upload = df.drop(columns=['APPROVED'])
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_csv:
            df_to_upload.to_csv(temp_csv.name, index=False)
            temp_csv_name = temp_csv.name
            
        try:
            with open(temp_csv_name, 'rb') as f:
                data = {
                    'file': (f, 'test_batch.csv')
                }
                response = self.client.post('/batch', data=data, content_type='multipart/form-data', follow_redirects=True)
                
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Batch Processing Complete', response.data)
            self.assertIn(b'Download Prediction Results', response.data)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_csv_name):
                os.remove(temp_csv_name)

if __name__ == '__main__':
    unittest.main()

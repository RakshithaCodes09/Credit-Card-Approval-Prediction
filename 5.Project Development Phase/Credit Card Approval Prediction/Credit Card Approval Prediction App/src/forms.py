from flask_wtf import FlaskForm
from wtforms import SelectField, IntegerField, DecimalField, BooleanField, SubmitField
from wtforms.validators import DataRequired, NumberRange, ValidationError

class CreditCardForm(FlaskForm):
    gender = SelectField('Gender', choices=[('F', 'Female'), ('M', 'Male')], validators=[DataRequired()])
    own_car = SelectField('Own a Car?', choices=[('N', 'No'), ('Y', 'Yes')], validators=[DataRequired()])
    own_realty = SelectField('Own Property?', choices=[('Y', 'Yes'), ('N', 'No')], validators=[DataRequired()])
    
    children_count = IntegerField('Number of Children', default=0, 
                                  validators=[NumberRange(min=0, max=15, message="Children must be between 0 and 15")])
    
    annual_income = DecimalField('Total Annual Income (USD)', validators=[
        DataRequired(message="Annual income is required"),
        NumberRange(min=5000, max=5000000, message="Income must be between $5,000 and $5,000,000")
    ])
    
    income_type = SelectField('Income Type', choices=[
        ('Working', 'Working'),
        ('Commercial associate', 'Commercial associate'),
        ('Pensioner', 'Pensioner / Retired'),
        ('State servant', 'State servant / Government'),
        ('Student', 'Student')
    ], validators=[DataRequired()])
    
    education_type = SelectField('Education Level', choices=[
        ('Secondary / secondary special', 'Secondary / Special Secondary'),
        ('Higher education', 'Higher Education / Degree'),
        ('Incomplete higher', 'Incomplete Higher'),
        ('Lower secondary', 'Lower Secondary'),
        ('Academic degree', 'Academic Degree (PhD/Postgrad)')
    ], validators=[DataRequired()])
    
    family_status = SelectField('Family Status', choices=[
        ('Married', 'Married'),
        ('Single / not married', 'Single / Not Married'),
        ('Civil marriage', 'Civil marriage'),
        ('Separated', 'Separated'),
        ('Widow', 'Widow')
    ], validators=[DataRequired()])
    
    housing_type = SelectField('Housing Type', choices=[
        ('House / apartment', 'House / Apartment'),
        ('With parents', 'With Parents'),
        ('Rented apartment', 'Rented Apartment'),
        ('Municipal apartment', 'Municipal Apartment'),
        ('Co-op apartment', 'Co-op Apartment'),
        ('Office apartment', 'Office Apartment')
    ], validators=[DataRequired()])
    
    age = IntegerField('Age (Years)', validators=[
        DataRequired(),
        NumberRange(min=18, max=100, message="Age must be between 18 and 100")
    ])
    
    years_employed = DecimalField('Years Employed', default=0.0, validators=[
        NumberRange(min=0.0, max=60.0, message="Years employed must be between 0 and 60")
    ])
    
    work_phone = SelectField('Has Work Phone?', choices=[('0', 'No'), ('1', 'Yes')], validators=[DataRequired()])
    phone = SelectField('Has Personal Phone?', choices=[('0', 'No'), ('1', 'Yes')], validators=[DataRequired()])
    email = SelectField('Has Email?', choices=[('0', 'No'), ('1', 'Yes')], validators=[DataRequired()])
    
    occupation_type = SelectField('Occupation Type', choices=[
        ('Retired', 'Retired / Not Working'),
        ('Laborers', 'Laborers'),
        ('Core staff', 'Core staff'),
        ('Sales staff', 'Sales staff'),
        ('Managers', 'Managers'),
        ('Drivers', 'Drivers'),
        ('High skill tech staff', 'High skill tech staff'),
        ('Accountants', 'Accountants'),
        ('Medicine staff', 'Medicine staff'),
        ('Security staff', 'Security staff'),
        ('Cooking staff', 'Cooking staff'),
        ('Cleaning staff', 'Cleaning staff'),
        ('Private service staff', 'Private service staff'),
        ('Low-skill Laborers', 'Low-skill Laborers'),
        ('Waiters/barmen staff', 'Waiters/barmen staff'),
        ('Secretaries', 'Secretaries'),
        ('HR staff', 'HR staff'),
        ('Realty agents', 'Realty agents'),
        ('IT staff', 'IT staff')
    ], validators=[DataRequired()])
    
    family_size = IntegerField('Family Size', default=1, validators=[
        DataRequired(),
        NumberRange(min=1, max=20, message="Family size must be between 1 and 20")
    ])
    
    submit = SubmitField('Predict')
    
    def validate_family_size(self, field):
        # family size must be at least children_count + 1
        children = self.children_count.data if self.children_count.data is not None else 0
        if field.data < children + 1:
            raise ValidationError(f"Family size must be at least {children + 1} (Children + Applicant)")
            
    def validate_years_employed(self, field):
        # years employed cannot be greater than age - 15 (assume legal working age begins at 15)
        if self.age.data and field.data:
            if field.data > (self.age.data - 15):
                raise ValidationError("Years employed cannot exceed Age minus 15")

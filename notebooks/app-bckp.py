import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template, redirect, url_for
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = 'loan_default_xgboost_model.pkl'

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Define the feature names expected by the model
expected_features = ['AGE', 'CREDIT_SCORE', 'NO_DEFAULT_LOAN', 'NET_INCOME', 
                     'PRINCIPAL_DISBURSED', 'EMI', 'GENDER', 'MARITAL_STATUS', 'PRODUCT']

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    try:
        # Check if the request contains JSON data
        if request.is_json:
            data = request.get_json()
            logger.info(f"Received JSON data: {data}")
        else:
            # Process form data
            data = {key: request.form.get(key) for key in request.form}
            logger.info(f"Received form data: {data}")
        
        # Convert data to DataFrame
        input_df = prepare_input_data(data)
        
        # Make prediction
        prediction_result = make_prediction(input_df)
        
        # Return result based on request type
        if request.is_json:
            return jsonify(prediction_result)
        else:
            return render_template('result.html', prediction=prediction_result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        error_message = str(e)
        if request.is_json:
            return jsonify({'error': error_message}), 400
        else:
            return render_template('error.html', error=error_message), 400

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch predictions"""
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the uploaded CSV file
        df = pd.read_csv(file)
        logger.info(f"Batch prediction request with {len(df)} records")
        
        # Process each row
        results = []
        for _, row in df.iterrows():
            input_df = prepare_input_data(row.to_dict())
            prediction = make_prediction(input_df)
            results.append(prediction)
        
        # Return batch results
        return jsonify({'predictions': results})
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

def prepare_input_data(data):
    """Prepare input data for prediction"""
    # Create a DataFrame with the expected structure
    input_data = {}
    
    # Process numerical features
    numerical_features = ['AGE', 'CREDIT_SCORE', 'NO_DEFAULT_LOAN', 'NET_INCOME', 
                          'PRINCIPAL_DISBURSED', 'EMI']
    for feature in numerical_features:
        if feature in data:
            try:
                input_data[feature] = float(data[feature])
            except ValueError:
                raise ValueError(f"Invalid value for {feature}: {data[feature]}")
        else:
            raise ValueError(f"Missing required feature: {feature}")
    
    # Process categorical features
    categorical_features = ['GENDER', 'MARITAL_STATUS', 'PRODUCT']
    for feature in categorical_features:
        if feature in data:
            input_data[feature] = str(data[feature])
        else:
            raise ValueError(f"Missing required feature: {feature}")
    
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Create dummy variables for categorical features - this matches model training
    # Process each categorical feature separately to ensure proper column naming
    df_processed = df[numerical_features].copy()
    
    # Handle GENDER (create GENDER_MALE)
    if df['GENDER'].iloc[0].upper() == 'MALE':
        df_processed['GENDER_MALE'] = 1
    else:
        df_processed['GENDER_MALE'] = 0
    
    # Handle MARITAL_STATUS (create MARITAL_STATUS_X for each category)
    marital_categories = ['ENGAGED', 'MARRIED', 'OTHER', 'PARTNER', 'SINGLE', 'WIDOWED']
    marital_status = df['MARITAL_STATUS'].iloc[0].upper()
    for category in marital_categories:
        col_name = f'MARITAL_STATUS_{category}'
        df_processed[col_name] = 1 if marital_status == category else 0
    
    # Handle PRODUCT (create PRODUCT_X for each product type)
    product_types = [
        'AFFORDABLE HOUSING', 'AGRIBUSINESS SCHEME LOAN PRODUCT', 'ASSET FINANCE LOAN', 
        'AUTO EQUITY LOAN', 'BUY AND BUILD LOANS', 'CASH COVERED LOAN', 
        'CASH COVERED LOANS', 'CASH COVERED PERSONAL LOANS', 'CBA STAFF CAR LOAN', 
        'CBA STAFF EQUITY RELEASE', 'CBA STAFF MORTGAGE LOAN', 'CBA STAFF SHAMBA LOAN', 
        'COMMERCIAL VEHICLES', 'COMPANY IPF', 'CONSTRUCTION LOAN - CONSUMER', 
        'CONSTRUCTION LOANS', 'CONSTRUTION FINANCE', 'CONSUMER SECURED LOAN', 
        'CONSUMER UNSECURED LOAN', 'CONTRACT FINANCING', 'CONTRACTOR EQUIPMENT', 
        'CORPORATE TERM LOAN', 'DIGITAL PERSONAL LOAN', 'DISTRIBUTOR FINANCE', 
        'DMB DISTRIBUTOR FINANCE', 'DMB LOAN - ASSET FINANCE', 'DMB LOAN - ASSET FINANCE HP', 
        'DMB LOAN - COMMERCIAL', 'DMB LOAN - INSUR. PREM. FINANCE', 'DMB LOAN - MORTGAGE', 
        'DMB LOAN - PERSONAL', 'EDUCATION LOANS', 'EMERGENCY LOAN', 'EQUITY FINANCE', 
        'EQUITY RELEASE', 'EX NCBA MOTOR VEHICLE LOAN', 'EX NCBA STAFF MORTGAGE', 
        'EX NCBA STAFF PERSONAL LOAN', 'EX NCBA STAFF SHAMBA LOAN', 'INDIVIDUAL IPF', 
        'LOAN - ASSET FINANCE', 'LOAN - COMMERCIAL', 'LOAN - INSURANCE PREMIUM FINANCE', 
        'LOAN - MORTGAGE', 'LOAN - PERSONAL', 'LOAN - STOCK FINANCE', 'MARKET HOUSING - AHP', 
        'MOBILE LOAN', 'MORTGAGE BUY OUT LOANS', 'MORTGAGE LOAN', 'MOTOR VEHICLE LOAN', 
        'NCBA EASYBUILD (DESIGN AND BUILD)', 'PB BUY AND BUILD', 'PERSONAL SECURED LOANS', 
        'PERSONAL UNSECURED NON SCHEME LOAN', 'PERSONAL UNSECURED SCHEME LOAN', 
        'PLOT PURCHASE', 'PLOT PURCHASE LOAN', 'PLOT PURCHASE LOANS', 
        'PREMIUM FINANCE - CONSUMER', 'PROPERTY FINANCE', 'PROPERTY PURCHASE LOANS', 
        'PSV BUSES', 'PSV MATATU', 'PSV VEHICLES- TAXIS CAR HIRE', 'SALARY ADVANCE LOAN', 
        'SALOON CARS', 'SCHOOL BUSES', 'SECURED BUSINESS LOANS', 'SPECIALIZED EQUIPMENT', 
        'STAFF SECURED LOAN', 'STAFF UNSECURED LOAN', 'STOCK LOANS', 
        'TRACTORS AND RELATED IMPLEMENTS', 'TRAILERS', 'UNSECURED BUSINESS LOAN'
    ]
    
    product = df['PRODUCT'].iloc[0].upper()
    for prod_type in product_types:
        col_name = f'PRODUCT_{prod_type}'
        df_processed[col_name] = 1 if product == prod_type else 0
    
    logger.info(f"Processed input shape: {df_processed.shape}")
    return df_processed

def make_prediction(input_df):
    """Make prediction with the loaded model"""
    if model is None:
        raise ValueError("Model not loaded correctly")
    
    try:
        # Get prediction probability
        prob = model.predict_proba(input_df)[0, 1]
        # Get binary prediction (1 = DEFAULT, 0 = NO DEFAULT)
        prediction = model.predict(input_df)[0]
        
        # Calculate risk level
        if prob < 0.3:
            risk_level = "Low Risk"
        elif prob < 0.7:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        # Extract some key features for display
        credit_score = int(input_df['CREDIT_SCORE'].iloc[0]) if 'CREDIT_SCORE' in input_df else 'N/A'
        
        # Calculate income to EMI ratio
        income = float(input_df['NET_INCOME'].iloc[0]) if 'NET_INCOME' in input_df else 0
        emi = float(input_df['EMI'].iloc[0]) if 'EMI' in input_df else 1  # Avoid division by zero
        income_to_emi_ratio = round(income / emi, 2) if emi > 0 else 'N/A'
        
        # Get loan amount
        loan_amount = int(input_df['PRINCIPAL_DISBURSED'].iloc[0]) if 'PRINCIPAL_DISBURSED' in input_df else 'N/A'
        
        # Get current date for the report
        from datetime import datetime
        analysis_date = datetime.now().strftime("%B %d, %Y")
        
        return {
            'default_probability': float(prob),
            'default_prediction': int(prediction),
            'risk_level': risk_level,
            'credit_score': credit_score,
            'income_to_emi_ratio': income_to_emi_ratio,
            'loan_amount': loan_amount,
            'analysis_date': analysis_date
        }
    
    except Exception as e:
        logger.error(f"Prediction calculation error: {str(e)}")
        raise ValueError(f"Error making prediction: {str(e)}")

@app.route('/health')
def health():
    """Health check endpoint"""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({'status': 'healthy', 'model': MODEL_PATH})

@app.route('/documentation')
def documentation():
    """Display model documentation"""
    try:
        with open('loan_default_model_documentation.md', 'r') as f:
            doc_content = f.read()
        return render_template('documentation.html', content=doc_content)
    except Exception as e:
        logger.error(f"Error reading documentation: {str(e)}")
        return render_template('error.html', error="Documentation not available"), 404

@app.route('/business-impact')
def business_impact():
    """Display business impact assessment"""
    try:
        with open('business_impact_assessment.txt', 'r') as f:
            impact_content = f.read()
        return render_template('business_impact.html', content=impact_content)
    except Exception as e:
        logger.error(f"Error reading business impact: {str(e)}")
        return render_template('error.html', error="Business impact assessment not available"), 404

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create basic HTML templates if they don't exist
    templates = {
        'index.html': '''
<!DOCTYPE html>
<html>
<head>
    <title>Loan Default Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 50px; }
        .form-container { max-width: 800px; margin: 0 auto; }
    </style>
</head>
<body>
    <div class="container">
        <div class="form-container">
            <h1 class="text-center mb-4">Loan Default Risk Prediction</h1>
            <div class="card">
                <div class="card-header">
                    <ul class="nav nav-tabs card-header-tabs">
                        <li class="nav-item">
                            <a class="nav-link active" id="single-tab" data-bs-toggle="tab" href="#single">Single Prediction</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" id="batch-tab" data-bs-toggle="tab" href="#batch">Batch Prediction</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/documentation">Documentation</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/business-impact">Business Impact</a>
                        </li>
                    </ul>
                </div>
                <div class="card-body">
                    <div class="tab-content">
                        <div class="tab-pane fade show active" id="single">
                            <form action="/predict" method="post">
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="AGE" class="form-label">Age</label>
                                        <input type="number" class="form-control" id="AGE" name="AGE" required>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="CREDIT_SCORE" class="form-label">Credit Score</label>
                                        <input type="number" class="form-control" id="CREDIT_SCORE" name="CREDIT_SCORE" required>
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="NO_DEFAULT_LOAN" class="form-label">Number of Default-Free Loans</label>
                                        <input type="number" class="form-control" id="NO_DEFAULT_LOAN" name="NO_DEFAULT_LOAN" required>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="NET_INCOME" class="form-label">Net Income</label>
                                        <input type="number" class="form-control" id="NET_INCOME" name="NET_INCOME" required>
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="PRINCIPAL_DISBURSED" class="form-label">Principal Amount</label>
                                        <input type="number" class="form-control" id="PRINCIPAL_DISBURSED" name="PRINCIPAL_DISBURSED" required>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="EMI" class="form-label">EMI</label>
                                        <input type="number" class="form-control" id="EMI" name="EMI" required>
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-4 mb-3">
                                        <label for="GENDER" class="form-label">Gender</label>
                                        <select class="form-select" id="GENDER" name="GENDER" required>
                                            <option value="MALE">Male</option>
                                            <option value="FEMALE">Female</option>
                                        </select>
                                    </div>
                                    <div class="col-md-4 mb-3">
                                        <label for="MARITAL_STATUS" class="form-label">Marital Status</label>
                                        <select class="form-select" id="MARITAL_STATUS" name="MARITAL_STATUS" required>
                                            <option value="SINGLE">Single</option>
                                            <option value="MARRIED">Married</option>
                                            <option value="OTHER">Other</option>
                                        </select>
                                    </div>
                                    <div class="col-md-4 mb-3">
                                        <label for="PRODUCT" class="form-label">Product</label>
                                        <select class="form-select" id="PRODUCT" name="PRODUCT" required>
                                            <option value="PERSONAL UNSECURED SCHEME LOAN">Personal Unsecured Scheme Loan</option>
                                            <option value="INDIVIDUAL IPF">Individual IPF</option>
                                            <option value="MOBILE LOAN">Mobile Loan</option>
                                            <option value="COMMERCIAL VEHICLES">Commercial Vehicles</option>
                                            <option value="DIGITAL PERSONAL LOAN">Digital Personal Loan</option>
                                        </select>
                                    </div>
                                </div>
                                <div class="text-center mt-3">
                                    <button type="submit" class="btn btn-primary">Predict Default Risk</button>
                                </div>
                            </form>
                        </div>
                        <div class="tab-pane fade" id="batch">
                            <form action="/batch-predict" method="post" enctype="multipart/form-data">
                                <div class="mb-3">
                                    <label for="file" class="form-label">Upload CSV File</label>
                                    <input type="file" class="form-control" id="file" name="file" accept=".csv" required>
                                    <div class="form-text">Upload a CSV file with all required features.</div>
                                </div>
                                <div class="text-center mt-3">
                                    <button type="submit" class="btn btn-primary">Run Batch Prediction</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
        ''',
        'result.html': '''
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 50px; }
        .result-container { max-width: 600px; margin: 0 auto; }
    </style>
</head>
<body>
    <div class="container">
        <div class="result-container">
            <h1 class="text-center mb-4">Loan Default Prediction Result</h1>
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Prediction Summary</h5>
                    
                    <div class="mb-4">
                        <div class="progress" style="height: 30px;">
                            <div class="progress-bar 
                                {% if prediction.default_probability < 0.3 %}bg-success
                                {% elif prediction.default_probability < 0.7 %}bg-warning
                                {% else %}bg-danger{% endif %}" 
                                role="progressbar" 
                                style="width: {{ prediction.default_probability * 100 }}%"
                                aria-valuenow="{{ prediction.default_probability * 100 }}" 
                                aria-valuemin="0" 
                                aria-valuemax="100">
                                {{ "%.1f"|format(prediction.default_probability * 100) }}%
                            </div>
                        </div>
                    </div>
                    
                    <div class="alert 
                        {% if prediction.default_probability < 0.3 %}alert-success
                        {% elif prediction.default_probability < 0.7 %}alert-warning
                        {% else %}alert-danger{% endif %}">
                        <h4 class="alert-heading">{{ prediction.risk_level }}</h4>
                        <p>
                            {% if prediction.default_prediction == 1 %}
                                This application is predicted to <strong>DEFAULT</strong> on the loan.
                            {% else %}
                                This application is predicted to <strong>NOT DEFAULT</strong> on the loan.
                            {% endif %}
                        </p>
                        <hr>
                        <p class="mb-0">Default Probability: {{ "%.2f"|format(prediction.default_probability * 100) }}%</p>
                    </div>
                    
                    <div class="text-center mt-4">
                        <a href="/" class="btn btn-primary">Make Another Prediction</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
        ''',
        'error.html': '''
<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 50px; }
        .error-container { max-width: 600px; margin: 0 auto; }
    </style>
</head>
<body>
    <div class="container">
        <div class="error-container">
            <div class="card border-danger">
                <div class="card-header bg-danger text-white">
                    <h4 class="mb-0">Error</h4>
                </div>
                <div class="card-body">
                    <p class="card-text">{{ error }}</p>
                    <div class="text-center mt-3">
                        <a href="/" class="btn btn-primary">Return to Home</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
        ''',
        'documentation.html': '''
<!DOCTYPE html>
<html>
<head>
    <title>Model Documentation</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.1.0/github-markdown.min.css">
    <style>
        body { padding-top: 20px; padding-bottom: 20px; }
        .documentation-container { max-width: 900px; margin: 0 auto; }
        .markdown-body { padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="documentation-container">
            <div class="card">
                <div class="card-header">
                    <ul class="nav nav-tabs card-header-tabs">
                        <li class="nav-item">
                            <a class="nav-link" href="/">Home</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link active" href="/documentation">Documentation</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/business-impact">Business Impact</a>
                        </li>
                    </ul>
                </div>
                <div class="card-body markdown-body">
                    {{ content | safe }}
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.0.2/marked.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('.markdown-body').forEach(function(el) {
                el.innerHTML = marked.parse(el.textContent);
            });
        });
    </script>
</body>
</html>
        ''',
        'business_impact.html': '''
<!DOCTYPE html>
<html>
<head>
    <title>Business Impact Assessment</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 20px; padding-bottom: 20px; }
        .impact-container { max-width: 900px; margin: 0 auto; }
        pre { white-space: pre-wrap; background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="impact-container">
            <div class="card">
                <div class="card-header">
                    <ul class="nav nav-tabs card-header-tabs">
                        <li class="nav-item">
                            <a class="nav-link" href="/">Home</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/documentation">Documentation</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link active" href="/business-impact">Business Impact</a>
                        </li>
                    </ul>
                </div>
                <div class="card-body">
                    <h4 class="card-title">Business Impact Assessment</h4>
                    <pre>{{ content }}</pre>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
        '''
    }
    
    for filename, content in templates.items():
        filepath = os.path.join('templates', filename)
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write(content)
            logger.info(f"Created template: {filename}")
    
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)
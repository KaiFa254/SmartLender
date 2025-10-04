from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import shap
import numpy as np
import os
import logging


app = Flask(__name__)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load the pre-trained pipeline
pipeline = joblib.load('loan_default_pipeline.pkl')

# pipeline = joblib.load('notebooks/loan_default_pipeline.pkl')

# For SHAP: Load a sample of training data or create background data
# Option A: Load saved sample
try:
    X_train_sample = pd.read_csv('X_train_sample.csv')  # Save this file separately
    X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train_sample)
    explainer = shap.TreeExplainer(pipeline.named_steps['model'], X_train_transformed)
except:
    explainer = None  # Handle case where sample data isn't available

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Use UPPERCASE column names to match training data 
        data = {
            'AGE': int(request.form.get('age')),
            'CREDIT_SCORE': int(request.form.get('credit_score')),
            'NO_DEFAULT_LOAN': int(request.form.get('no_default_loan')),
            'NET_INCOME': float(request.form.get('net_income')),
            'PRINCIPAL_DISBURSED': float(request.form.get('principal_disbursed')),
            'EMI': float(request.form.get('emi')),
            'GENDER': request.form.get('gender'),
            'MARITAL_STATUS': request.form.get('marital_status'),
            'PRODUCT': request.form.get('product')
        }

        input_df = pd.DataFrame([data])

        # Predict
        default_prob = pipeline.predict_proba(input_df)[:, 1][0]
        default_prediction = pipeline.predict(input_df)[0]

        # Compute SHAP values (if explainer is available)
        contributions = {}
        if explainer is not None:
            input_transformed = pipeline.named_steps['preprocessor'].transform(input_df)
            shap_values = explainer.shap_values(input_transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            feature_importance = np.abs(shap_values[0])
            feature_percent = 100 * feature_importance / np.sum(feature_importance)
            
            feature_names = input_df.columns
            contributions = {feature_names[i]: f"{feature_percent[i]:.2f}%" 
                           for i in range(len(feature_names))}
        
        result = {
            'prediction': int(default_prediction),
            'probability': float(default_prob),
            'status': 'Default Risk' if default_prediction == 1 else 'No Default Risk',
            'risk_percentage': f"{default_prob * 100:.2f}%",
            'feature_contributions': contributions
        }

        return render_template('result.html', result=result, data=data)

    except Exception as e:
        return jsonify({'error': str(e)})


# business impact assessment route
@app.route('/documentation')
def documentation():
    try:
        # Try to load the markdown documentation file
        doc_path = 'loan_default_model_documentation.md'
        if os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = "# Documentation\n\nDocumentation file not found."
        
        return render_template('documentation.html', content=content)
    except Exception as e:
        logger.error(f"Error loading documentation: {e}")
        return render_template('error.html', error=str(e))

@app.route('/business-impact')
def business_impact():
    try:
        # Load the business impact assessment file
        impact_path = 'business_impact_assessment.txt'  # Note: corrected filename
        if os.path.exists(impact_path):
            with open(impact_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = "Business impact assessment file not found."
        
        return render_template('business_impact.html', content=content)
    except Exception as e:
        logger.error(f"Error loading business impact: {e}")
        return render_template('error.html', error=str(e))

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Make predictions
        predictions = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)[:, 1]
        
        # Add predictions to dataframe
        df['Prediction'] = predictions
        df['Default_Probability'] = probabilities
        df['Risk_Level'] = df['Default_Probability'].apply(
            lambda x: 'Low Risk' if x < 0.3 else ('Medium Risk' if x < 0.7 else 'High Risk')
        )
        
        # Convert to HTML table
        table_html = df.to_html(classes='table table-striped table-hover', index=False)
        
        return render_template('batch_result.html', table=table_html)
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return render_template('error.html', error=str(e))
    


if __name__ == '__main__':
    app.run(debug=True)

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
                                        <label for="NET INCOME" class="form-label">Net Income</label>
                                        <input type="number" class="form-control" id="NET INCOME" name="NET INCOME" required>
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
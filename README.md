# Loan Default Prediction

This project implements a machine learning pipeline for predicting loan defaults. It utilizes a pre-trained XGBoost model to classify loan applicants based on risk. The project includes data preprocessing, model inference, and deployment steps.

## Features
- Pre-trained XGBoost model for classification.
- Data preprocessing pipeline.
- API integration for model inference.
- Deployment using Flask and Render.

---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [API Endpoints](#api-endpoints)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- Virtual environment tool (optional but recommended)

### Setup
Clone the repository and navigate to the project folder:

```sh
$ git clone https://github.com/yourusername/loan-default-prediction.git
$ cd loan-default-prediction
```

Create a virtual environment and install dependencies:

```sh
$ python -m venv venv
$ source venv/bin/activate  # On Windows use `venv\Scripts\activate`
$ pip install -r requirements.txt
```

---

## Usage

### Running the API Server
The project uses Flask to serve the model as an API. To start the server:

```sh
$ python app.py
```

The API will be available at `http://127.0.0.1:5000/`.

### Making Predictions
Send a `POST` request to the `/predict` endpoint with JSON data containing loan applicant details.

Example request using cURL:

```sh
curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{
  "feature1": 10,
  "feature2": 5.5,
  "feature3": "some_category"
}'
```

Example response:
```json
{
  "prediction": "default"
}
```

---

## Model Details
The pre-trained model (`loan_default_xgboost_model.pkl`) is loaded for inference. The pipeline (`loan_default_pipeline.pkl`) ensures preprocessing consistency.

**Libraries Used:**
- `scikit-learn`
- `xgboost`
- `pandas`
- `Flask`

---

## API Endpoints

| Endpoint   | Method | Description |
|------------|--------|-------------|
| `/predict` | POST   | Predicts if a loan will default based on input features. |
| `/health`  | GET    | Returns server health status. |

---

## Deployment

### Local Deployment
Run the Flask application locally:

```sh
$ python app.py
```

### Deployment on Render
1. Create a new service on [Render](https://render.com/).
2. Select `Python` as the environment.
3. Connect your GitHub repository.
4. Define a `start command`:  
   ```sh
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```
5. Set up environment variables if needed.
6. Deploy the service and note the live API URL.

---

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.


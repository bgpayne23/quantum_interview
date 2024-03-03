from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler_newer.pkl')

application = Flask(__name__)  # Elastic Beanstalk expects an 'application' callable

@application.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Extract and scale the features from the JSON
    features = np.array([data['Age'], data['SystolicBP'], data['DiastolicBP'], data['BS'], data['BodyTemp'], data['HeartRate']]).reshape(1, -1)
    scaled_features = scaler.transform(features)
    
    # Make prediction and calculate likelihood score
    prediction = model.predict(scaled_features)
    likelihood = model.predict_proba(scaled_features)
    
    # Convert prediction to risk level
    risk_levels = ['high risk', 'low risk', 'mid risk']  # Adjust based on your label encoder
    response = {
        'Prediction': risk_levels[prediction[0]],
        'Likelihood_Score': likelihood.tolist()[0]
    }
    return jsonify(response)

if __name__ == '__main__':
    application.run(debug=True)
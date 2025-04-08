from flask import Flask, request, jsonify,render_template
import joblib
import numpy as np
import pandas as pd
from scipy.stats import boxcox
# Load the trained model
model = joblib.load('./model/best_svm_model.pkl')  # Replace with your actual saved model path

# Initialize Flask app
app = Flask(__name__)

# Dictionary to store the lambda values from training
boxcox_lambdas = {
    'age': 1.1884623632766138,
    'trestbps': -0.566961807810652,
    'chol': -0.12552639286383113,
    'thalach': 2.445456150508822,
    'oldpeak': 0.17759774670247044
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        input_data = request.json

        # Required feature names (raw input)
        required_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]

        # Check if all required features are present
        for feature in required_features:
            if feature not in input_data:
                return jsonify({"error": f"Missing required feature: {feature}"}), 400

        # Create a DataFrame from input JSON
        data = {feature: [input_data[feature]] for feature in required_features}
        input_df = pd.DataFrame(data)

        # Add small constant to 'oldpeak' to ensure all values are positive
        if 'oldpeak' in input_df.columns:
            input_df['oldpeak'] = input_df['oldpeak'] + 0.001

        # Apply Box-Cox transformation to continuous features
        continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        for feature in continuous_features:
            if feature in input_df.columns and feature in boxcox_lambdas:
                lambda_value = boxcox_lambdas[feature]
                # Ensure the values are positive (Box-Cox requirement)
                if input_df[feature].min() > 0:
                    input_df[feature] = boxcox(input_df[feature], lmbda=lambda_value)

        # Define expected columns for one-hot encoding
        categorical_columns = {
            'cp': ['cp_1', 'cp_2', 'cp_3'],
            'restecg': ['restecg_1', 'restecg_2'],
            'thal': ['thal_1', 'thal_2', 'thal_3']
        }

        # Perform one-hot encoding manually to ensure all expected columns are created
        for col, expected_encoded_cols in categorical_columns.items():
            # Initialize all expected one-hot columns with 0
            for encoded_col in expected_encoded_cols:
                input_df[encoded_col] = 0

            # Set the correct column to 1 based on the input value
            if col in input_data:
                col_value = input_data[col]
                if col_value > 0:  # Avoid setting invalid indices
                    encoded_col_name = f"{col}_{col_value}"
                    if encoded_col_name in expected_encoded_cols:
                        input_df[encoded_col_name] = 1

        # Drop the original categorical columns
        input_df.drop(columns=['cp', 'restecg', 'thal'], inplace=True)

        # Add any remaining numeric columns (if missing)
        numeric_features = [
            'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang',
            'oldpeak', 'slope', 'ca'
        ]
        for col in numeric_features:
            if col not in input_df.columns:
                input_df[col] = 0

        # Ensure columns are ordered to match the training data
        expected_columns = [
            'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang',
            'oldpeak', 'slope', 'ca',
            'cp_1', 'cp_2', 'cp_3',
            'restecg_1', 'restecg_2',
            'thal_1', 'thal_2', 'thal_3'
        ]
        input_df = input_df[expected_columns]

        # Ensure the column types match the training data (optional, for safety)
        input_df = input_df.astype(float)

        # Make prediction
        prediction = model.predict(input_df)
        
        answer = True if int(prediction[0]) == 1 else False

        # Return the result as JSON
        return jsonify({"Disease": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)

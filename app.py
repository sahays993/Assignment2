from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model using joblib
model = joblib.load('diabetes_model.pkl')

# Load the scaler using joblib
scaler = joblib.load('scaler.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Scale the input features
        scaled_features = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)[0][1] * 100  # For probability %

        # Return results
        output = f"{probability:.2f}%"
        if prediction == 1:
            result = "High likelihood of diabetes."
        else:
            result = "Low likelihood of diabetes."
        
        return render_template('result.html', prediction_text=result, probability=output)

    except ValueError:
        return "Invalid input. Please enter numeric values."

if __name__ == '__main__':
    app.run(debug=True)

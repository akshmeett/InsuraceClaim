from flask import Flask, render_template, request
import joblib
import numpy as np

# Init Flask app
app = Flask(__name__)

# Load regressor, classifier, scaler
regressor = joblib.load('model_reg.pkl')
classifier = joblib.load('model_cls.pkl')
scaler = joblib.load('scaler_reg.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        sex = 1 if request.form['sex'] == 'male' else 0
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = 1 if request.form['smoker'] == 'yes' else 0
        region = int(request.form['region'])
        claim_amount = float(request.form['claim_amount'])

        bmi_age_interaction = bmi * age

        # Predict expected charge
        features = np.array([[age, sex, bmi, children, smoker, region, bmi_age_interaction]])
        scaled_features = scaler.transform(features)
        expected_charge = np.expm1(regressor.predict(scaled_features)[0])

        # Calculate residual & ratio
        residual = claim_amount - expected_charge
        ratio = claim_amount / (expected_charge + 1)

        # Prepare classifier input
        classifier_features = np.array([[age, sex, bmi, children, smoker, region,
                                         bmi_age_interaction, expected_charge, residual, ratio]])
        anomaly_proba = classifier.predict_proba(classifier_features)[0][1]
        is_anomaly = classifier.predict(classifier_features)[0]

        return render_template(
            'result.html',
            expected=round(expected_charge, 2),
            claim=round(claim_amount, 2),
            residual=round(residual, 2),
            ratio=round(ratio, 2),
            is_anomaly='Yes' if is_anomaly == 1 else 'No',
            prob=round(anomaly_proba * 100, 2)
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

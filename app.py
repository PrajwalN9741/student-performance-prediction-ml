from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import os

app = Flask(__name__)
MODEL_PATH = os.path.join('models', 'student_model.joblib')

# Load model
artifacts = joblib.load(MODEL_PATH)
model = artifacts['model']
scaler = artifacts['scaler']
le = artifacts['label_encoder']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        attendance = float(request.form['attendance'])
        assignment = float(request.form['assignment_score'])
        internal = float(request.form['internal_marks'])
        participation = float(request.form['participation'])

        X = np.array([[attendance, assignment, internal, participation]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        label = le.inverse_transform([pred])[0]

        # Also provide probability distribution
        probs = model.predict_proba(X_scaled)[0]
        classes = le.inverse_transform(np.arange(len(probs)))
        prob_dict = dict(zip(classes, [float(round(p, 3)) for p in probs]))

        return render_template('result.html', label=label, probs=prob_dict)
    except Exception as e:
        return f'Error: {str(e)}', 400

if __name__ == '__main__':
    app.run(debug=True)


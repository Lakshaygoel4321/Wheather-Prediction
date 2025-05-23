from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and preprocessor
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Extract form data
        input_data = {
            'Location': request.form['Location'],
            'MinTemp': float(request.form['MinTemp']),
            'MaxTemp': float(request.form['MaxTemp']),
            'Rainfall': float(request.form['Rainfall']),
            'WindGustDir': request.form['WindGustDir'],
            'WindGustSpeed': float(request.form['WindGustSpeed']),
            'WindDir9am': request.form['WindDir9am'],
            'WindDir3pm': request.form['WindDir3pm'],
            'WindSpeed9am': float(request.form['WindSpeed9am']),
            'WindSpeed3pm': float(request.form['WindSpeed3pm']),
            'Humidity9am': float(request.form['Humidity9am']),
            'Humidity3pm': float(request.form['Humidity3pm']),
            'Pressure9am': float(request.form['Pressure9am']),
            'Pressure3pm': float(request.form['Pressure3pm']),
            'Temp9am': float(request.form['Temp9am']),
            'Temp3pm': float(request.form['Temp3pm']),

            'RainToday': request.form['RainToday']
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Preprocess and predict
        X = preprocessor.transform(df)
        pred = model.predict(X)[0]
        prediction = "Yes" if pred == 1 else "No"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

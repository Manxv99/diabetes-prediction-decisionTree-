from flask import Flask, request, app, render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)
app=application

scaler = pickle.load(open('Models\StandardScaler.pkl', "rb"))
model = pickle.load(open('Models\diabetesPredictionModel.pkl', "rb"))

#Route for Home page
@app.route("/")
def index():
    return render_template('index.html')

#Route for prediction page
@app.route("/predictdata", methods=['GET', 'POST'])
def predict_diabetes():
    if request.method == 'POST':
        preg = int(request.form.get("Pregnancies"))
        gluc = float(request.form.get("Glucose"))
        bp = float(request.form.get("BloodPressure"))
        skinthickness = float(request.form.get("SkinThickness"))
        insulin = float(request.form.get("Insulin"))
        bmi = float(request.form.get("BMI"))
        dpf = float(request.form.get("DiabetesPedigreeFunction"))
        age = float(request.form.get("Age"))

        new_data = scaler.transform([[preg, gluc, bp, skinthickness, insulin, bmi, dpf, age]])
        prediction = model.predict(new_data)

        if prediction[0] == 0:
            result = "Not Diabetic:)"
        elif prediction[0] == 1:
            result = "Diabetic:("

        return render_template("result.html", answer = result)
    
    else:
       return render_template('home.html') 
    
#entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0')
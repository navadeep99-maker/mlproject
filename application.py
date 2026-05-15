from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
<<<<<<< HEAD
from src.pipeline.predict_pipeline import Predict_Pipeline, custom_data     

application = Flask(__name__)
app = application

@app.route("/", methods=['GET', 'POST'])
=======

def predict_datapoint():
    if request.method == 'POST':
        try:
            # Collect form data
            data = custom_data(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )
            print("Form data:", request.form)

            # Convert to dataframe
            pred_df = data.get_data_as_dataframe()

            # Run prediction
            predict_pipeline = Predict_Pipeline()
            results = predict_pipeline.predict(pred_df)
            print("Prediction results:", results)

            # Render home.html with results
            return render_template("home.html", results=results[0])
        except Exception as e:
            # Render home.html with error message
            return render_template("home.html", error=str(e))
    else:
        # GET request → just show the form
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0")

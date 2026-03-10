from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import Predict_Pipeline,custom_data     
application=Flask(__name__)
app=application
@app.route('/')
def index():
    return render_template("index.html")
@app.route("/predictdata",methods=['POST','GET'])
def predict_datapoint():
    if request.method=='POST':
        try:
            data=custom_data(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )
            print("Form data:", request.form)
            pred_df=data.get_data_as_dataframe()
            predict_pipeline=Predict_Pipeline()
            results=predict_pipeline.predict(pred_df)
            print(results)
            return render_template("home.html", results=results[0])
        except Exception as e:
            return render_template("home.html", error=str(e))
    else:
        return render_template("home.html")
if __name__=="__main__":
    app.run(host="0.0.0.0")

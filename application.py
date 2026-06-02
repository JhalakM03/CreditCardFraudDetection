from flask import Flask, render_template, request
import pandas as pd 

from src.pipelines.prediction_pipeline import PredictionPipeline
import sys
print("Python running from:", sys.executable)
app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_form", methods = ["GET","POST"])

def predict_form():

    if request.method=="GET":
        return render_template("form.html")
    
    try:
        data=request.form
        df=pd.DataFrame([data.to_dict()])

        df=df.astype(float)

        pipeline=PredictionPipeline()
        prediction=pipeline.predict(df)

        result= "Not Fraud" if float(prediction[0])==0 else "Fraud"

        return render_template("form.html", result=result)
    except Exception as e:
        raise f"Error: {str(e)}"
    
if __name__ =="__main__":
    app.run(host="'0.0.0.0", port=5000, debug=True)
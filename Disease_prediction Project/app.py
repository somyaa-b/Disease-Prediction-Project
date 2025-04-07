from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open("logistic_regression.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("decision_tree.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    scaled_data = scaler.transform([data]) 

    lr_pred = lr_model.predict(scaled_data)[0]
    dt_pred = dt_model.predict(scaled_data)[0]

    lr_disease = label_encoder.inverse_transform([lr_pred])[0]
    dt_disease = label_encoder.inverse_transform([dt_pred])[0]

    return render_template("index.html", lr_result=lr_disease, dt_result=dt_disease)

if __name__ == "__main__":
    app.run(debug=True)

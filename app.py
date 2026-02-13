import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from flask import Flask, render_template, request # type: ignore
import joblib, cv2, numpy as np # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from utils.feature_extractor import feature_extractor

app = Flask(__name__, template_folder='Template')
model = joblib.load("models/best_model.pkl")

labels = ["Electric Bus 🚌","Electric Car 🚗"]

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(224,224))
    img = preprocess_input(img)
    img = np.expand_dims(img,0)

    feature = feature_extractor.predict(img, verbose=0)
    pred = model.predict(feature)[0]
    return labels[pred]

@app.route("/", methods=["GET","POST"])
def home():
    if request.method=="POST":
        file = request.files["image"]
        path = "static/test.jpg"
        file.save(path)

        result = predict_image(path)
        return render_template("index.html", result=result)

    return render_template("index.html", result="Upload an EV image")

app.run(debug=True)
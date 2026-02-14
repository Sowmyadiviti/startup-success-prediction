from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

# Proper absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")

# Flask app initialization
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Load trained model
model = joblib.load(MODEL_PATH)


# Home route
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        input_feature = [x for x in request.form.values()]
        input_feature = np.array(input_feature, dtype=float).reshape(1, -1)

        # Column names
        names = [
            "age_first_funding_year",
            "age_last_funding_year",
            "age_first_milestone_year",
            "age_last_milestone_year",
            "relationships",
            "funding_rounds",
            "funding_total_usd",
            "milestones",
            "avg_participants",
        ]
        # Create dataframe
        data = pd.DataFrame(input_feature, columns=names)
        # Prediction
        prediction = model.predict(data)[0]
        result = "acquired" if int(prediction) == 1 else "closed"

        return render_template(
            "result.html",
            prediction_text=f"The Startup is: {result}",
        )
    except Exception as e:
        return f"Error occurred: {str(e)}"

# Run app
if __name__ == "__main__":
    app.run(debug=True)

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load files relative to this script, so the app works from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))  # 15 features the model expects

# The scaler was trained on 24 features: its own list is the ground truth for scaling
SCALER_FEATURES = scaler.feature_names_in_.tolist()

LABELS = {
    "NObeyesdad_Normal_Weight": "Normal weight",
    "NObeyesdad_Overweight_Level_I": "Overweight (Level I)",
    "NObeyesdad_Overweight_Level_II": "Overweight (Level II)",
    "NObeyesdad_Obesity_Type_I": "Obesity (Type I)",
    "NObeyesdad_Obesity_Type_II": "Obesity (Type II)",
    "NObeyesdad_Obesity_Type_III": "Obesity (Type III)",
}


@app.route("/")
def home():
    return render_template("index.html")


def read_number(user_input, name, low, high):
    """Read a numeric field and check it is within a sensible range."""
    raw = str(user_input.get(name, "")).strip()
    if raw == "":
        raise ValueError(f"{name} is required.")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number.")
    if not (low <= value <= high):
        raise ValueError(f"{name} must be between {low} and {high}.")
    return value


def preprocess_input(user_input):
    """Maps raw form input to all 24 features the scaler expects."""
    processed = {}

    # 1. Numeric fields
    processed["Age"] = read_number(user_input, "Age", 5, 120)
    processed["Height"] = read_number(user_input, "Height", 0.5, 2.5)  # meters
    processed["Weight"] = read_number(user_input, "Weight", 10, 400)   # kg
    processed["BMI"] = processed["Weight"] / (processed["Height"] ** 2)

    # 2. Ordinal fields (text choice -> number used in training)
    processed["FCVC"] = {"Never": 1, "Sometimes": 2, "Always": 3}.get(user_input.get("FCVC"), 2)
    processed["NCP"] = {"1-2": 1, "3": 3, ">3": 4}.get(user_input.get("NCP"), 3)
    processed["CH2O"] = {"<1L": 1, "1-2L": 2, ">2L": 3}.get(user_input.get("CH2O"), 2)
    processed["FAF"] = {"0": 0, "1-2": 1, "2-4": 2, "4-5": 3}.get(user_input.get("FAF"), 0)
    processed["TUE"] = {"0-2": 0, "3-5": 1, ">5": 2}.get(user_input.get("TUE"), 0)

    # 3. Binary fields
    def yes(name):
        return 1 if str(user_input.get(name, "")).lower() == "yes" else 0

    processed["Gender_Male"] = 1 if user_input.get("Gender") == "Male" else 0
    processed["family_history_with_overweight_yes"] = yes("family_history_with_overweight")
    processed["FAVC_yes"] = yes("FAVC")
    processed["SMOKE_yes"] = yes("SMOKE")
    processed["SCC_yes"] = yes("SCC")

    # 4. One-hot fields ("Always" / "Automobile" are the dropped reference categories)
    for prefix in ("CAEC", "CALC"):
        value = str(user_input.get(prefix, "")).lower()
        for v in ("Frequently", "Sometimes", "no"):
            processed[f"{prefix}_{v}"] = 1 if value == v.lower() else 0

    mtrans = user_input.get("MTRANS")
    for v in ("Bike", "Motorbike", "Public_Transportation", "Walking"):
        processed[f"MTRANS_{v}"] = 1 if mtrans == v else 0

    return processed


@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if not user_input:
        return jsonify({"error": "No data received."}), 400

    try:
        X = preprocess_input(user_input)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Build a row with all 24 scaler features, scale it, keep the 15 the model uses
    df_24 = pd.DataFrame([{f: X.get(f, 0) for f in SCALER_FEATURES}])
    df_scaled = pd.DataFrame(scaler.transform(df_24), columns=SCALER_FEATURES)
    df_model = df_scaled[features]

    pred = str(model.predict(df_model)[0])
    return jsonify({"prediction": LABELS.get(pred, pred)})


if __name__ == "__main__":
    # debug mode only if explicitly requested (it is unsafe on a public server)
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")

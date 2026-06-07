from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained Machine Learning model
model = joblib.load("student_stress_model.pkl")


# ---------------- HOME PAGE ---------------- #

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- ABOUT PAGE ---------------- #

@app.route("/about")
def about():
    return render_template("about.html")


# ---------------- PREDICTION FORM ---------------- #

@app.route("/predict")
def predict():
    return render_template("predict.html")


# ---------------- RESULT PAGE ---------------- #

@app.route("/result", methods=["POST"])
def result():

    # Get values from HTML form

    anxiety_level = int(request.form["anxiety_level"])
    self_esteem = int(request.form["self_esteem"])
    mental_health_history = int(request.form["mental_health_history"])
    depression = int(request.form["depression"])
    headache = int(request.form["headache"])
    blood_pressure = int(request.form["blood_pressure"])
    sleep_quality = int(request.form["sleep_quality"])
    breathing_problem = int(request.form["breathing_problem"])
    noise_level = int(request.form["noise_level"])
    living_conditions = int(request.form["living_conditions"])
    safety = int(request.form["safety"])
    basic_needs = int(request.form["basic_needs"])
    academic_performance = int(request.form["academic_performance"])
    study_load = int(request.form["study_load"])
    teacher_student_relationship = int(request.form["teacher_student_relationship"])
    future_career_concerns = int(request.form["future_career_concerns"])
    social_support = int(request.form["social_support"])
    peer_pressure = int(request.form["peer_pressure"])
    extracurricular_activities = int(request.form["extracurricular_activities"])
    bullying = int(request.form["bullying"])

    # Create DataFrame for prediction

    student = pd.DataFrame({

        "anxiety_level": [anxiety_level],
        "self_esteem": [self_esteem],
        "mental_health_history": [mental_health_history],
        "depression": [depression],
        "headache": [headache],
        "blood_pressure": [blood_pressure],
        "sleep_quality": [sleep_quality],
        "breathing_problem": [breathing_problem],
        "noise_level": [noise_level],
        "living_conditions": [living_conditions],
        "safety": [safety],
        "basic_needs": [basic_needs],
        "academic_performance": [academic_performance],
        "study_load": [study_load],
        "teacher_student_relationship": [teacher_student_relationship],
        "future_career_concerns": [future_career_concerns],
        "social_support": [social_support],
        "peer_pressure": [peer_pressure],
        "extracurricular_activities": [extracurricular_activities],
        "bullying": [bullying]

    })

    # Predict stress level
    prediction = model.predict(student)
    prediction = int(prediction[0])

    probabilities = model.predict_proba(student)[0]

    low_prob = round(probabilities[0] * 100, 2)
    medium_prob = round(probabilities[1] * 100, 2)
    high_prob = round(probabilities[2] * 100, 2)

    # Recommendation

    if prediction == 0:

        status = "Low Stress"

        color = "success"

        message = "Excellent! Keep maintaining your healthy lifestyle."

    elif prediction == 1:

        status = "Medium Stress"

        color = "warning"

        message = "Your stress level is moderate. Maintain a good study-life balance."

    else:

        status = "High Stress"

        color = "danger"

        message = "Your stress level is high. Please seek support and reduce stress."

    return render_template(
    "result.html",
    prediction=prediction,
    status=status,
    color=color,
    message=message,
    low_prob=low_prob,
    medium_prob=medium_prob,
    high_prob=high_prob
)


# ---------------- RUN APP ---------------- #

if __name__ == "__main__":
    app.run(debug=True)


def risk_band(p: float) -> str:
    if p < 0.33:
        return "Low"
    if p < 0.66:
        return "Moderate"
    return "High"


def general_recommendations(band: str):
    recs = [
        "Aim for at least 150 minutes/week of moderate physical activity.",
        "Maintain a balanced diet; limit high-salt and high-saturated-fat foods.",
        "Avoid smoking and prioritise good sleep."
    ]
    if band == "Moderate":
        recs.append("Consider routine check-ups for blood pressure, cholesterol, and glucose.")
    if band == "High":
        recs.append("If you have symptoms or concerning readings, seek medical advice promptly.")
    return recs


def heart_feature_recommendations(user: dict):
    recs = []

    chol = user.get("chol")
    trestbps = user.get("trestbps")
    exang = user.get("exang")
    oldpeak = user.get("oldpeak")
    thalach = user.get("thalach")
    fbs = user.get("fbs")

    if chol is not None and chol >= 240:
        recs.append("Your cholesterol appears elevated; reduce saturated fat intake and consider a lipid check with a clinician.")

    if trestbps is not None and trestbps >= 140:
        recs.append("Your resting blood pressure appears high; reduce salt intake and monitor your blood pressure regularly.")

    if exang == 1:
        recs.append("Exercise-induced angina was reported; avoid overexertion and seek clinical assessment before intense exercise.")

    if oldpeak is not None and oldpeak >= 2.0:
        recs.append("ST depression is elevated; this may warrant further cardiovascular assessment.")

    if thalach is not None and thalach < 120:
        recs.append("Your maximum heart rate is relatively low; discuss exercise tolerance and cardiovascular fitness with a clinician if concerned.")

    if fbs == 1:
        recs.append("Raised fasting blood sugar may indicate glucose regulation issues; consider follow-up testing.")

    return recs


def diabetes_feature_recommendations(user: dict):
    recs = []

    bmi = user.get("bmi")
    bp = user.get("bp")
    s5 = user.get("s5")
    s6 = user.get("s6")

    if bmi is not None and bmi > 0.03:
        recs.append("Your BMI-related input is above average; weight management may help reduce diabetes risk.")

    if bp is not None and bp > 0.03:
        recs.append("Your blood-pressure-related input is elevated; monitor blood pressure and maintain a heart-healthy diet.")

    if s5 is not None and s5 > 0.03:
        recs.append("Your triglyceride-related input is elevated; reduce sugary foods and refined carbohydrates.")

    if s6 is not None and s6 > 0.03:
        recs.append("Your blood-sugar-related input is elevated; consider glucose screening and reduce added sugar intake.")

    return recs


def hypertension_feature_recommendations(user: dict):
    recs = []

    trestbps = user.get("trestbps")
    chol = user.get("chol")
    fbs = user.get("fbs")
    exang = user.get("exang")

    if trestbps is not None and trestbps >= 140:
        recs.append("Your resting blood pressure is high; reduce salt intake and monitor your blood pressure regularly.")

    if chol is not None and chol >= 240:
        recs.append("Your cholesterol appears elevated; improving dietary quality may help reduce cardiovascular strain.")

    if fbs == 1:
        recs.append("Raised fasting blood sugar can increase long-term vascular risk; consider follow-up glucose testing.")

    if exang == 1:
        recs.append("Exercise-related chest symptoms should be reviewed clinically before intense physical activity.")

    return recs


def metabolic_feature_recommendations(user: dict):
    recs = []

    bmi = user.get("bmi")
    waist_cm = user.get("waist_cm")
    systolic_bp = user.get("systolic_bp")
    fbs_high = user.get("fbs_high")
    activity_minutes = user.get("activity_minutes")

    if bmi is not None and bmi >= 25:
        recs.append("Your BMI suggests excess body weight; gradual weight management may reduce metabolic risk.")

    if waist_cm is not None:
        recs.append("If waist circumference is elevated, reducing abdominal fat can improve metabolic health.")

    if systolic_bp is not None and systolic_bp >= 130:
        recs.append("Your blood pressure is elevated; reducing salt intake and regular exercise may help.")

    if fbs_high == 1:
        recs.append("Raised fasting blood sugar suggests glucose regulation may need review; consider follow-up testing.")

    if activity_minutes is not None and activity_minutes < 150:
        recs.append("Your weekly activity is below the recommended level; increasing movement may reduce metabolic risk.")

    return recs
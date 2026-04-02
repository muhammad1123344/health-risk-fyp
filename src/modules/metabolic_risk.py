def pretty_feature_name(name: str) -> str:
    labels = {
        "age": "Age",
        "sex": "Sex",
        "bmi": "Body Mass Index (BMI)",
        "waist_cm": "Waist Circumference (cm)",
        "systolic_bp": "Systolic Blood Pressure",
        "fbs_high": "Fasting Blood Sugar > 120 mg/dl",
        "activity_minutes": "Weekly Activity Minutes",
    }
    return labels.get(name, name)


def predict_risk(user: dict) -> float:
    """
    Rule-based metabolic risk screening score converted to 0-1 probability.
    This is a structured screening module, not a trained ML model.
    """
    score = 0
    max_score = 9

    age = user.get("age")
    sex = user.get("sex")
    bmi = user.get("bmi")
    waist_cm = user.get("waist_cm")
    systolic_bp = user.get("systolic_bp")
    fbs_high = user.get("fbs_high")
    activity_minutes = user.get("activity_minutes")

    if age is not None and age >= 45:
        score += 1

    if bmi is not None:
        if bmi >= 30:
            score += 2
        elif bmi >= 25:
            score += 1

    if waist_cm is not None and sex is not None:
        if sex == 1 and waist_cm >= 102:   # Male
            score += 2
        elif sex == 0 and waist_cm >= 88:  # Female
            score += 2

    if systolic_bp is not None:
        if systolic_bp >= 140:
            score += 2
        elif systolic_bp >= 130:
            score += 1

    if fbs_high == 1:
        score += 2

    if activity_minutes is not None and activity_minutes < 150:
        score += 1

    return min(score / max_score, 1.0)


def driver_lines(user: dict):
    lines = []

    age = user.get("age")
    sex = user.get("sex")
    bmi = user.get("bmi")
    waist_cm = user.get("waist_cm")
    systolic_bp = user.get("systolic_bp")
    fbs_high = user.get("fbs_high")
    activity_minutes = user.get("activity_minutes")

    if age is not None and age >= 45:
        lines.append("Age increases estimated metabolic risk")

    if bmi is not None:
        if bmi >= 30:
            lines.append("High BMI increases estimated metabolic risk")
        elif bmi >= 25:
            lines.append("Elevated BMI increases estimated metabolic risk")

    if waist_cm is not None and sex is not None:
        if sex == 1 and waist_cm >= 102:
            lines.append("High waist circumference increases estimated metabolic risk")
        elif sex == 0 and waist_cm >= 88:
            lines.append("High waist circumference increases estimated metabolic risk")

    if systolic_bp is not None:
        if systolic_bp >= 140:
            lines.append("High systolic blood pressure increases estimated metabolic risk")
        elif systolic_bp >= 130:
            lines.append("Elevated systolic blood pressure increases estimated metabolic risk")

    if fbs_high == 1:
        lines.append("Raised fasting blood sugar increases estimated metabolic risk")

    if activity_minutes is not None and activity_minutes < 150:
        lines.append("Low weekly activity increases estimated metabolic risk")

    if not lines:
        lines.append("No major metabolic risk drivers were detected from the provided values.")

    return lines
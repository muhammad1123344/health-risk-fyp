# app/streamlit_app.py

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules import heart_disease, diabetes, hypertension, metabolic_risk
from src.recommendations import (
    risk_band,
    general_recommendations,
    heart_feature_recommendations,
    diabetes_feature_recommendations,
    hypertension_feature_recommendations,
    metabolic_feature_recommendations,
)
from src.explainability import top_drivers_logreg, drivers_to_readable_lines
from src.reporting import build_pdf_report

st.set_page_config(
    page_title="Health Risk System",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
        .main-title {
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }
        .sub-text {
            color: #9aa0a6;
            margin-bottom: 1.5rem;
        }
        .section-card {
            padding: 1rem 1.2rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }
        .small-note {
            font-size: 0.9rem;
            color: #9aa0a6;
        }
        .result-card {
            padding: 1rem 1.2rem;
            border-radius: 16px;
            background: rgba(0, 128, 0, 0.12);
            border: 1px solid rgba(0, 255, 128, 0.20);
            margin: 0.8rem 0 1rem 0;
        }
        .driver-card {
            padding: 0.8rem 1rem;
            border-radius: 12px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            margin-bottom: 0.6rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">AI-Powered Health Risk Prediction and Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Educational system — not medical advice.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("System Overview")
    st.write("This system predicts health-related risk and provides recommendations across multiple modules.")
    st.write("**Modules included:**")
    st.write("• Heart Disease (CVD)")
    st.write("• Diabetes Risk")
    st.write("• Hypertension Risk")
    st.write("• Obesity / Metabolic Risk")
    st.divider()
    st.write("**Workflow**")
    st.write("1. Choose a module")
    st.write("2. Enter values")
    st.write("3. Predict risk")
    st.write("4. Review explanation")
    st.write("5. Download report")
    st.divider()
    if st.button("Reset Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()


def numeric_with_unsure(label: str, default: float, minv=None, maxv=None, step=None, key=None):
    unsure = st.checkbox(f"Not sure ({label})", value=False, key=f"unsure_{key or label}")
    if unsure:
        return None
    return st.number_input(
        label,
        value=float(default),
        min_value=minv,
        max_value=maxv,
        step=step,
        key=key
    )


def confidence_score(prob: float) -> float:
    return abs(prob - 0.5) * 2


def show_confidence(prob: float):
    conf = confidence_score(prob)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Risk Probability", f"{prob:.2f}")
    with c2:
        st.metric("Confidence Score", f"{conf:.2f}")

    if conf < 0.30:
        st.warning("Low confidence prediction — the risk is close to the decision boundary or several values were uncertain.")
    elif conf < 0.60:
        st.info("Moderate confidence prediction.")
    else:
        st.success("High confidence prediction.")


def show_rule_based_confidence(missing: int, total_inputs: int):
    conf = max(0.0, 1.0 - (missing / total_inputs))
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Screening Completion", f"{((total_inputs-missing)/total_inputs)*100:.0f}%")
    with c2:
        st.metric("Confidence Score", f"{conf:.2f}")

    if conf < 0.50:
        st.warning("Low confidence screening — several values were missing.")
    elif conf < 0.80:
        st.info("Moderate confidence screening.")
    else:
        st.success("High confidence screening.")


def show_input_summary(inputs_display: dict):
    for k, v in inputs_display.items():
        st.write(f"**{k}:** {v}")


def json_download(data: dict, filename: str):
    st.download_button(
        label="Download JSON Summary",
        data=json.dumps(data, indent=2),
        file_name=filename,
        mime="application/json",
        use_container_width=True
    )


module = st.selectbox(
    "Select health risk module",
    ["Heart Disease (CVD)", "Diabetes Risk", "Hypertension Risk", "Obesity / Metabolic Risk"]
)

# -------------------------
# HEART DISEASE MODULE
# -------------------------
if module == "Heart Disease (CVD)":
    model = heart_disease.load_model()

    st.markdown("### Heart Disease (CVD) Module")
    st.info("This module estimates cardiovascular risk from clinical and self-reported indicators.")

    input_col, result_col = st.columns([1.15, 1], gap="large")

    with input_col:
        st.markdown("#### Enter Patient Information")

        age = numeric_with_unsure("Age (years)", 50, minv=18.0, maxv=100.0, step=1.0, key="hd_age")
        trestbps = numeric_with_unsure("Resting Blood Pressure (mm Hg)", 120, minv=50.0, maxv=250.0, step=1.0, key="hd_trestbps")
        chol = numeric_with_unsure("Serum Cholesterol (mg/dl)", 200, minv=50.0, maxv=700.0, step=1.0, key="hd_chol")
        thalach = numeric_with_unsure("Maximum Heart Rate Achieved", 150, minv=50.0, maxv=250.0, step=1.0, key="hd_thalach")
        oldpeak = numeric_with_unsure("ST Depression (Oldpeak)", 1.0, minv=0.0, maxv=10.0, step=0.1, key="hd_oldpeak")

        sex = st.selectbox("Sex", ["Not sure", "Male", "Female"], key="hd_sex")
        sex_val = 1 if sex == "Male" else 0 if sex == "Female" else None

        cp = st.selectbox("Chest Pain Type", ["Not sure"] + list(heart_disease.CP_MAP.keys()), key="hd_cp")
        cp_val = None if cp == "Not sure" else heart_disease.CP_MAP[cp]

        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["Not sure", "No", "Yes"], key="hd_fbs")
        fbs_val = None if fbs == "Not sure" else (1 if fbs == "Yes" else 0)

        restecg = st.selectbox("Resting ECG Results", ["Not sure"] + list(heart_disease.RESTECG_MAP.keys()), key="hd_restecg")
        restecg_val = None if restecg == "Not sure" else heart_disease.RESTECG_MAP[restecg]

        exang = st.selectbox("Exercise Induced Angina?", ["Not sure", "No", "Yes"], key="hd_exang")
        exang_val = None if exang == "Not sure" else (1 if exang == "Yes" else 0)

        slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Not sure"] + list(heart_disease.SLOPE_MAP.keys()), key="hd_slope")
        slope_val = None if slope == "Not sure" else heart_disease.SLOPE_MAP[slope]

        ca = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", ["Not sure", 0, 1, 2, 3], key="hd_ca")
        ca_val = None if ca == "Not sure" else float(ca)

        thal = st.selectbox("Thalassemia Status", ["Not sure"] + list(heart_disease.THAL_MAP.keys()), key="hd_thal")
        thal_val = None if thal == "Not sure" else heart_disease.THAL_MAP[thal]

        user = {
            "age": age, "sex": sex_val, "cp": cp_val, "trestbps": trestbps, "chol": chol,
            "fbs": fbs_val, "restecg": restecg_val, "thalach": thalach, "exang": exang_val,
            "oldpeak": oldpeak, "slope": slope_val, "ca": ca_val, "thal": thal_val
        }

        inputs_display = {
            "Age (years)": "Not sure" if age is None else int(age),
            "Sex": sex,
            "Chest Pain Type": cp,
            "Resting Blood Pressure (mm Hg)": "Not sure" if trestbps is None else int(trestbps),
            "Serum Cholesterol (mg/dl)": "Not sure" if chol is None else int(chol),
            "Fasting Blood Sugar > 120 mg/dl?": fbs,
            "Resting ECG Results": restecg,
            "Maximum Heart Rate Achieved": "Not sure" if thalach is None else int(thalach),
            "Exercise Induced Angina?": exang,
            "ST Depression (Oldpeak)": "Not sure" if oldpeak is None else float(oldpeak),
            "Slope of Peak Exercise ST Segment": slope,
            "Number of Major Vessels (0-3)": ca,
            "Thalassemia Status": thal,
        }

        with st.expander("Review entered values"):
            show_input_summary(inputs_display)

        predict_clicked = st.button("Predict Risk", key="hd_predict", use_container_width=True)

    with result_col:
        st.markdown("#### Prediction Output")

        if predict_clicked:
            X_user = pd.DataFrame([user])
            missing = int(X_user.isna().sum().sum())

            if missing > 0:
                st.warning(f"{missing} value(s) were estimated because you selected 'Not sure'.")

            p = heart_disease.predict_risk(model, user)
            band = risk_band(p)

            st.markdown(
                f'<div class="result-card"><b>Predicted risk probability:</b> {p:.2f} ({band})</div>',
                unsafe_allow_html=True
            )

            show_confidence(p)

            st.markdown("### Risk Explanation")
            st.write(
                "Your cardiovascular risk is assessed from the pattern of clinical indicators provided."
                if band != "Low" else
                "Your overall cardiovascular risk is currently low based on the information provided."
            )

            st.markdown("### Key Drivers (Top 3)")
            try:
                feature_names = heart_disease.get_feature_names(model)
                drivers = top_drivers_logreg(model, X_user, feature_names, top_k=3)
                driver_lines = drivers_to_readable_lines(drivers, heart_disease.pretty_feature_name)
            except Exception as e:
                st.error(f"Explainability error: {e}")
                driver_lines = ["Top drivers could not be generated for this prediction."]

            for line in driver_lines:
                st.markdown(f'<div class="driver-card">• {line}</div>', unsafe_allow_html=True)

            st.markdown("### Recommendations")
            recs = general_recommendations(band)
            feature_recs = heart_feature_recommendations(user)
            extra = (
                ["Maintain regular aerobic activity and a balanced diet."]
                if band == "Low" else
                ["Consider routine checks for blood pressure and cholesterol with a clinician.", "Reduce salt intake and limit saturated fats."]
                if band == "Moderate" else
                ["Seek medical advice for personalised cardiovascular assessment.", "If you have chest pain, shortness of breath, or dizziness, get urgent medical help."]
            )

            all_recs = recs + feature_recs + extra
            for r in all_recs:
                st.write(f"• {r}")

            pdf_bytes = build_pdf_report(
                title="AI-Powered Health Risk Prediction and Recommendation System",
                module_name="Heart Disease (CVD)",
                risk_probability=p,
                risk_band=band,
                inputs_display=inputs_display,
                drivers_lines=driver_lines,
                recommendations=all_recs,
                missing_count=missing,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download PDF Report",
                    pdf_bytes,
                    "heart_disease_risk_report.pdf",
                    "application/pdf",
                    use_container_width=True
                )
            with c2:
                json_download({
                    "module": "Heart Disease (CVD)",
                    "risk_probability": round(p, 4),
                    "risk_band": band,
                    "inputs": inputs_display,
                    "drivers": driver_lines,
                    "recommendations": all_recs,
                    "missing_values_estimated": missing,
                }, "heart_disease_risk_summary.json")

# -------------------------
# DIABETES MODULE
# -------------------------
if module == "Diabetes Risk":
    model = diabetes.load_model()

    st.markdown("### Diabetes Risk Module")
    st.info("This module uses a diabetes dataset with standardised numeric features. Values are not raw clinical units.")

    input_col, result_col = st.columns([1.15, 1], gap="large")

    with input_col:
        st.markdown("#### Enter Patient Information")

        user = {}
        inputs_display = {}

        for f in diabetes.FEATURES:
            label = diabetes.pretty_feature_name(f)
            val = numeric_with_unsure(label, 0.0, key=f"db_{f}")
            user[f] = val
            inputs_display[label] = "Not sure" if val is None else float(val)

        with st.expander("Review entered values"):
            show_input_summary(inputs_display)

        predict_clicked = st.button("Predict Risk", key="db_predict", use_container_width=True)

    with result_col:
        st.markdown("#### Prediction Output")

        if predict_clicked:
            X_user = pd.DataFrame([user])
            missing = int(X_user.isna().sum().sum())

            if missing > 0:
                st.warning(f"{missing} value(s) were estimated because you selected 'Not sure'.")

            p = diabetes.predict_risk(model, user)
            band = risk_band(p)

            st.markdown(
                f'<div class="result-card"><b>Predicted risk probability:</b> {p:.2f} ({band})</div>',
                unsafe_allow_html=True
            )

            show_confidence(p)

            st.markdown("### Risk Explanation")
            st.write(
                "Your diabetes-related risk is low based on the current inputs."
                if band == "Low" else
                "Your diabetes-related risk is elevated enough that preventive action may be useful."
            )

            st.markdown("### Key Drivers (Top 3)")
            try:
                drivers = top_drivers_logreg(model, X_user, diabetes.FEATURES, top_k=3)
                driver_lines = drivers_to_readable_lines(drivers, diabetes.pretty_feature_name)
            except Exception as e:
                st.error(f"Explainability error: {e}")
                driver_lines = ["Top drivers could not be generated for this prediction."]

            for line in driver_lines:
                st.markdown(f'<div class="driver-card">• {line}</div>', unsafe_allow_html=True)

            st.markdown("### Recommendations")
            recs = general_recommendations(band)
            feature_recs = diabetes_feature_recommendations(user)
            extra = (
                ["Maintain a healthy weight and stay physically active consistently."]
                if band == "Low" else
                ["Consider checking HbA1c or fasting glucose during routine screening.", "Increase fibre intake and reduce sugary drinks and processed snacks."]
                if band == "Moderate" else
                ["Speak to a clinician about glucose testing and personalised risk management.", "If you have symptoms such as excess thirst, frequent urination, or fatigue, seek medical advice promptly."]
            )

            all_recs = recs + feature_recs + extra
            for r in all_recs:
                st.write(f"• {r}")

            pdf_bytes = build_pdf_report(
                title="AI-Powered Health Risk Prediction and Recommendation System",
                module_name="Diabetes Risk",
                risk_probability=p,
                risk_band=band,
                inputs_display=inputs_display,
                drivers_lines=driver_lines,
                recommendations=all_recs,
                missing_count=missing,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download PDF Report",
                    pdf_bytes,
                    "diabetes_risk_report.pdf",
                    "application/pdf",
                    use_container_width=True
                )
            with c2:
                json_download({
                    "module": "Diabetes Risk",
                    "risk_probability": round(p, 4),
                    "risk_band": band,
                    "inputs": inputs_display,
                    "drivers": driver_lines,
                    "recommendations": all_recs,
                    "missing_values_estimated": missing,
                }, "diabetes_risk_summary.json")

# -------------------------
# HYPERTENSION MODULE
# -------------------------
if module == "Hypertension Risk":
    model = hypertension.load_model()

    st.markdown("### Hypertension Risk Module")
    st.info("This module screens hypertension-related risk using blood pressure and related indicators.")

    input_col, result_col = st.columns([1.15, 1], gap="large")

    with input_col:
        st.markdown("#### Enter Patient Information")

        age = numeric_with_unsure("Age (years)", 50, minv=18.0, maxv=100.0, step=1.0, key="ht_age")
        trestbps = numeric_with_unsure("Resting Blood Pressure (mm Hg)", 130, minv=50.0, maxv=250.0, step=1.0, key="ht_trestbps")
        chol = numeric_with_unsure("Serum Cholesterol (mg/dl)", 200, minv=50.0, maxv=700.0, step=1.0, key="ht_chol")
        thalach = numeric_with_unsure("Maximum Heart Rate Achieved", 150, minv=50.0, maxv=250.0, step=1.0, key="ht_thalach")
        oldpeak = numeric_with_unsure("ST Depression (Oldpeak)", 1.0, minv=0.0, maxv=10.0, step=0.1, key="ht_oldpeak")

        sex = st.selectbox("Sex", ["Not sure", "Male", "Female"], key="ht_sex")
        sex_val = 1 if sex == "Male" else 0 if sex == "Female" else None

        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["Not sure", "No", "Yes"], key="ht_fbs")
        fbs_val = None if fbs == "Not sure" else (1 if fbs == "Yes" else 0)

        exang = st.selectbox("Exercise Induced Angina?", ["Not sure", "No", "Yes"], key="ht_exang")
        exang_val = None if exang == "Not sure" else (1 if exang == "Yes" else 0)

        user = {
            "age": age,
            "sex": sex_val,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs_val,
            "thalach": thalach,
            "oldpeak": oldpeak,
            "exang": exang_val,
        }

        inputs_display = {
            "Age (years)": "Not sure" if age is None else int(age),
            "Sex": sex,
            "Resting Blood Pressure (mm Hg)": "Not sure" if trestbps is None else int(trestbps),
            "Serum Cholesterol (mg/dl)": "Not sure" if chol is None else int(chol),
            "Fasting Blood Sugar > 120 mg/dl?": fbs,
            "Maximum Heart Rate Achieved": "Not sure" if thalach is None else int(thalach),
            "ST Depression (Oldpeak)": "Not sure" if oldpeak is None else float(oldpeak),
            "Exercise Induced Angina?": exang,
        }

        with st.expander("Review entered values"):
            show_input_summary(inputs_display)

        predict_clicked = st.button("Predict Risk", key="ht_predict", use_container_width=True)

    with result_col:
        st.markdown("#### Prediction Output")

        if predict_clicked:
            X_user = pd.DataFrame([user])
            missing = int(X_user.isna().sum().sum())

            if missing > 0:
                st.warning(f"{missing} value(s) were estimated because you selected 'Not sure'.")

            p = hypertension.predict_risk(model, user)
            band = risk_band(p)

            st.markdown(
                f'<div class="result-card"><b>Predicted risk probability:</b> {p:.2f} ({band})</div>',
                unsafe_allow_html=True
            )

            show_confidence(p)

            st.markdown("### Risk Explanation")
            st.write("This module estimates hypertension-related risk from blood pressure and related health indicators.")

            st.markdown("### Key Drivers (Top 3)")
            try:
                feature_names = hypertension.get_feature_names(model)
                drivers = top_drivers_logreg(model, X_user, feature_names, top_k=3)
                driver_lines = drivers_to_readable_lines(drivers, hypertension.pretty_feature_name)
            except Exception as e:
                st.error(f"Explainability error: {e}")
                driver_lines = ["Top drivers could not be generated for this prediction."]

            for line in driver_lines:
                st.markdown(f'<div class="driver-card">• {line}</div>', unsafe_allow_html=True)

            st.markdown("### Recommendations")
            recs = general_recommendations(band)
            feature_recs = hypertension_feature_recommendations(user)
            extra = (
                ["Maintain a healthy lifestyle and monitor your blood pressure periodically."]
                if band == "Low" else
                ["Reduce salt intake, improve dietary quality, and monitor blood pressure regularly."]
                if band == "Moderate" else
                ["Seek clinical review for blood pressure assessment and management.", "Monitor blood pressure closely and avoid ignoring persistent high readings."]
            )

            all_recs = recs + feature_recs + extra
            for r in all_recs:
                st.write(f"• {r}")

            pdf_bytes = build_pdf_report(
                title="AI-Powered Health Risk Prediction and Recommendation System",
                module_name="Hypertension Risk",
                risk_probability=p,
                risk_band=band,
                inputs_display=inputs_display,
                drivers_lines=driver_lines,
                recommendations=all_recs,
                missing_count=missing,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download PDF Report",
                    pdf_bytes,
                    "hypertension_risk_report.pdf",
                    "application/pdf",
                    use_container_width=True
                )
            with c2:
                json_download({
                    "module": "Hypertension Risk",
                    "risk_probability": round(p, 4),
                    "risk_band": band,
                    "inputs": inputs_display,
                    "drivers": driver_lines,
                    "recommendations": all_recs,
                    "missing_values_estimated": missing,
                }, "hypertension_risk_summary.json")

# -------------------------
# METABOLIC RISK MODULE
# -------------------------
if module == "Obesity / Metabolic Risk":
    st.markdown("### Obesity / Metabolic Risk Module")
    st.info("This is a structured screening module based on common metabolic risk indicators.")

    input_col, result_col = st.columns([1.15, 1], gap="large")

    with input_col:
        st.markdown("#### Enter Patient Information")

        age = numeric_with_unsure("Age (years)", 45, minv=18.0, maxv=100.0, step=1.0, key="mr_age")
        sex = st.selectbox("Sex", ["Not sure", "Male", "Female"], key="mr_sex")
        sex_val = 1 if sex == "Male" else 0 if sex == "Female" else None
        bmi = numeric_with_unsure("Body Mass Index (BMI)", 27.0, minv=10.0, maxv=60.0, step=0.1, key="mr_bmi")
        waist_cm = numeric_with_unsure("Waist Circumference (cm)", 95.0, minv=40.0, maxv=200.0, step=0.5, key="mr_waist")
        systolic_bp = numeric_with_unsure("Systolic Blood Pressure", 130.0, minv=50.0, maxv=250.0, step=1.0, key="mr_bp")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["Not sure", "No", "Yes"], key="mr_fbs")
        fbs_val = None if fbs == "Not sure" else (1 if fbs == "Yes" else 0)
        activity_minutes = numeric_with_unsure("Weekly Activity Minutes", 120.0, minv=0.0, maxv=2000.0, step=10.0, key="mr_activity")

        user = {
            "age": age,
            "sex": sex_val,
            "bmi": bmi,
            "waist_cm": waist_cm,
            "systolic_bp": systolic_bp,
            "fbs_high": fbs_val,
            "activity_minutes": activity_minutes,
        }

        inputs_display = {
            "Age (years)": "Not sure" if age is None else int(age),
            "Sex": sex,
            "Body Mass Index (BMI)": "Not sure" if bmi is None else float(bmi),
            "Waist Circumference (cm)": "Not sure" if waist_cm is None else float(waist_cm),
            "Systolic Blood Pressure": "Not sure" if systolic_bp is None else float(systolic_bp),
            "Fasting Blood Sugar > 120 mg/dl?": fbs,
            "Weekly Activity Minutes": "Not sure" if activity_minutes is None else float(activity_minutes),
        }

        with st.expander("Review entered values"):
            show_input_summary(inputs_display)

        predict_clicked = st.button("Predict Risk", key="mr_predict", use_container_width=True)

    with result_col:
        st.markdown("#### Prediction Output")

        if predict_clicked:
            X_user = pd.DataFrame([user])
            missing = int(X_user.isna().sum().sum())

            if missing > 0:
                st.warning(f"{missing} value(s) were estimated because you selected 'Not sure'.")

            p = metabolic_risk.predict_risk(user)
            band = risk_band(p)

            st.markdown(
                f'<div class="result-card"><b>Predicted risk probability:</b> {p:.2f} ({band})</div>',
                unsafe_allow_html=True
            )

            show_rule_based_confidence(missing, total_inputs=7)

            st.markdown("### Risk Explanation")
            st.write("This module estimates metabolic risk from body composition, blood pressure, glucose status, and activity level.")

            st.markdown("### Key Drivers (Top 3)")
            driver_lines = metabolic_risk.driver_lines(user)[:3]
            for line in driver_lines:
                st.markdown(f'<div class="driver-card">• {line}</div>', unsafe_allow_html=True)

            st.markdown("### Recommendations")
            recs = general_recommendations(band)
            feature_recs = metabolic_feature_recommendations(user)
            extra = (
                ["Maintain healthy body weight and regular physical activity."]
                if band == "Low" else
                ["Increase physical activity and review diet quality to improve metabolic health."]
                if band == "Moderate" else
                ["Seek clinical review for personalised metabolic risk assessment.", "Focus on weight management, blood pressure control, and glucose monitoring."]
            )

            all_recs = recs + feature_recs + extra
            for r in all_recs:
                st.write(f"• {r}")

            pdf_bytes = build_pdf_report(
                title="AI-Powered Health Risk Prediction and Recommendation System",
                module_name="Obesity / Metabolic Risk",
                risk_probability=p,
                risk_band=band,
                inputs_display=inputs_display,
                drivers_lines=driver_lines,
                recommendations=all_recs,
                missing_count=missing,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download PDF Report",
                    pdf_bytes,
                    "metabolic_risk_report.pdf",
                    "application/pdf",
                    use_container_width=True
                )
            with c2:
                json_download({
                    "module": "Obesity / Metabolic Risk",
                    "risk_probability": round(p, 4),
                    "risk_band": band,
                    "inputs": inputs_display,
                    "drivers": driver_lines,
                    "recommendations": all_recs,
                    "missing_values_estimated": missing,
                }, "metabolic_risk_summary.json")
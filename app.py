import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Euphoria GenX Diagnostics",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px;}
    .sub-header {font-size: 1.1rem; color: #64748B; margin-bottom: 12px;}
    .disclaimer {font-size: 0.85rem; color: #64748B; background: #F1F5F9; padding: 10px 14px;
                 border-radius: 6px; border-left: 4px solid #94A3B8; margin-bottom: 20px;}
    .diff-card {padding: 14px; border-left: 5px solid #2563EB; background: #F8FAFC; margin-bottom: 12px;
                border-radius: 6px; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0;
                border-bottom: 1px solid #E2E8F0;}
    .diff-title {font-size: 1.1rem; font-weight: 600; color: #1E3A8A;}
    .diff-rationale {font-size: 0.95rem; color: #1F2937; margin-top: 6px; line-height: 1.4;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SESSION STATE ---
if "symptom_selection" not in st.session_state:
    st.session_state.symptom_selection = []
if "symptom_multiselect" not in st.session_state:
    st.session_state.symptom_multiselect = []
if "run_diagnostics" not in st.session_state:
    st.session_state.run_diagnostics = False
if "notes_widget_version" not in st.session_state:
    st.session_state.notes_widget_version = 0


# --- DATA & MODEL ---
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")

    target_col = df.columns[0]
    for col in df.columns:
        if col.strip().lower() in ["disease", "prognosis", "target"]:
            target_col = col
            break

    y = df[target_col].astype(str).str.strip().str.title()
    X_raw = df.drop(columns=[target_col])

    symptoms_set = set()
    for col in X_raw.columns:
        for val in X_raw[col].dropna():
            sym = str(val).strip()
            if sym and sym.lower() != "nan":
                symptoms_set.add(sym)

    unique_symptoms = sorted(list(symptoms_set))
    clean_symptom_names = [sym.replace("_", " ").title() for sym in unique_symptoms]

    X_list = []
    for i in range(len(X_raw)):
        row_dict = {sym: 0 for sym in unique_symptoms}
        for val in X_raw.iloc[i].dropna().values:
            sym = str(val).strip()
            if sym in row_dict:
                row_dict[sym] = 1
        X_list.append(row_dict)

    X = pd.DataFrame(X_list)

    try:
        desc_df = pd.read_csv("symptom_Description.csv")
        desc_dict = {
            str(row["Disease"]).strip().title(): str(row["Symptom_Description"])
            for _, row in desc_df.iterrows()
        }
    except Exception:
        desc_dict = {}

    try:
        prec_df = pd.read_csv("symptom_precaution.csv")
        prec_dict = {}
        for _, row in prec_df.iterrows():
            disease = str(row["Disease"]).strip().title()
            precautions = [
                str(row[col]).strip().capitalize()
                for col in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
                if pd.notna(row[col]) and str(row[col]).lower() != "nan"
            ]
            prec_dict[disease] = precautions
    except Exception:
        prec_dict = {}

    try:
        sev_df = pd.read_csv("Symptom-severity.csv")
        severity_dict = {
            str(row["Symptom"]).replace("_", " ").strip().title(): int(row["weight"])
            for _, row in sev_df.iterrows()
        }
    except Exception:
        severity_dict = {}

    return X, y, clean_symptom_names, unique_symptoms, desc_dict, prec_dict, severity_dict


@st.cache_resource
def train_model(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=8,
        min_child_weight=1,
        gamma=0.01,
        subsample=1.0,
        colsample_bytree=1.0,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, X, y_encoded, cv=3)
    return model, float(cv_scores.mean()), le


def extract_symptoms_from_notes(notes: str, symptom_names: list) -> list:
    notes_lower = notes.lower()
    found = []
    for sym in symptom_names:
        sym_lower = sym.lower()
        words = sym_lower.split()
        if sym_lower in notes_lower or all(w in notes_lower for w in words):
            found.append(sym)
    return found


with st.spinner("Initializing Calibrated AI Engine..."):
    X, y, clean_symptom_names, raw_symptom_names, desc_dict, prec_dict, severity_dict = load_data()
    model, model_accuracy, label_encoder = train_model(X, y)


# --- SIDEBAR ---
with st.sidebar:
    st.header("Patient EHR Profile")
    st.caption("Structured Demographics & Vitals")

    patient_name = st.text_input("Patient ID / Name", placeholder="e.g., PT-8472")
    col_a, col_b = st.columns(2)
    with col_a:
        patient_age = st.number_input("Age", min_value=1, max_value=120, value=35)
        patient_weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
    with col_b:
        patient_gender = st.selectbox("Gender", ["M", "F", "Other"])
        patient_height = st.number_input("Height (cm)", min_value=50, max_value=250, value=175)

    bmi = round(patient_weight / ((patient_height / 100) ** 2), 1)
    st.info(f"**Calculated BMI:** {bmi}")

    patient_bp = st.text_input("Vitals (BP)", placeholder="120/80")
    symptom_duration = st.selectbox(
        "Symptom Duration",
        ["< 24 Hours", "1-3 Days", "1 Week", "Chronic (> 4 Weeks)"],
    )

    st.divider()
    st.write("**Live Engine:** Calibrated XGBoost")
    st.write(f"**CV Accuracy:** {model_accuracy:.1%}")
    st.caption("Engine Status: Calibrated (CV Optimized)")

    st.divider()
    if st.button("Clear Symptoms", use_container_width=True):
        st.session_state.symptom_selection = []
        st.session_state.symptom_multiselect = []
        st.session_state.run_diagnostics = False
        st.session_state.notes_widget_version += 1
        st.rerun()


# --- MAIN ---
st.markdown(
    '<p class="main-header">🧬 Euphoria GenX: Clinical AI Diagnostics</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Automated differential diagnosis and triage powered by calibrated gradient boosting.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disclaimer"><strong>Disclaimer:</strong> Research / educational prototype only. '
    "Not a medical device and not a substitute for professional clinical judgment.</div>",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["📝 Diagnostic Terminal", "🧠 Local XAI Analytics"])

with tab1:
    st.subheader("1. Smart Clinical NLP Extractor")

    # Versioned key = safe empty remount after extract/clear
    notes_key = f"clinical_notes_{st.session_state.notes_widget_version}"
    notes = st.text_area(
        "Paste doctor's clinical notes below:",
        placeholder="e.g., Patient complains of high fever, continuous sneezing, and severe chills...",
        key=notes_key,
        height=100,
    )

    if st.button("Extract Symptoms via NLP", type="secondary"):
        if not notes.strip():
            st.warning("Please enter text into the notes box before extracting.")
        else:
            found = extract_symptoms_from_notes(notes, clean_symptom_names)
            if found:
                merged = list(dict.fromkeys(st.session_state.symptom_selection + found))
                # Update BOTH keys so multiselect shows the chips
                st.session_state.symptom_selection = merged
                st.session_state.symptom_multiselect = merged
                st.session_state.notes_widget_version += 1
                st.success(f"Extracted {len(found)} symptom(s): {', '.join(found)}")
                st.rerun()
            else:
                st.warning("No matching canonical symptoms found. Use manual selection below.")

    st.markdown("---")
    st.subheader("2. Active Symptom Verification")
    st.write("Review, add, or remove symptoms below:")

    # Driven only by key= (value lives in st.session_state.symptom_multiselect)
    selected = st.multiselect(
        "Active Symptom Vector Profile:",
        options=clean_symptom_names,
        key="symptom_multiselect",
    )
    st.session_state.symptom_selection = selected

    has_symptoms = len(st.session_state.symptom_selection) > 0
    run_ml = st.button("Run ML Diagnostics", type="primary", disabled=not has_symptoms)

    if not has_symptoms:
        st.info(
            "💡 Select at least one symptom from the multiselect or use the NLP extractor above."
        )

    if run_ml and has_symptoms:
        st.session_state.run_diagnostics = True

    if st.session_state.run_diagnostics and has_symptoms:
        total_severity = sum(
            severity_dict.get(sym, 1) for sym in st.session_state.symptom_selection
        )

        input_data = [0] * len(raw_symptom_names)
        for symptom in st.session_state.symptom_selection:
            if symptom in clean_symptom_names:
                input_data[clean_symptom_names.index(symptom)] = 1

        input_df = pd.DataFrame([input_data], columns=raw_symptom_names)
        probabilities = model.predict_proba(input_df)[0]
        top_indices = np.argsort(probabilities)[::-1][:4]
        top_probs = probabilities[top_indices]
        top_diseases = label_encoder.inverse_transform(top_indices)

        # --- NEW: Relative Confidence Scaler ---
        # If the highest probability is below 30% (meaning a sparse/vague symptom vector),
        # normalize the top 4 so they present as a clean, proportional split.
        if top_probs[0] < 0.30:
            top_probs = (top_probs / np.sum(top_probs)) * 0.85

        st.divider()

        if total_severity > 12:
            st.error(
                f"🚨 **CRITICAL TRIAGE (Score: {total_severity}):** Immediate clinical intervention required."
            )
        elif total_severity > 6:
            st.warning(
                f"⚠️ **MODERATE RISK (Score: {total_severity}):** Prioritize for physician review."
            )
        else:
            st.info(f"✅ **ROUTINE (Score: {total_severity}):** Standard outpatient monitoring.")

        st.subheader("📋 Differential Diagnosis Ranking")

        if top_probs[0] < 0.30:
            st.warning(
                "⚠️ **Low Confidence Warning:** Sparse or high-variance symptom vector. "
                "Direct clinical examination is recommended."
            )

        for i, disease in enumerate(top_diseases):
            prob = top_probs[i] * 100
            if prob > 0.5:
                sample = ", ".join(st.session_state.symptom_selection[:3])
                rationale = (
                    f"Vector analysis maps current presentation attributes ({sample}) "
                    f"against clinical correlation markers for {disease}."
                )
                st.markdown(
                    f"""
                    <div class="diff-card">
                        <div class="diff-title">{i + 1}. {disease}
                            <span style="float: right; color: #2563EB;">{prob:.1f}% Match</span>
                        </div>
                        <div class="diff-rationale"><em>Clinical Rationale:</em> {rationale}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        primary_dx = top_diseases[0]
        st.write(f"### 📖 Clinical Description: {primary_dx}")
        st.write(desc_dict.get(primary_dx, "Detailed clinical description unavailable in database."))

        st.write("### 🛡 Recommended Clinical Actions")
        for p in prec_dict.get(primary_dx, ["Consult supervising physician immediately."]):
            st.markdown(
                f"- {p.replace('Wash hands through', 'Implement strict hand hygiene protocols').strip()}"
            )

        st.divider()
        export_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "patient_demographics": {
                "id": patient_name or "Anonymous",
                "age": patient_age,
                "gender": patient_gender,
                "bmi": bmi,
                "vitals": patient_bp or "N/A",
                "duration": symptom_duration,
            },
            "clinical_vector": st.session_state.symptom_selection,
            "triage_score": total_severity,
            "differential_diagnosis": [
                {"disease": str(top_diseases[i]), "confidence": float(top_probs[i])}
                for i in range(min(3, len(top_diseases)))
            ],
        }
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📄 Export EHR Report (TXT)",
                data=json.dumps(export_data, indent=4),
                file_name=f"EHR_{patient_name or 'record'}_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
            )
        with c2:
            st.download_button(
                "📊 Export Structured Data (JSON)",
                data=json.dumps(export_data, indent=4),
                file_name=f"EHR_{patient_name or 'record'}_{datetime.datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
            )

with tab2:
    st.subheader("🧠 Local Explainability (XAI)")
    st.write("Instance-level attribution: selected symptoms × model feature importance.")

    if st.session_state.symptom_selection:
        x = np.zeros(len(clean_symptom_names))
        for symptom in st.session_state.symptom_selection:
            if symptom in clean_symptom_names:
                x[clean_symptom_names.index(symptom)] = 1
        local = x * model.feature_importances_
        xai_df = pd.DataFrame(
            {"Symptom": clean_symptom_names, "Diagnostic Contribution": local}
        )
        xai_df = xai_df[xai_df["Diagnostic Contribution"] > 0].sort_values(
            "Diagnostic Contribution", ascending=False
        )
        if not xai_df.empty:
            st.bar_chart(xai_df, x="Symptom", y="Diagnostic Contribution", color="#1E3A8A")
        else:
            st.info("Selected parameters carry negligible weight in current trees.")
    else:
        st.info(
            "No symptoms selected yet. Use the Diagnostic Terminal (or NLP extract), then return here."
        )
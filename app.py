"""
╔══════════════════════════════════════════════════════════════╗
║        MULTI-DISEASE PREDICTION SYSTEM — Streamlit App       ║
║  Diseases: Diabetes | Heart | Kidney | Parkinson's |         ║
║            Liver | Lung Cancer | Thyroid | Alzheimer's |     ║
║            Dengue                                            ║
╚══════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── sklearn imports ───────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import joblib, os, io

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict — Multi-Disease AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0a0f1e; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #112240 100%);
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * { color: #ccd6f6 !important; }

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #64ffda, #00b4d8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: #8892b0;
    font-size: 1.05rem;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Cards */
.disease-card {
    background: linear-gradient(135deg, #112240, #0d1b2a);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #64ffda;
    margin-bottom: 1rem;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.5rem;
}

/* Result boxes */
.result-positive {
    background: linear-gradient(135deg, #3d0000, #5c1a1a);
    border: 2px solid #ff4444;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    font-family: 'Syne', sans-serif;
}
.result-negative {
    background: linear-gradient(135deg, #003d1a, #0d5c2e);
    border: 2px solid #44ff88;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    font-family: 'Syne', sans-serif;
}
.result-title { font-size: 1.8rem; font-weight: 800; margin-bottom: 0.3rem; }
.result-prob  { font-size: 1.1rem; color: #ccd6f6; }

/* Accuracy badge */
.acc-badge {
    display: inline-block;
    background: #0d2137;
    border: 1px solid #64ffda;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.85rem;
    color: #64ffda;
    font-weight: 600;
    margin-top: 0.5rem;
}

/* Streamlit overrides */
div[data-testid="stForm"] { background: transparent; border: none; }
.stButton > button {
    background: linear-gradient(135deg, #64ffda, #00b4d8);
    color: #0a0f1e;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
label { color: #a8b2d8 !important; font-size: 0.9rem !important; }
.stSelectbox > div > div { background: #112240; color: #ccd6f6; border-color: #1e3a5f; }
.stNumberInput > div > div > input { background: #112240; color: #ccd6f6; border-color: #1e3a5f; }
.stSlider .stSlider { color: #64ffda; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL TRAINING FUNCTIONS  (cached — run once)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_diabetes(path="diabetes.csv"):
    df = pd.read_csv(path)
    zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    imp = SimpleImputer(strategy="median")
    X = pd.DataFrame(imp.fit_transform(df.drop("Outcome", axis=1)), columns=df.drop("Outcome",axis=1).columns)
    y = df["Outcome"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, X.columns.tolist(), round(acc*100, 2)

@st.cache_resource(show_spinner=False)
def train_heart(path="heart_disease.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    X, y = df.drop("target", axis=1), df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([("sc", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))])
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, X.columns.tolist(), round(acc*100, 2)

@st.cache_resource(show_spinner=False)
def train_kidney(path="kidney_disease.csv"):
    df = pd.read_csv(path)
    df.drop(columns=["id"], errors="ignore", inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df["classification"] = df["classification"].replace({"ckd\t": "ckd"})
    for col in ["pcv","wc","rc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    cat_cols = [c for c in df.select_dtypes("object").columns if c != "classification"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    tgt_le = LabelEncoder()
    df["classification"] = tgt_le.fit_transform(df["classification"])
    df.fillna(df.median(numeric_only=True), inplace=True)
    X, y = df.drop("classification", axis=1), df["classification"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, le_dict, tgt_le, X.columns.tolist(), round(acc*100, 2)

@st.cache_resource(show_spinner=False)
def train_parkinsons(path="parkinsons.csv"):
    df = pd.read_csv(path)
    df.drop(columns=["name"], errors="ignore", inplace=True)
    X, y = df.drop("status", axis=1), df["status"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([("sc", StandardScaler()), ("clf", SVC(probability=True, random_state=42))])
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, X.columns.tolist(), round(acc*100, 2)

@st.cache_resource(show_spinner=False)
def train_liver(path="indian_liver_patient.csv"):
    df = pd.read_csv(path)
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])
    df["Albumin_and_Globulin_Ratio"].fillna(df["Albumin_and_Globulin_Ratio"].median(), inplace=True)
    df["Dataset"] = df["Dataset"].map({1: 1, 2: 0})
    X, y = df.drop("Dataset", axis=1), df["Dataset"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, X.columns.tolist(), round(acc*100, 2)

@st.cache_resource(show_spinner=False)
def train_lungcancer(path="survey_lung_cancer.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["GENDER"] = (df["GENDER"] == "M").astype(int)
    df["LUNG_CANCER"] = (df["LUNG_CANCER"] == "YES").astype(int)
    bin_cols = [c for c in df.columns if c not in ["GENDER","AGE","LUNG_CANCER"]]
    df[bin_cols] = df[bin_cols] - 1
    X, y = df.drop("LUNG_CANCER", axis=1), df["LUNG_CANCER"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([("sc", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))])
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, X.columns.tolist(), round(acc*100, 2)

@st.cache_resource(show_spinner=False)
def train_thyroid(path="thyroidDF.csv"):
    df = pd.read_csv(path)
    df.drop(columns=["patient_id","referral_source"], errors="ignore", inplace=True)
    df["target"] = (df["target"].astype(str).str.strip() != "-").astype(int)
    for col in df.select_dtypes("object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    X, y = df.drop("target", axis=1), df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, X.columns.tolist(), round(acc*100, 2)

@st.cache_resource(show_spinner=False)
def train_alzheimers(path="alzheimers_disease_data.csv"):
    df = pd.read_csv(path)
    drop_cols = ["PatientID","DoctorInCharge","Subject ID","MRI ID","Hand"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    for col in df.select_dtypes("object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df.fillna(df.median(numeric_only=True), inplace=True)
    X, y = df.drop("Diagnosis", axis=1), df["Diagnosis"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([("sc", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))])
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, X.columns.tolist(), round(acc*100, 2)

@st.cache_resource(show_spinner=False)
def train_dengue(path="dengue.csv"):
    df = pd.read_csv(path)
    df.drop(columns=["Name"], errors="ignore", inplace=True)
    X, y = df.drop("Dengue", axis=1), df["Dengue"]
    # Sample for speed
    if len(X) > 50000:
        idx = pd.Series(range(len(X))).sample(50000, random_state=42)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    svm = Pipeline([("sc", StandardScaler()),
                    ("clf", CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42), cv=3))])
    svm.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, svm.predict(X_te))
    return svm, X.columns.tolist(), round(acc*100, 2)

# ─────────────────────────────────────────────────────────────
# HELPER — show prediction result
# ─────────────────────────────────────────────────────────────
def show_result(pred, prob, positive_label="Disease Detected", negative_label="No Disease Detected"):
    if pred == 1:
        st.markdown(f"""
        <div class="result-positive">
            <div class="result-title" style="color:#ff6b6b;">⚠ {positive_label}</div>
            <div class="result-prob">Probability: <b>{prob:.1f}%</b></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <div class="result-title" style="color:#44ff88;">✔ {negative_label}</div>
            <div class="result-prob">Probability of disease: <b>{prob:.1f}%</b></div>
        </div>""", unsafe_allow_html=True)
    risk = "🔴 HIGH" if prob >= 70 else ("🟡 MODERATE" if prob >= 40 else "🟢 LOW")
    st.markdown(f"<div style='text-align:center;margin-top:0.8rem;color:#ccd6f6;font-size:1rem;'>Risk Level: <b>{risk}</b></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
DISEASES = {
    "🏠 Home"              : "home",
    "🩸 Diabetes"          : "diabetes",
    "❤️ Heart Disease"     : "heart",
    "🫘 Kidney Disease"    : "kidney",
    "🧠 Parkinson's"       : "parkinsons",
    "🫀 Liver Disease"     : "liver",
    "🫁 Lung Cancer"       : "lung",
    "🦋 Thyroid"           : "thyroid",
    "🧬 Alzheimer's"       : "alzheimers",
    "🦟 Dengue"            : "dengue",
}

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0;'>
        <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;
                    background:linear-gradient(135deg,#64ffda,#a78bfa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            🏥 MediPredict
        </div>
        <div style='color:#8892b0;font-size:0.8rem;margin-top:0.2rem;'>AI-Powered Disease Prediction</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    choice = st.radio("Navigate to:", list(DISEASES.keys()), label_visibility="collapsed")
    page = DISEASES[choice]
    st.markdown("---")
    st.markdown("<div style='color:#8892b0;font-size:0.75rem;text-align:center;'>⚠️ For educational purposes only.<br>Always consult a medical professional.</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────
if page == "home":
    st.markdown('<div class="hero-title">Multi-Disease<br>Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AI-powered screening for 9 diseases using Machine Learning</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    cards = [
        ("🩸", "Diabetes", "Glucose, BMI, Insulin-based prediction"),
        ("❤️", "Heart Disease", "ECG, cholesterol, chest pain analysis"),
        ("🫘", "Kidney Disease", "Blood, urine markers screening"),
        ("🧠", "Parkinson's", "Voice biomarker analysis"),
        ("🫀", "Liver Disease", "Enzyme & protein level screening"),
        ("🫁", "Lung Cancer", "Symptom & lifestyle risk assessment"),
        ("🦋", "Thyroid", "Hormone level classification"),
        ("🧬", "Alzheimer's", "Cognitive & lifestyle risk factors"),
        ("🦟", "Dengue", "Symptom-based fever screening"),
    ]
    for i, (icon, name, desc) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="disease-card" style="min-height:120px;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-family:Syne,sans-serif;font-weight:700;color:#ccd6f6;font-size:1.1rem;">{name}</div>
                <div style="color:#8892b0;font-size:0.85rem;margin-top:0.3rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disease-card" style="margin-top:1rem;">
        <div class="section-title">How to Use</div>
        <div style="color:#8892b0;line-height:1.9;">
        1. Select a disease from the <b style="color:#64ffda;">sidebar</b><br>
        2. Models train automatically on first load<br>
        3. Enter the patient's clinical values<br>
        4. Click <b style="color:#64ffda;">Predict</b> to get instant AI results<br>
        5. Review probability score and risk level
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DIABETES
# ─────────────────────────────────────────────────────────────
elif page == "diabetes":
    st.markdown('<div class="hero-title">🩸 Diabetes Predictor</div>', unsafe_allow_html=True)
    with st.spinner("Training model on diabetes dataset..."):
        model, features, acc = train_diabetes()
    st.markdown(f'<div class="acc-badge">Model Accuracy: {acc}%</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.form("diabetes_form"):
        st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose     = st.number_input("Glucose (mg/dL)", 0, 300, 120)
            bp          = st.number_input("Blood Pressure (mmHg)", 0, 200, 70)
        with c2:
            skin     = st.number_input("Skin Thickness (mm)", 0, 100, 20)
            insulin  = st.number_input("Insulin (µU/mL)", 0, 1000, 80)
            bmi      = st.number_input("BMI", 0.0, 70.0, 25.0)
        with c3:
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 1, 120, 30)
        submitted = st.form_submit_button("🔍 Predict Diabetes")

    if submitted:
        inp = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]], columns=features)
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0][1] * 100
        show_result(pred, prob, "Diabetes Detected", "No Diabetes Detected")

# ─────────────────────────────────────────────────────────────
# HEART DISEASE
# ─────────────────────────────────────────────────────────────
elif page == "heart":
    st.markdown('<div class="hero-title">❤️ Heart Disease Predictor</div>', unsafe_allow_html=True)
    with st.spinner("Training model on heart disease dataset..."):
        model, features, acc = train_heart()
    st.markdown(f'<div class="acc-badge">Model Accuracy: {acc}%</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.form("heart_form"):
        st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age      = st.number_input("Age", 1, 120, 50)
            sex      = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
            cp       = st.selectbox("Chest Pain Type", [0,1,2,3], format_func=lambda x: ["Typical Angina","Atypical","Non-anginal","Asymptomatic"][x])
            trestbps = st.number_input("Resting BP (mmHg)", 80, 250, 120)
            chol     = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        with c2:
            fbs     = st.selectbox("Fasting Blood Sugar >120", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            restecg = st.selectbox("Resting ECG", [0,1,2])
            thalach = st.number_input("Max Heart Rate", 60, 250, 150)
            exang   = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        with c3:
            oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
            slope   = st.selectbox("Slope of ST", [0,1,2])
            ca      = st.number_input("Major Vessels (0-4)", 0, 4, 0)
            thal    = st.selectbox("Thalassemia", [0,1,2,3])
        submitted = st.form_submit_button("🔍 Predict Heart Disease")

    if submitted:
        inp  = pd.DataFrame([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]], columns=features)
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0][1] * 100
        show_result(pred, prob, "Heart Disease Detected", "No Heart Disease Detected")

# ─────────────────────────────────────────────────────────────
# KIDNEY DISEASE
# ─────────────────────────────────────────────────────────────
elif page == "kidney":
    st.markdown('<div class="hero-title">🫘 Kidney Disease Predictor</div>', unsafe_allow_html=True)
    with st.spinner("Training model on kidney disease dataset..."):
        model, le_dict, tgt_le, features, acc = train_kidney()
    st.markdown(f'<div class="acc-badge">Model Accuracy: {acc}%</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.form("kidney_form"):
        st.markdown('<div class="section-title">Patient Lab Values</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age  = st.number_input("Age", 1, 100, 45)
            bp   = st.number_input("Blood Pressure (mmHg)", 50, 200, 80)
            sg   = st.selectbox("Specific Gravity", [1.005,1.010,1.015,1.020,1.025])
            al   = st.number_input("Albumin (0-5)", 0, 5, 0)
            su   = st.number_input("Sugar (0-5)", 0, 5, 0)
            rbc  = st.selectbox("Red Blood Cells", ["normal","abnormal"])
            pc   = st.selectbox("Pus Cell", ["normal","abnormal"])
            pcc  = st.selectbox("Pus Cell Clumps", ["notpresent","present"])
        with c2:
            ba   = st.selectbox("Bacteria", ["notpresent","present"])
            bgr  = st.number_input("Blood Glucose (mg/dL)", 50, 500, 120)
            bu   = st.number_input("Blood Urea (mg/dL)", 10, 200, 40)
            sc   = st.number_input("Serum Creatinine (mg/dL)", 0.5, 20.0, 1.2)
            sod  = st.number_input("Sodium (mEq/L)", 100, 170, 137)
            pot  = st.number_input("Potassium (mEq/L)", 2.0, 10.0, 4.5)
        with c3:
            hemo = st.number_input("Hemoglobin (g/dL)", 3.0, 20.0, 14.0)
            pcv  = st.number_input("Packed Cell Volume", 10, 60, 44)
            wc   = st.number_input("White Cell Count", 2000, 25000, 8000)
            rc   = st.number_input("Red Cell Count (millions)", 2.0, 8.0, 5.0)
            htn  = st.selectbox("Hypertension", ["no","yes"])
            dm   = st.selectbox("Diabetes Mellitus", ["no","yes"])
            cad  = st.selectbox("Coronary Artery Disease", ["no","yes"])
            appet= st.selectbox("Appetite", ["good","poor"])
            pe   = st.selectbox("Pedal Edema", ["no","yes"])
            ane  = st.selectbox("Anemia", ["no","yes"])
        submitted = st.form_submit_button("🔍 Predict Kidney Disease")

    if submitted:
        raw = dict(age=age, bp=bp, sg=sg, al=al, su=su, rbc=rbc, pc=pc, pcc=pcc,
                   ba=ba, bgr=bgr, bu=bu, sc=sc, sod=sod, pot=pot, hemo=hemo,
                   pcv=pcv, wc=wc, rc=rc, htn=htn, dm=dm, cad=cad,
                   appet=appet, pe=pe, ane=ane)
        inp = pd.DataFrame([raw])
        for col, le in le_dict.items():
            if col in inp.columns:
                val = str(inp[col].iloc[0]).strip()
                inp[col] = le.transform([val])[0] if val in le.classes_ else 0
        inp = inp[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        pred = model.predict(inp)[0]
        prob_arr = model.predict_proba(inp)[0]
        # ckd=0 in target_le means disease; check class order
        ckd_idx = list(tgt_le.classes_).index("ckd") if "ckd" in tgt_le.classes_ else 0
        prob = prob_arr[ckd_idx] * 100
        show_result(1 if pred == ckd_idx else 0, prob, "CKD Detected", "No CKD Detected")

# ─────────────────────────────────────────────────────────────
# PARKINSON'S
# ─────────────────────────────────────────────────────────────
elif page == "parkinsons":
    st.markdown('<div class="hero-title">🧠 Parkinson\'s Predictor</div>', unsafe_allow_html=True)
    with st.spinner("Training model on Parkinson's dataset..."):
        model, features, acc = train_parkinsons()
    st.markdown(f'<div class="acc-badge">Model Accuracy: {acc}%</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.info("💡 These are voice biomarker measurements. Use average values from patient voice recordings.")

    with st.form("park_form"):
        st.markdown('<div class="section-title">Voice Biomarkers</div>', unsafe_allow_html=True)
        defaults = {"MDVP:Fo(Hz)":119.99,"MDVP:Fhi(Hz)":157.30,"MDVP:Flo(Hz)":74.99,
                    "MDVP:Jitter(%)":0.00784,"MDVP:Jitter(Abs)":0.00007,"MDVP:RAP":0.0037,
                    "MDVP:PPQ":0.00554,"Jitter:DDP":0.01109,"MDVP:Shimmer":0.04374,
                    "MDVP:Shimmer(dB)":0.426,"Shimmer:APQ3":0.02182,"Shimmer:APQ5":0.0313,
                    "MDVP:APQ":0.02971,"Shimmer:DDA":0.06545,"NHR":0.02211,"HNR":21.033,
                    "RPDE":0.41478,"DFA":0.81529,"spread1":-4.813,"spread2":0.2665,
                    "D2":2.3014,"PPE":0.28465}
        vals = {}
        cols_per_row = 4
        feat_chunks = [features[i:i+cols_per_row] for i in range(0, len(features), cols_per_row)]
        for chunk in feat_chunks:
            cs = st.columns(len(chunk))
            for i, feat in enumerate(chunk):
                with cs[i]:
                    vals[feat] = st.number_input(feat, value=float(defaults.get(feat, 0.0)), format="%.5f")
        submitted = st.form_submit_button("🔍 Predict Parkinson's")

    if submitted:
        inp = pd.DataFrame([[vals[f] for f in features]], columns=features)
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0][1] * 100
        show_result(pred, prob, "Parkinson's Detected", "No Parkinson's Detected")

# ─────────────────────────────────────────────────────────────
# LIVER DISEASE
# ─────────────────────────────────────────────────────────────
elif page == "liver":
    st.markdown('<div class="hero-title">🫀 Liver Disease Predictor</div>', unsafe_allow_html=True)
    with st.spinner("Training model on liver disease dataset..."):
        model, features, acc = train_liver()
    st.markdown(f'<div class="acc-badge">Model Accuracy: {acc}%</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.form("liver_form"):
        st.markdown('<div class="section-title">Patient Lab Values</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age    = st.number_input("Age", 1, 100, 40)
            gender = st.selectbox("Gender", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
            tb     = st.number_input("Total Bilirubin", 0.0, 100.0, 0.7)
            db     = st.number_input("Direct Bilirubin", 0.0, 50.0, 0.1)
        with c2:
            alkphos = st.number_input("Alkaline Phosphotase", 50, 3000, 187)
            alamine = st.number_input("Alamine Aminotransferase", 5, 2500, 16)
            aspart  = st.number_input("Aspartate Aminotransferase", 5, 5000, 18)
        with c3:
            tp   = st.number_input("Total Proteins (g/dL)", 1.0, 15.0, 6.8)
            alb  = st.number_input("Albumin (g/dL)", 0.5, 6.0, 3.3)
            agr  = st.number_input("Albumin/Globulin Ratio", 0.1, 3.0, 0.9)
        submitted = st.form_submit_button("🔍 Predict Liver Disease")

    if submitted:
        inp  = pd.DataFrame([[age,gender,tb,db,alkphos,alamine,aspart,tp,alb,agr]], columns=features)
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0][1] * 100
        show_result(pred, prob, "Liver Disease Detected", "No Liver Disease Detected")

# ─────────────────────────────────────────────────────────────
# LUNG CANCER
# ─────────────────────────────────────────────────────────────
elif page == "lung":
    st.markdown('<div class="hero-title">🫁 Lung Cancer Predictor</div>', unsafe_allow_html=True)
    with st.spinner("Training model on lung cancer dataset..."):
        model, features, acc = train_lungcancer()
    st.markdown(f'<div class="acc-badge">Model Accuracy: {acc}%</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.form("lung_form"):
        st.markdown('<div class="section-title">Patient Information & Symptoms</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        yn = lambda label, col: col.selectbox(label, [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        with c1:
            gender  = st.selectbox("Gender", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
            age     = st.number_input("Age", 1, 120, 55)
            smoking = yn("Smoking", c1)
            yf      = yn("Yellow Fingers", c1)
        with c2:
            anxiety  = yn("Anxiety", c2)
            peer_p   = yn("Peer Pressure", c2)
            chronic  = yn("Chronic Disease", c2)
            fatigue  = yn("Fatigue", c2)
            allergy  = yn("Allergy", c2)
        with c3:
            wheeze  = yn("Wheezing", c3)
            alcohol = yn("Alcohol Consuming", c3)
            cough   = yn("Coughing", c3)
            sob     = yn("Shortness of Breath", c3)
            swallow = yn("Swallowing Difficulty", c3)
            chest   = yn("Chest Pain", c3)
        submitted = st.form_submit_button("🔍 Predict Lung Cancer Risk")

    if submitted:
        inp = pd.DataFrame([[gender,age,smoking,yf,anxiety,peer_p,chronic,fatigue,
                              allergy,wheeze,alcohol,cough,sob,swallow,chest]], columns=features)
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0][1] * 100
        show_result(pred, prob, "High Lung Cancer Risk", "Low Lung Cancer Risk")

# ─────────────────────────────────────────────────────────────
# THYROID
# ─────────────────────────────────────────────────────────────
elif page == "thyroid":
    st.markdown('<div class="hero-title">🦋 Thyroid Disease Predictor</div>', unsafe_allow_html=True)
    with st.spinner("Training model on thyroid dataset..."):
        model, features, acc = train_thyroid()
    st.markdown(f'<div class="acc-badge">Model Accuracy: {acc}%</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.form("thyroid_form"):
        st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 1, 100, 35)
            sex = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
            on_thyroxine   = st.selectbox("On Thyroxine", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            query_thyrox   = st.selectbox("Query on Thyroxine", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            on_antithyroid = st.selectbox("On Antithyroid Meds", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            sick           = st.selectbox("Sick", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            pregnant       = st.selectbox("Pregnant", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        with c2:
            thyroid_surgery = st.selectbox("Thyroid Surgery", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            I131            = st.selectbox("I131 Treatment", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            query_hypo      = st.selectbox("Query Hypothyroid", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            query_hyper     = st.selectbox("Query Hyperthyroid", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            lithium         = st.selectbox("Lithium", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            goitre          = st.selectbox("Goitre", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            tumor           = st.selectbox("Tumor", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        with c3:
            hypopituitary  = st.selectbox("Hypopituitary", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            psych          = st.selectbox("Psych", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            TSH_measured   = st.selectbox("TSH Measured", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            TSH            = st.number_input("TSH Level", 0.0, 600.0, 2.0)
            T3_measured    = st.selectbox("T3 Measured", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            T3             = st.number_input("T3 Level", 0.0, 15.0, 2.0)
            TT4_measured   = st.selectbox("TT4 Measured", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            TT4            = st.number_input("TT4 Level", 0.0, 600.0, 100.0)
            T4U_measured   = st.selectbox("T4U Measured", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            T4U            = st.number_input("T4U Level", 0.0, 3.0, 1.0)
            FTI_measured   = st.selectbox("FTI Measured", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            FTI            = st.number_input("FTI Level", 0.0, 600.0, 100.0)
            TBG_measured   = st.selectbox("TBG Measured", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            TBG            = st.number_input("TBG Level", 0.0, 200.0, 30.0)
        submitted = st.form_submit_button("🔍 Predict Thyroid Disease")

    if submitted:
        vals = [age,sex,on_thyroxine,query_thyrox,on_antithyroid,sick,pregnant,
                thyroid_surgery,I131,query_hypo,query_hyper,lithium,goitre,tumor,
                hypopituitary,psych,TSH_measured,TSH,T3_measured,T3,TT4_measured,TT4,
                T4U_measured,T4U,FTI_measured,FTI,TBG_measured,TBG]
        inp  = pd.DataFrame([vals[:len(features)]], columns=features)
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0][1] * 100
        show_result(pred, prob, "Thyroid Disease Detected", "No Thyroid Disease Detected")

# ─────────────────────────────────────────────────────────────
# ALZHEIMER'S
# ─────────────────────────────────────────────────────────────
elif page == "alzheimers":
    st.markdown('<div class="hero-title">🧬 Alzheimer\'s Predictor</div>', unsafe_allow_html=True)
    with st.spinner("Training model on Alzheimer's dataset..."):
        model, features, acc = train_alzheimers()
    st.markdown(f'<div class="acc-badge">Model Accuracy: {acc}%</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.form("alz_form"):
        st.markdown('<div class="section-title">Patient Profile</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age        = st.number_input("Age", 1, 120, 70)
            gender     = st.selectbox("Gender", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
            ethnicity  = st.number_input("Ethnicity (encoded)", 0, 3, 0)
            edu_level  = st.number_input("Education Level (0-3)", 0, 3, 2)
            bmi        = st.number_input("BMI", 10.0, 50.0, 25.0)
            smoking    = st.selectbox("Smoking", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            alcohol    = st.number_input("Alcohol Consumption", 0.0, 30.0, 5.0)
            phys_act   = st.number_input("Physical Activity (hrs/wk)", 0.0, 15.0, 5.0)
        with c2:
            diet_qual  = st.number_input("Diet Quality (0-10)", 0.0, 10.0, 5.0)
            sleep_qual = st.number_input("Sleep Quality (0-10)", 0.0, 10.0, 7.0)
            fam_hist   = st.selectbox("Family History Alzheimer's", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            cardio     = st.selectbox("Cardiovascular Disease", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            diabetes   = st.selectbox("Diabetes", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            depression = st.selectbox("Depression", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            head_inj   = st.selectbox("Head Injury", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            hyper      = st.selectbox("Hypertension", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        with c3:
            sys_bp     = st.number_input("Systolic BP", 80, 220, 130)
            dia_bp     = st.number_input("Diastolic BP", 50, 150, 80)
            chol_tot   = st.number_input("Total Cholesterol", 100, 400, 200)
            chol_ldl   = st.number_input("LDL Cholesterol", 30, 300, 100)
            chol_hdl   = st.number_input("HDL Cholesterol", 10, 120, 50)
            chol_trig  = st.number_input("Triglycerides", 50, 600, 150)
            mmse       = st.number_input("MMSE Score (0-30)", 0.0, 30.0, 25.0)
            func_assess= st.number_input("Functional Assessment (0-10)", 0.0, 10.0, 8.0)
            mem_comp   = st.selectbox("Memory Complaints", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            beh_prob   = st.selectbox("Behavioral Problems", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            adl        = st.number_input("ADL Score (0-10)", 0.0, 10.0, 8.0)
            confusion  = st.selectbox("Confusion", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            disorientation = st.selectbox("Disorientation", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            personality= st.selectbox("Personality Changes", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            diff_tasks = st.selectbox("Difficulty Completing Tasks", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            forgetful  = st.selectbox("Forgetfulness", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        submitted = st.form_submit_button("🔍 Predict Alzheimer's Risk")

    if submitted:
        vals_map = dict(Age=age, Gender=gender, Ethnicity=ethnicity, EducationLevel=edu_level,
                        BMI=bmi, Smoking=smoking, AlcoholConsumption=alcohol,
                        PhysicalActivity=phys_act, DietQuality=diet_qual, SleepQuality=sleep_qual,
                        FamilyHistoryAlzheimers=fam_hist, CardiovascularDisease=cardio,
                        Diabetes=diabetes, Depression=depression, HeadInjury=head_inj,
                        Hypertension=hyper, SystolicBP=sys_bp, DiastolicBP=dia_bp,
                        CholesterolTotal=chol_tot, CholesterolLDL=chol_ldl,
                        CholesterolHDL=chol_hdl, CholesterolTriglycerides=chol_trig,
                        MMSE=mmse, FunctionalAssessment=func_assess,
                        MemoryComplaints=mem_comp, BehavioralProblems=beh_prob,
                        ADL=adl, Confusion=confusion, Disorientation=disorientation,
                        PersonalityChanges=personality, DifficultyCompletingTasks=diff_tasks,
                        Forgetfulness=forgetful)
        row = [vals_map.get(f, 0) for f in features]
        inp  = pd.DataFrame([row], columns=features)
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0][1] * 100
        show_result(pred, prob, "Alzheimer's Risk Detected", "Low Alzheimer's Risk")

# ─────────────────────────────────────────────────────────────
# DENGUE
# ─────────────────────────────────────────────────────────────
elif page == "dengue":
    st.markdown('<div class="hero-title">🦟 Dengue Predictor</div>', unsafe_allow_html=True)
    with st.spinner("Training SVM (Linear) model on dengue dataset..."):
        model, features, acc = train_dengue()
    st.markdown(f'<div class="acc-badge">Model Accuracy: {acc}%</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.form("dengue_form"):
        st.markdown('<div class="section-title">Patient Symptoms</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: fever     = st.selectbox("Fever",     [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        with c2: headache  = st.selectbox("Headache",  [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        with c3: joint_pain= st.selectbox("Joint Pain",[0,1], format_func=lambda x: "No" if x==0 else "Yes")
        with c4: bleeding  = st.selectbox("Bleeding",  [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        submitted = st.form_submit_button("🔍 Predict Dengue")

    if submitted:
        inp  = pd.DataFrame([[fever, headache, joint_pain, bleeding]], columns=features)
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0][1] * 100
        show_result(pred, prob, "Dengue Positive", "Dengue Negative")

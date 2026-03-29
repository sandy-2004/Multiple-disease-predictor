# 🏥 MediPredict — Multi-Disease Prediction System

AI-powered prediction for **9 diseases** using Machine Learning.

## Diseases Covered
| Disease | Model | Dataset |
|---|---|---|
| 🩸 Diabetes | Random Forest | diabetes.csv |
| ❤️ Heart Disease | Gradient Boosting | heart_disease.csv |
| 🫘 Kidney Disease | Random Forest | kidney_disease.csv |
| 🧠 Parkinson's | SVM (RBF) | parkinsons.csv |
| 🫀 Liver Disease | Random Forest | indian_liver_patient.csv |
| 🫁 Lung Cancer | Gradient Boosting | survey_lung_cancer.csv |
| 🦋 Thyroid | Random Forest | thyroidDF.csv |
| 🧬 Alzheimer's | Gradient Boosting | alzheimers_disease_data.csv |
| 🦟 Dengue | SVM (Linear) | dengue.csv |

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place all CSV files in the same folder as app.py
```
project/
├── app.py
├── requirements.txt
├── diabetes.csv
├── heart_disease.csv
├── kidney_disease.csv
├── parkinsons.csv
├── indian_liver_patient.csv
├── survey_lung_cancer.csv
├── thyroidDF.csv
├── alzheimers_disease_data.csv
└── dengue.csv
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

## Deploy to Streamlit Cloud (Free)

1. Push all files to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — live in ~2 minutes!

## Deploy to Hugging Face Spaces (Free)

1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Streamlit** as the SDK
3. Upload all files
4. App auto-deploys!

## Notes
- Models train once on first load and are cached
- All predictions are for **educational purposes only**
- Always consult a qualified medical professional
- Dengue model uses LinearSVC (handles 1M rows efficiently)

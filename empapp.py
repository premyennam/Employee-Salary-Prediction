import streamlit as st
import pandas as pd
import joblib

# Load model pipeline (includes preprocessing)
try:
    model = joblib.load("best_model.pkl")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# App config
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar input form
st.sidebar.header("📋 Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
educational_num = st.sidebar.slider("Education Number", 1, 16, 10)
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=0, value=100000)
race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
])
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay"
])
marital_status = st.sidebar.selectbox("Marital Status", [
    "Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent"
])
relationship = st.sidebar.selectbox("Relationship", [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "India", "Mexico", "Philippines", "Germany", "Canada", "England", "Italy", "Other"
])

# Input DataFrame
input_df = pd.DataFrame([{
    'age': age,
    'education': education,
    'educational-num': educational_num,
    'occupation': occupation,
    'hours-per-week': hours_per_week,
    'experience': experience,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'fnlwgt': fnlwgt,
    'race': race,
    'workclass': workclass,
    'marital-status': marital_status,
    'relationship': relationship,
    'gender': gender,
    'native-country': native_country
}])

st.write("### 🔍 Input Summary")
st.write(input_df)

# Predict single input
if st.button("🔎 Predict Salary Class"):
    try:
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        st.success(f"✅ Prediction: {prediction} (Confidence: {max(prob)*100:.2f}%)")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# Batch prediction
st.markdown("---")
st.subheader("📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:")
        st.write(batch_data.head())

        preds = model.predict(batch_data)
        batch_data["PredictedClass"] = preds

        st.success("✅ Batch prediction complete.")
        st.write(batch_data.head())

        csv = batch_data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "predicted_output.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Failed to process batch file: {e}")

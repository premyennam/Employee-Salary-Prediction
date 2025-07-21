 💼 Employee Salary Prediction

This project predicts whether an employee earns more than 50K or less/equal based on various personal and professional attributes. It uses the UCI Adult Income dataset and is deployed via Streamlit.

---

## 📊 Features

- Predict salary class (>50K or ≤50K) based on:
  - Age, education, occupation, work hours, experience
  - Workclass, relationship, marital status, gender, race
  - Capital gain/loss, education-num, native country, etc.
- Supports both **single input** and **batch CSV predictions**
- Built with a **scikit-learn pipeline**, including preprocessing
- Interactive **Streamlit app** frontend

---

## 📁 Project Structure

├── app.py # Streamlit app
├── best_model.pkl # Trained ML model pipeline
├── feature_columns.pkl # Saved column order for app alignment
├── employee_salary_prediction.ipynb # Notebook for training
├── adult.csv # Dataset (UCI Adult Income)
├── requirements.txt
├── .gitignore
└── README.md

yaml
Copy code

---

## 🚀 How to Run Locally

### 1. Clone the repository

git clone https://github.com/your-username/employee-salary-prediction.git
cd employee-salary-prediction

2. Create virtual environment (recommended)

Copy code
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

3. Install dependencies
bash
Copy code
pip install -r requirements.txt

5. Run the Streamlit App
bash
Copy code
streamlit run app.py

Model Overview:

Models evaluated: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN

Final model: Best-performing classifier with full preprocessing pipeline

Feature engineering includes:

Grouping low-frequency countries

Label encoding + one-hot encoding

Scaling for numeric features

Requirements
See requirements.txt

Dataset Source
UCI Adult Income Dataset: https://archive.ics.uci.edu/ml/datasets/adult

License
This project is open-source and free to use under the MIT License.

Acknowledgments
UCI Machine Learning Repository

scikit-learn, Streamlit, pandas, seaborn



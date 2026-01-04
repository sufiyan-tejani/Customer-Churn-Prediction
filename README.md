# Customer Churn Prediction System (End-to-End Machine Learning Project)

---

**Live Demo (Streamlit App):** https://sufiyan-tejani-customer-churn-prediction-app-upt8yw.streamlit.app/
## Overview
This project implements a complete end-to-end machine learning pipeline to predict customer churn using demographic and service-related data. It follows industry-standard practices including data preprocessing, exploratory data analysis, model training, hyperparameter tuning, and deployment through an interactive web application.

The system helps identify customers who are likely to churn, enabling businesses to take proactive retention actions.

---

## Objective
To build, evaluate, and deploy a machine learning model that predicts customer churn and provides real-time predictions through a user-friendly web interface.

---

## Project Workflow
1. Data ingestion and validation  
2. Exploratory Data Analysis (EDA)  
3. Feature engineering and preprocessing  
4. Model training and hyperparameter tuning  
5. Model evaluation and selection  
6. Model serialization  
7. Deployment using Streamlit  

---

## Dataset
- **Type:** Structured customer data  
- **Features:** Age, Gender, Tenure, Monthly Charges  
- **Target Variable:** Churn (Yes / No)

---

## Exploratory Data Analysis (EDA)
- Analyzed churn distribution across different customer segments  
- Studied relationships between churn and key numerical features  
- Used visualizations and statistical summaries to extract insights  

---

## Feature Engineering & Preprocessing
- Handled missing and duplicate values  
- Converted categorical features into numerical representations  
- Applied feature scaling using `StandardScaler`  
- Prevented data leakage by fitting preprocessing steps only on training data  

---

## Machine Learning Models
The following classification models were trained and evaluated:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  

### Model Optimization
- Hyperparameter tuning performed using `GridSearchCV`  
- Cross-validation used to ensure robust performance  
- Best-performing model selected based on accuracy  

---

## Model Evaluation
- Evaluated models on unseen test data  
- Compared performance across multiple algorithms  
- Selected the final model based on accuracy and generalization capability  

---

## Deployment
- Developed an interactive **Streamlit web application**  
- Users can input customer details and receive real-time churn predictions  
- Ensured consistent preprocessing and inference using saved model artifacts  

---

## Project Structure
├── churn_model.ipynb # Complete EDA, preprocessing, and model training
├── customer_churn_data.csv # Dataset
├── scaler.pkl # Saved feature scaler
├── model.pkl # Trained machine learning model
├── app.py # Streamlit web application
├── README.md # Project documentation

yaml
Copy code

---

## Tech Stack
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Deployment:** Streamlit  
- **Tools:** Git, GitHub, Jupyter Notebook, Joblib  

---

## Key Learnings
- Building production-ready machine learning pipelines  
- Preventing data leakage during preprocessing  
- Comparing and tuning multiple ML models  
- Deploying ML models for real-time predictions  

---

## Future Improvements
- Handle class imbalance using advanced sampling techniques  
- Add model explainability using SHAP or feature importance  
- Evaluate models using precision, recall, and ROC-AUC  
- Deploy on cloud platforms such as AWS or Streamlit Cloud  

---

## Author
**Sufiyan Ashraf Tejani**  
M.Sc. Mathematics, IIT Kharagpur 

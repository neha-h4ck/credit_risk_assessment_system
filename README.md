# CREDIT RISK ASSESSMENT SYSTEM 💳

A professional, end-to-end Machine Learning project designed to analyze customer financial and credit profile information, predict default probabilities, categorize applicants into risk tiers (**Low Risk**, **Medium Risk**, **High Risk**), and issue automated credit decision recommendations (**Approve**, **Manual Review**, **Reject**).

---

## 1. Project Title
**CREDIT RISK ASSESSMENT SYSTEM**  
*Machine Learning Powered Credit Evaluation & Risk Decision Engine*

---

## 2. Project Objective
The primary objective of this system is to evaluate retail credit loan applicants by applying statistical machine learning algorithms to historical financial and credit behavioral data. The system:
- Estimates the probability of an applicant defaulting on credit (**Bad Credit**).
- Classifies applicants into transparent risk tiers (**Low Risk**, **Medium Risk**, **High Risk**).
- Provides an automated credit decision recommendation (**Approve**, **Manual Review**, **Reject**).
- Delivers an interactive Streamlit UI dashboard and reproducible CLI pipelines for local IDE (VS Code / Antigravity IDE) and Google Colab environments.

---

## 3. Problem Statement
Credit lending institutions face significant financial risk when extending loans to individuals without clear predictive indicators of repayment likelihood. Traditional credit scoring rules can be rigid and fail to capture complex non-linear feature interactions. Machine learning offers a data-driven method to calculate default probabilities, reduce non-performing loans (NPLs), streamline loan processing speed, and provide standardized risk scoring.

---

## 4. Key Features
- **Data Preprocessing Pipeline**: ColumnTransformer with StandardScaler for numerical features, OneHotEncoder for categorical attributes, and missing value imputation—fitted strictly on training data to prevent data leakage.
- **Primary Classification Model**: Logistic Regression with balanced class weighting (`class_weight='balanced'`) for maximum interpretability.
- **Model Comparison**: Benchmark comparison against Random Forest Classifier based on ROC-AUC and F1-Score metrics.
- **Configurable Risk Scoring**: Probability-driven risk tiers (< 0.30 Low Risk, 0.30–0.60 Medium Risk, >= 0.60 High Risk).
- **Interactive Streamlit Web App**: Real-time single applicant form assessor, batch evaluator, and comprehensive model analytics dashboard.
- **Hypothetical Profile Testing**: Pre-configured test runner (`sample_test.py`) demonstrating predictions on 5 diverse customer profiles.
- **Executable Jupyter Notebook**: Standalone analysis notebook for experimentation.

---

## 5. Technology Stack
- **Language**: Python 3
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Serialization**: Joblib / Pickle
- **Visualizations**: Matplotlib, Seaborn, Plotly
- **Web Application**: Streamlit

---

## 6. Dataset Description
The dataset is structured after the benchmark **UCI Statlog German Credit Dataset** (1,000 applicant records) mapping to 15 key features:

| Feature Name | Feature Type | Description / Values |
|---|---|---|
| `age` | Numerical | Applicant age in years (18–100) |
| `employment_status` | Categorical | Employment duration (`Unemployed`, `< 1 Year`, `1 to 4 Years`, `4 to 7 Years`, `>= 7 Years`) |
| `housing` | Categorical | Housing ownership (`Rent`, `Own`, `Free`) |
| `dependents` | Numerical | Number of maintenance dependents (1, 2) |
| `credit_history` | Categorical | Rating (`Critical Account`, `Delay in Payment`, `Existing Credits Paid`, `All Credits Paid`, `No Credits`) |
| `credit_amount` | Numerical | Credit loan amount requested in currency units |
| `loan_duration` | Numerical | Loan duration in months (4–72) |
| `purpose` | Categorical | Purpose (`New Car`, `Used Car`, `Furniture/Equipment`, `Radio/TV`, `Repairs`, `Education`, `Business`, `Others`) |
| `existing_credits` | Numerical | Number of existing credits at this bank (1–4) |
| `other_payment_plans`| Categorical | Other installment plans (`Bank`, `Stores`, `None`) |
| `savings_status` | Categorical | Financial reserves (`< 100 DM`, `100 to 500 DM`, `500 to 1000 DM`, `>= 1000 DM`, `Unknown/No Savings`) |
| `installment_rate` | Numerical | Installment rate in % of disposable income (1, 2, 3, 4) |
| `property` | Categorical | Assets (`Real Estate`, `Building Society Savings`, `Car or Other`, `Unknown/No Property`) |
| `checking_status` | Categorical | Checking account balance (`< 0 DM`, `0 to 200 DM`, `>= 200 DM`, `No Checking Account`) |
| `target` | Binary Target | Target credit risk (`0 = Good Credit`, `1 = Bad Credit`) |

> **Financial Capacity Note**: Continuous raw monthly income is not recorded as a separate column in the standard German Credit Dataset. Financial burden is captured through `installment_rate` (% of disposable income), `savings_status`, `checking_status`, and `employment_status`.

---

## 7. Machine Learning Methodology
1. **Supervised Binary Classification**: The target variable $y \in \{0, 1\}$ represents Good Credit ($0$) vs Bad Credit ($1$).
2. **Logistic Regression (Primary)**: Modeled using log-odds optimization with balanced class weighting to handle class imbalance:
   $$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots + \beta_p X_p)}}$$
3. **Interpretability**: Logistic Regression coefficients indicate the direction and relative log-odds association of each preprocessed feature with default probability.

---

## 8. Data Preprocessing
Preprocessing is constructed using scikit-learn `Pipeline` and `ColumnTransformer`:
- **Numerical Features**: Scaled using `StandardScaler` ($\mu=0, \sigma=1$) after median imputation.
- **Categorical Features**: Encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` after mode imputation.
- **Data Leakage Prevention**: Preprocessing parameters are fitted **exclusively on the training split** (`X_train`) and applied downstream to `X_test` and new applicant inference.

---

## 9. Model Training
Execute stratified train-test splitting (80% training, 20% testing):
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
Model artifacts (`credit_risk_model.pkl`, `preprocessor.pkl`, `scaler.pkl`) are automatically serialized into the `models/` folder using `joblib`.

---

## 10. Model Evaluation
The trained model produces evaluation metrics on unseen test data:
- **Accuracy**: Overall prediction correctness.
- **Precision**: Proportion of true bad credit defaults among predicted defaults.
- **Recall**: Proportion of actual bad credit defaults correctly identified.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve.

---

## 11. Risk Classification Methodology
The estimated probability of bad credit ($P_{\text{bad}}$) is mapped to configurable risk tiers:

| Probability Range ($P_{\text{bad}}$) | Risk Tier | Indicator | Credit Decision Recommendation |
|---|---|---|---|
| $P_{\text{bad}} < 0.30$ | **LOW RISK** | 🟢 Green | **APPROVE** |
| $0.30 \le P_{\text{bad}} < 0.60$ | **MEDIUM RISK** | 🟡 Yellow | **MANUAL REVIEW** |
| $P_{\text{bad}} \ge 0.60$ | **HIGH RISK** | 🔴 Red | **REJECT** |

---

## 12. Project Architecture
```text
credit-risk-assessment/
│
├── data/
│   └── credit_data.csv             # 1,000 applicant benchmark dataset
│
├── models/
│   ├── credit_risk_model.pkl       # Trained Logistic Regression classifier
│   ├── preprocessor.pkl            # Fitted ColumnTransformer preprocessor
│   └── scaler.pkl                  # Fitted StandardScaler object
│
├── notebooks/
│   └── credit_risk_analysis.ipynb  # Interactive Jupyter Notebook
│
├── src/
│   ├── generate_data.py            # Dataset generation script
│   ├── data_preprocessing.py       # Data loading & preprocessing module
│   ├── train_model.py              # Model training & artifact serialization
│   ├── predict.py                  # Single & batch prediction engine
│   └── evaluation.py               # Evaluation metrics & visualization module
│
├── app.py                          # Streamlit web application
├── sample_test.py                  # 5-profile hypothetical test script
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
└── .gitignore                      # Git ignore file
```

---

## 13. Installation Instructions

1. **Clone or Navigate to the Project Directory**:
   ```bash
   cd credit-risk-assessment
   ```

2. **Create and Activate a Python Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # On macOS/Linux
   # venv\Scripts\activate          # On Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 14. How to Train the Model
To execute the data preprocessing, model training, evaluation, and artifact saving pipeline:
```bash
python src/train_model.py
```
This updates all model binaries inside the `models/` folder.

---

## 15. How to Run the Streamlit Application
To launch the interactive web dashboard:
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## 16. Example Prediction
Run the sample testing script to evaluate 5 hypothetical applicant profiles:
```bash
python sample_test.py
```

**Example Output**:
```text
📌 Profile 1: Strong Financial Profile
Age: 48 | Employment: >= 7 Years | Housing: Own
Credit History: All Credits Paid | Amount: $1,500 | Duration: 12 mos

Prediction Output:
  Risk Level            : 🟢 LOW RISK
  Probability Bad Credit: 21.8%
  Probability Good Credit: 78.2%
  Credit Decision       : APPROVE
```

---

## 17. Limitations
- **Dataset Scope**: Based on the 1,000-instance German Credit benchmark; real enterprise deployment requires hundreds of thousands of records across economic cycles.
- **Feature Exclusions**: Does not contain real-time macro-economic signals or open-banking transactional data.
- **Linear Decision Boundaries**: Logistic Regression assumes linear log-odds relationships.

---

## 18. Ethical Considerations
- **Non-Discrimination**: Protected attributes (race, religion, gender) should be excluded from credit decision models to prevent biased lending.
- **Explainability**: Applicants rejected by automated models have a right to clear explanations of key adverse risk drivers.
- **Human Oversight**: High Risk and Medium Risk recommendations should be subject to human credit officer review before final binding decisioning.

---

## 19. Future Enhancements
- **XAI Integration**: Integrate SHAP (SHapley Additive exPlanations) and LIME for local feature attribution on individual credit decisions.
- **Advanced Model Ensembles**: Benchmark XGBoost and LightGBM models against Logistic Regression.
- **API Deployment**: Wrap prediction engine into a RESTful FastAPI service with Docker containerization.

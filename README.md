# Online Exam Pass Prediction

A production-style machine learning project that predicts whether a student is likely to pass an online exam using academic and behavioral indicators such as study hours, previous score, attendance, assignment performance, sleep quality, and participation in online classes.

This project is built as a lightweight, interview-ready Streamlit dashboard that demonstrates end-to-end ML workflow: synthetic dataset generation, EDA, preprocessing, logistic regression modeling, evaluation, feature importance analysis, and live prediction.

## Why this project matters

Educational institutions increasingly need proactive intervention tools that identify at-risk students early. Instead of relying only on final grades, this project estimates pass probability from measurable inputs and helps answer questions like:

- Which students need academic support?
- Which features most strongly influence pass/fail outcomes?
- How likely is a student to pass based on current performance signals?

## Project highlights

- Binary classification problem using Logistic Regression
- StandardScaler + LogisticRegression pipeline
- Live user input prediction in Streamlit
- Model evaluation with accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and classification report
- Interactive dashboard with charts and feature analysis
- Easy to run locally without any cloud services

## Tech stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Business use case

This solution can be used by:

- Student support teams to flag at-risk learners
- Academic advisors to prioritize intervention strategies
- Online learning platforms to detect likely course completion risk
- Education researchers studying performance drivers

## Dataset

The project uses a synthetic educational dataset with the following features:

- study_hours
- previous_score
- attendance
- assignment_score
- practice_tests
- online_participation
- sleep_hours
- exam_result (target: 0 = Fail, 1 = Pass)

The dataset is intentionally realistic and structured to show a complete ML lifecycle while remaining lightweight and easy to understand.

## Modeling approach

The project implements a clean pipeline:

1. Load or generate the dataset
2. Handle missing values using median imputation
3. Split data into train and test sets
4. Standardize features using StandardScaler
5. Train a Logistic Regression classifier
6. Evaluate the model on unseen data
7. Visualize feature importance and confusion metrics
8. Use the trained model to predict pass/fail for new student inputs

## Key technical decisions

- Logistic Regression was chosen because it is interpretable, fast, and well-suited for binary classification
- StandardScaler improves model stability when feature magnitudes differ significantly
- Missing values are handled with median imputation to simulate real data quality issues
- Feature coefficients are interpreted to explain which attributes contribute most to passing

## Project structure

```text
ONLINE-EXAM-PASS-PREDICTION-Logistic-Regression/
├── app.py                      # Main Streamlit application
├── data/
│   └── online_exam_data.csv    # Sample student dataset
├── models/
│   └── logistic_regression.pkl # Saved trained model (created at runtime)
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── .gitignore                 # Optional local ignores
```

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/sgsinghashka-del/ONLINE-EXAM-PASS-PREDICTION-Logistic-Regression.git
cd ONLINE-EXAM-PASS-PREDICTION-Logistic-Regression
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

## Features of the Streamlit app

The dashboard includes four major sections:

- Dashboard: project overview, dataset summary, and EDA
- Model Evaluation: accuracy, ROC-AUC, confusion matrix, classification report
- Feature Analysis: logistic regression coefficients and variable impact
- Prediction: live form to enter student values and get pass/fail probability

## Example of model interpretation

Positive coefficients suggest that increasing a feature raises the probability of passing. For example, higher study hours, previous score, attendance, assignment performance, and practice tests generally improve the predicted result.

This interpretable model is valuable in interviews because it demonstrates not only predictive power but also explainability and business understanding.

## Sample output

The app predicts a probability between 0 and 1, where:

- 0.75 to 1.00: strong likelihood of passing
- 0.50 to 0.74: uncertain but plausible outcome
- below 0.50: lower likelihood of passing

## Future enhancements

To make this project stronger for interviews or production use, these additions would be valuable:

- Real student data integration instead of synthetic generation
- Hyperparameter tuning with GridSearchCV
- Explainable AI using SHAP or feature contribution plots
- Deployment through Streamlit Cloud, Heroku, or Docker
- Model monitoring and retraining pipeline
- API endpoint for model inference

## Interview-ready summary

This project demonstrates a complete machine learning workflow in a practical, business-relevant domain: predicting exam success using student behavior and academic performance. It combines data science, business understanding, model evaluation, and visualization into a clean, interactive application that is easy to explain and impressive in presentations.

## License

This project is intended for educational and portfolio use.

---

If you want, I can also create a stronger version of this project with a proper `requirements.txt`, CSV dataset, and an interview-oriented `architecture.md` or `presentation script`.

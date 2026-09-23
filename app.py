```python
"""
ONLINE EXAM PASS PREDICTION
Machine Learning Project using Logistic Regression

"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Online Exam Pass Predictor",
    page_icon="🎓",
    layout="wide",
)

DATA_DIR = "data"
MODEL_DIR = "models"

DATA_FILE = os.path.join(DATA_DIR, "online_exam_data.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "logistic_regression.pkl")


FEATURES = [
    "study_hours",
    "previous_score",
    "attendance",
    "assignment_score",
    "practice_tests",
    "online_participation",
    "sleep_hours",
]

TARGET = "exam_result"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }

    .pass-box {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }

    .fail-box {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(n_samples=1500, random_state=42):
    """
    Generate a realistic synthetic online-exam dataset.
    """

    np.random.seed(random_state)

    study_hours = np.random.uniform(1, 10, n_samples)
    previous_score = np.random.uniform(35, 95, n_samples)
    attendance = np.random.uniform(45, 100, n_samples)
    assignment_score = np.random.uniform(40, 100, n_samples)
    practice_tests = np.random.randint(0, 11, n_samples)
    online_participation = np.random.uniform(40, 100, n_samples)
    sleep_hours = np.random.uniform(4, 9, n_samples)

    # Generate a realistic probability score.
    score = (
        0.35 * study_hours
        + 0.045 * previous_score
        + 0.025 * attendance
        + 0.025 * assignment_score
        + 0.25 * practice_tests
        + 0.02 * online_participation
        + 0.20 * sleep_hours
        + np.random.normal(0, 1.5, n_samples)
    )

    # Convert continuous score into pass/fail.
    exam_result = (score >= 8.0).astype(int)

    df = pd.DataFrame(
        {
            "study_hours": study_hours.round(2),
            "previous_score": previous_score.round(2),
            "attendance": attendance.round(2),
            "assignment_score": assignment_score.round(2),
            "practice_tests": practice_tests,
            "online_participation": online_participation.round(2),
            "sleep_hours": sleep_hours.round(2),
            "exam_result": exam_result,
        }
    )

    return df


# ============================================================
# LOAD / CREATE DATA
# ============================================================

def load_data():
    """
    Load existing dataset or create one if it does not exist.
    """

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = generate_dataset()

        # Add a small amount of missing data to demonstrate
        # preprocessing.
        np.random.seed(42)

        for column in FEATURES:
            missing_indices = np.random.choice(
                df.index,
                size=int(len(df) * 0.01),
                replace=False,
            )
            df.loc[missing_indices, column] = np.nan

        df.to_csv(DATA_FILE, index=False)

    return df


# ============================================================
# DATA PREPROCESSING
# ============================================================

def preprocess_data(df):
    """
    Clean missing values and prepare X/y.
    """

    data = df.copy()

    # Fill numeric missing values with median.
    for column in FEATURES:
        data[column] = data[column].fillna(data[column].median())

    data[TARGET] = data[TARGET].fillna(data[TARGET].mode()[0])

    X = data[FEATURES]
    y = data[TARGET]

    return X, y, data


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(df):
    """
    Train Logistic Regression with StandardScaler.
    """

    X, y, clean_df = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_probability),
    }

    cm = confusion_matrix(y_test, y_pred)

    return (
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        y_probability,
        metrics,
        cm,
        clean_df,
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):
    """
    Save trained model to models directory.
    """

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

def get_coefficients(model):
    """
    Extract Logistic Regression coefficients.
    """

    logistic_model = model.named_steps["logistic_regression"]

    coefficients = logistic_model.coef_[0]

    coefficient_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Coefficient": coefficients,
            "Absolute Impact": np.abs(coefficients),
        }
    )

    coefficient_df = coefficient_df.sort_values(
        "Absolute Impact",
        ascending=False,
    )

    return coefficient_df


# ============================================================
# EDA
# ============================================================

def show_eda(df):
    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Exam Result Distribution")

        result_counts = df[TARGET].value_counts().rename(
            index={
                0: "Fail",
                1: "Pass",
            }
        )

        fig, ax = plt.subplots()
        ax.bar(
            result_counts.index,
            result_counts.values,
        )
        ax.set_ylabel("Number of Students")
        ax.set_xlabel("Exam Result")
        ax.set_title("Pass vs Fail")

        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.write("### Study Hours vs Previous Score")

        fig, ax = plt.subplots()

        for result, label in [(0, "Fail"), (1, "Pass")]:
            subset = df[df[TARGET] == result]

            ax.scatter(
                subset["study_hours"],
                subset["previous_score"],
                label=label,
                alpha=0.5,
            )

        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Previous Score")
        ax.set_title("Study Hours vs Previous Score")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

    st.write("### Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def show_evaluation(
    model,
    y_test,
    y_pred,
    y_probability,
    metrics,
    cm,
):
    st.subheader("📈 Model Evaluation")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.2%}",
    )

    col2.metric(
        "Precision",
        f"{metrics['precision']:.2%}",
    )

    col3.metric(
        "Recall",
        f"{metrics['recall']:.2%}",
    )

    col4.metric(
        "F1 Score",
        f"{metrics['f1']:.2%}",
    )

    col5.metric(
        "ROC-AUC",
        f"{metrics['roc_auc']:.2%}",
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    st.write("### Confusion Matrix")

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Fail", "Pass"])
    ax.set_yticklabels(["Fail", "Pass"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # ROC Curve
    # --------------------------------------------------------

    st.write("### ROC Curve")

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability,
    )

    fig, ax = plt.subplots()

    ax.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {metrics['roc_auc']:.3f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    st.write("### Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Fail", "Pass"],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True,
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(model):
    st.subheader("🔎 Feature Impact")

    coefficient_df = get_coefficients(model)

    st.dataframe(
        coefficient_df.round(4),
        use_container_width=True,
    )

    st.info(
        "Positive Logistic Regression coefficients increase the "
        "model's predicted probability of passing, while negative "
        "coefficients decrease it. Coefficients are based on "
        "standardized features."
    )

    fig, ax = plt.subplots()

    sorted_df = coefficient_df.sort_values(
        "Coefficient"
    )

    ax.barh(
        sorted_df["Feature"],
        sorted_df["Coefficient"],
    )

    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title("Feature Coefficients")

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_student(
    model,
    study_hours,
    previous_score,
    attendance,
    assignment_score,
    practice_tests,
    online_participation,
    sleep_hours,
):
    input_data = pd.DataFrame(
        [
            {
                "study_hours": study_hours,
                "previous_score": previous_score,
                "attendance": attendance,
                "assignment_score": assignment_score,
                "practice_tests": practice_tests,
                "online_participation": online_participation,
                "sleep_hours": sleep_hours,
            }
        ]
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    return prediction, probability


# ============================================================
# STUDENT PREDICTION UI
# ============================================================

def show_prediction(model):
    st.subheader("🎓 Predict Student Exam Result")

    st.write(
        "Enter the student's academic and participation information "
        "to predict the probability of passing."
    )

    col1, col2 = st.columns(2)

    with col1:

        study_hours = st.slider(
            "Study Hours per Day",
            min_value=0.0,
            max_value=12.0,
            value=5.0,
            step=0.5,
        )

        previous_score = st.slider(
            "Previous Exam Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0,
        )

        attendance = st.slider(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
        )

        assignment_score = st.slider(
            "Assignment Score (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
        )

    with col2:

        practice_tests = st.slider(
            "Practice Tests Completed",
            min_value=0,
            max_value=15,
            value=5,
            step=1,
        )

        online_participation = st.slider(
            "Online Class Participation (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
        )

        sleep_hours = st.slider(
            "Average Sleep Hours",
            min_value=2.0,
            max_value=12.0,
            value=7.0,
            step=0.5,
        )

    if st.button(
        "🔮 Predict Exam Result",
        type="primary",
        use_container_width=True,
    ):

        prediction, probability = predict_student(
            model,
            study_hours,
            previous_score,
            attendance,
            assignment_score,
            practice_tests,
            online_participation,
            sleep_hours,
        )

        probability_percent = probability * 100

        if prediction == 1:

            st.markdown(
                f"""
                <div class="prediction-box pass-box">
                    <h2>✅ PASS</h2>
                    <h3>Probability of Passing: {probability_percent:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="prediction-box fail-box">
                    <h2>❌ FAIL</h2>
                    <h3>Probability of Passing: {probability_percent:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("### Prediction Interpretation")

        if probability >= 0.75:

            st.success(
                "The model estimates a high probability of passing."
            )

        elif probability >= 0.50:

            st.warning(
                "The prediction is relatively uncertain. "
                "Additional preparation may improve the outcome."
            )

        else:

            st.error(
                "The model estimates a lower probability of passing."
            )


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar(df, metrics):

    st.sidebar.title("🎓 Online Exam Predictor")

    st.sidebar.markdown(
        """
        ### Machine Learning Model

        **Algorithm:** Logistic Regression

        **Problem:** Binary Classification

        **Target:**
        - 0 = Fail
        - 1 = Pass

        **Preprocessing:**
        - Missing value handling
        - StandardScaler
        - Train/Test Split
        """
    )

    st.sidebar.markdown("---")

    st.sidebar.write(
        f"**Dataset Rows:** {len(df):,}"
    )

    st.sidebar.write(
        f"**Features:** {len(FEATURES)}"
    )

    st.sidebar.write(
        f"**Accuracy:** {metrics['accuracy']:.2%}"
    )

    st.sidebar.write(
        f"**ROC-AUC:** {metrics['roc_auc']:.2%}"
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.markdown(
        '<div class="main-title">🎓 Online Exam Pass Prediction</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Predict whether a student is likely to pass an online exam "
        "using Logistic Regression."
        "</div>",
        unsafe_allow_html=True,
    )

    # Load dataset.
    df = load_data()

    # Train model.
    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        y_probability,
        metrics,
        cm,
        clean_df,
    ) = train_model(df)

    # Save trained model.
    save_model(model)

    # Sidebar.
    show_sidebar(
        clean_df,
        metrics,
    )

    # Tabs.
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Dashboard",
            "🤖 Model Evaluation",
            "🔎 Feature Analysis",
            "🎓 Prediction",
        ]
    )

    # --------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------

    with tab1:

        st.header("Project Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Students",
            f"{len(clean_df):,}",
        )

        col2.metric(
            "Training Samples",
            f"{len(X_train):,}",
        )

        col3.metric(
            "Testing Samples",
            f"{len(X_test):,}",
        )

        col4.metric(
            "Pass Rate",
            f"{clean_df[TARGET].mean():.2%}",
        )

        st.markdown("---")

        show_eda(clean_df)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with tab2:

        show_evaluation(
            model,
            y_test,
            y_pred,
            y_probability,
            metrics,
            cm,
        )

    # --------------------------------------------------------
    # Feature Analysis
    # --------------------------------------------------------

    with tab3:

        show_feature_importance(model)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with tab4:

        show_prediction(model)

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    st.markdown("---")

    st.caption(
        "Online Exam Pass Prediction | Logistic Regression | "
        "Educational Machine Learning Project"
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
```

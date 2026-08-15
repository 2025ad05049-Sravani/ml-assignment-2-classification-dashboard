import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------
# Paths and model configuration
# ---------------------------------------------------------
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}


# ---------------------------------------------------------
# Load trained models and metadata
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    return {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }


@st.cache_data
def load_metadata():
    metadata_path = MODEL_DIR / "metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Classification Dashboard",
    page_icon="",
    layout="wide",
)

st.title("Breast Cancer Classification Dashboard")
st.caption(
    "Comparative evaluation of five machine-learning classifiers"
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("Model and Data")

    uploaded_file = st.file_uploader(
        "Upload test_data.csv",
        type=["csv"],
        help="Upload the held-out test dataset generated from the UCI dataset.",
    )

    selected_model = st.selectbox(
        "Select classification model",
        list(MODEL_FILES.keys()),
    )

    st.divider()

    st.subheader("Dataset Summary")

    st.caption(
        "Dataset: UCI Breast Cancer Wisconsin (Diagnostic)"
    )
    st.write("Instances: 569")
    st.write("Features: 30")
    st.write("Test samples: 114")
    st.write("Positive class: Malignant")

# ---------------------------------------------------------
# Wait for test data upload
# ---------------------------------------------------------
if uploaded_file is None:
    st.info("Upload test_data.csv to evaluate the selected model.")
    st.write("Expected 114 test records and 30 feature columns.")
    st.stop()


# ---------------------------------------------------------
# Load test dataset and validate columns
# ---------------------------------------------------------
df = pd.read_csv(uploaded_file)
metadata = load_metadata()

required_columns = metadata["features"] + ["diagnosis"]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error("The uploaded file is missing the following columns:")
    st.write(missing_columns)
    st.stop()


# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------
models = load_models()
model = models[selected_model]

X = df[metadata["features"]]
y = df["diagnosis"].astype(str)

predictions = model.predict(X)

malignant_index = list(model.classes_).index("malignant")

probabilities = model.predict_proba(X)[:, malignant_index]

y_true_binary = (y == "malignant").astype(int)
y_pred_binary = (
    pd.Series(predictions, index=df.index) == "malignant"
).astype(int)


# ---------------------------------------------------------
# Calculate evaluation metrics
# ---------------------------------------------------------
metrics = {
    "Accuracy": accuracy_score(y, predictions),
    "AUC": roc_auc_score(y_true_binary, probabilities),
    "Precision": precision_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0,
    ),
    "Recall": recall_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0,
    ),
    "F1": f1_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0,
    ),
    "MCC": matthews_corrcoef(
        y_true_binary,
        y_pred_binary,
    ),
}


# ---------------------------------------------------------
# Selected model evaluation
# ---------------------------------------------------------
st.subheader(f"Evaluation: {selected_model}")

metric_columns = st.columns(6)

for column, (metric_name, metric_value) in zip(
    metric_columns,
    metrics.items(),
):
    column.metric(
        metric_name,
        f"{metric_value:.4f}",
    )


# ---------------------------------------------------------
# Confusion matrix and classification report
# ---------------------------------------------------------
left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Confusion Matrix")

    confusion = confusion_matrix(
        y_true_binary,
        y_pred_binary,
        labels=[1, 0],
    )

    confusion_df = pd.DataFrame(
        confusion,
        index=[
            "Actual malignant",
            "Actual benign",
        ],
        columns=[
            "Predicted malignant",
            "Predicted benign",
        ],
    )

    st.dataframe(
        confusion_df,
        use_container_width=True,
    )


with right_column:
    st.subheader("Classification Report")

    report = classification_report(
        y,
        predictions,
        zero_division=0,
    )

    st.code(
        report,
        language="text",
    )


# ---------------------------------------------------------
# Prediction results
# ---------------------------------------------------------
output_df = df.copy()
output_df["Predicted Diagnosis"] = predictions

st.subheader("Prediction Results")

st.dataframe(
    output_df,
    use_container_width=True,
)


# ---------------------------------------------------------
# Download predictions
# ---------------------------------------------------------
st.download_button(
    label="Download predictions CSV",
    data=output_df.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)


# ---------------------------------------------------------
# All-model comparison
# ---------------------------------------------------------
st.divider()
st.subheader("All-Model Comparison")

comparison_rows = []

for model_name, current_model in models.items():

    current_predictions = current_model.predict(X)

    current_probabilities = current_model.predict_proba(X)[
        :,
        list(current_model.classes_).index("malignant"),
    ]

    current_pred_binary = (
        pd.Series(
            current_predictions,
            index=df.index,
        )
        == "malignant"
    ).astype(int)

    comparison_rows.append(
        {
            "ML Model Name": model_name,
            "Accuracy": accuracy_score(
                y,
                current_predictions,
            ),
            "AUC": roc_auc_score(
                y_true_binary,
                current_probabilities,
            ),
            "Precision": precision_score(
                y_true_binary,
                current_pred_binary,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_true_binary,
                current_pred_binary,
                zero_division=0,
            ),
            "F1": f1_score(
                y_true_binary,
                current_pred_binary,
                zero_division=0,
            ),
            "MCC": matthews_corrcoef(
                y_true_binary,
                current_pred_binary,
            ),
        }
    )


comparison_df = (
    pd.DataFrame(comparison_rows)
    .set_index("ML Model Name")
)

st.dataframe(
    comparison_df.style.format("{:.4f}"),
    use_container_width=True,
)
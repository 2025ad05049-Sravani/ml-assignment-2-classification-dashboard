from pathlib import Path
import sys
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


# ---------------------------------------------------------
# Paths Configuration
# ---------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent

sys.path.insert(0, str(HERE))


# ---------------------------------------------------------
# Import the model implementations
# ---------------------------------------------------------
from logistic_regression import build_model as build_logistic_regression
from decision_tree import build_model as build_decision_tree
from knn import build_model as build_knn
from naive_bayes import build_model as build_naive_bayes
from random_forest import build_model as build_random_forest


# ---------------------------------------------------------
# Main function
# ---------------------------------------------------------
def main():

    dataset_path = PROJECT_DIR / "breast_cancer_wisconsin.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    data = pd.read_csv(dataset_path)

    print("Dataset loaded successfully.")
    print(f"Dataset shape: {data.shape}")

    # -----------------------------------------------------
    # Remove the ID column
    # -----------------------------------------------------
    if "id" in data.columns:
        data = data.drop(columns=["id"])

    # -----------------------------------------------------
    # Check the target column
    # -----------------------------------------------------
    if "diagnosis" not in data.columns:
        raise ValueError(
            "The dataset must contain a 'diagnosis' column."
        )

    # -----------------------------------------------------
    # Separate the features and target
    # -----------------------------------------------------
    X = data.drop(columns=["diagnosis"])
    y = data["diagnosis"].astype(str).str.lower()

    y = y.replace({
        "m": "malignant",
        "b": "benign"
    })

    print(f"Number of features: {X.shape[1]}")

    print("\nClass distribution:")
    print(y.value_counts())

    # -----------------------------------------------------
    # Train-test split
    # -----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    test_data = X_test.copy()
    test_data["diagnosis"] = y_test.values

    test_data_path = PROJECT_DIR / "test_data.csv"
    test_data.to_csv(test_data_path, index=False)

    print(f"Saved test data: {test_data_path}")

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------
    metadata = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "UCI Machine Learning Repository",
        "uci_id": 17,
        "n_instances": len(data),
        "n_features": X.shape[1],
        "target": "diagnosis",
        "classes": ["malignant", "benign"],
        "positive_class_for_metrics": "malignant",
        "test_size": 0.20,
        "random_state": 42,
        "features": list(X.columns),
    }

    metadata_path = HERE / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"Saved metadata: {metadata_path}")

    # -----------------------------------------------------
    # Build models
    # -----------------------------------------------------
    models = {
        "Logistic Regression": build_logistic_regression(),
        "Decision Tree": build_decision_tree(),
        "kNN": build_knn(),
        "Naive Bayes": build_naive_bayes(),
        "Random Forest (Ensemble)": build_random_forest(),
    }

    # -----------------------------------------------------
    # Train, evaluate and save the models
    # -----------------------------------------------------
    results = []

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        malignant_index = list(
            model.classes_
        ).index("malignant")

        probabilities = model.predict_proba(
            X_test
        )[:, malignant_index]

        y_true_binary = (
            y_test == "malignant"
        ).astype(int)

        y_pred_binary = (
            pd.Series(
                predictions,
                index=y_test.index
            ) == "malignant"
        ).astype(int)

        # Evaluation metrics
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        auc = roc_auc_score(
            y_true_binary,
            probabilities
        )

        precision = precision_score(
            y_true_binary,
            y_pred_binary,
            zero_division=0
        )

        recall = recall_score(
            y_true_binary,
            y_pred_binary,
            zero_division=0
        )

        f1 = f1_score(
            y_true_binary,
            y_pred_binary,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_true_binary,
            y_pred_binary
        )

        # Storing the results
        results.append({
            "ML Model Name": model_name,
            "Accuracy": accuracy,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MCC": mcc,
        })

        filename = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        model_path = HERE / f"{filename}.joblib"

        joblib.dump(model, model_path)

        print(f"Saved: {model_path}")

    # -----------------------------------------------------
    # Save model metrics
    # -----------------------------------------------------
    metrics_df = pd.DataFrame(results)

    metrics_path = PROJECT_DIR / "model_metrics.csv"

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print(f"\nSaved metrics: {metrics_path}")

    print("\nFinal model comparison:")
    print(
        metrics_df.to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# Run script
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
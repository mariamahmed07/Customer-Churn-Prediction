
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"


def main():
    print("=" * 55)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 55)

    # Load the dataset
    try:
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"\nError: Dataset not found: {DATA_FILE}")
        print("Make sure the CSV file is in the project folder.")
        return

    # Clean the dataset
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )
    data = data.dropna(subset=["TotalCharges"]).copy()

    print("\nDataset loaded successfully!")
    print(f"Total customers: {len(data)}")

    # Separate features and target
    X = data.drop(columns=[TARGET, "customerID"])
    y = data[TARGET].map({"No": 0, "Yes": 1})

    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    # Preprocess categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            ("numerical", "passthrough", numerical_columns),
        ]
    )

    # Build the machine learning pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # Train the model
    print("\nTraining the model...")
    model.fit(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\nMODEL EVALUATION")
    print("-" * 35)
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )

    # Calculate and display the confusion matrix
    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    print("CONFUSION MATRIX")
    print(matrix)

    print(
        "\nNote: Predictions are estimates, not guaranteed outcomes."
    )

    # Plot the confusion matrix
    plt.figure(figsize=(7, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Customer Churn - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks([0, 1], ["No Churn", "Churn"])
    plt.yticks([0, 1], ["No Churn", "Churn"])

    for row in range(2):
        for col in range(2):
            plt.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # Calculate feature importance
    fitted_preprocessor = model.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    importance_data = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    ).sort_values(
        "Importance", ascending=False
    ).head(10)

    # Plot the top 10 important features
    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_data["Feature"][::-1],
        importance_data["Importance"][::-1],
    )
    plt.title("Top 10 Features for Customer Churn Prediction")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

    # Save feature importance to a CSV file
    importance_data.to_csv(
        "feature_importance.csv", index=False
    )

    print("\nCharts and results saved:")
    print("- confusion_matrix.png")
    print("- feature_importance.png")
    print("- feature_importance.csv")


if __name__ == "__main__":
    main()

"""
In this file a UPF classification model is being created. This is the Naive version
with no MLOps guardrails built in to see how this can cause issues.

This is the quick and dirty approach to ML. There is:
    - No experiment tracking
    - No model versioning
    - The model is just trained and saved to a pickle file

This will be improved upon in later sections.
"""
import pandas as pd 
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix, 
    classification_report
)

NUTRITION_COLS = [
    "energy_kcal", "fat_100g", "saturated_fat_100g", "carbs_100g", "sugars_100g",
    "fiber_100g", "proteins_100g", "salt_100g", "sodium_100g"
]

ALLERGEN_COLS = [
    "contains_gluten", "contains_dairy", "contains_nuts", "contains_soy",
    "contains_eggs", "contains_fish"
]

CATEGORICAL_COLS = ["top_category", "nutriscore_grade"]

def main():
    print("Loading data")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Encode categorcial features
    # The encoders need to be saved to use the same mapping at inference time
    print("\nEncoding categorical features")
    encoders = {}
    for col in CATEGORICAL_COLS:
        encoder = LabelEncoder()
        train_df[f"{col}_encoded"] = encoder.fit_transform(train_df[col])

        known = set(encoder.classes_)
        fallback = train_df[col].mode()[0]
        test_df[col] = test_df[col].where(test_df[col].isin(known), fallback)

        test_df[f"{col}_encoded"] = encoder.transform(test_df[col])
        encoders[col] = encoder
        print(f"{col} mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

    # Prepare features and labels
    feature_cols = (
        NUTRITION_COLS + ALLERGEN_COLS
        + ["ingredient_count", "has_e_number", "has_additive_keyword"]
        + [f"{col}_encoded" for col in CATEGORICAL_COLS]
    )

    X_train = train_df[feature_cols]
    y_train = train_df["is_ultra_processed"]

    X_test = test_df[feature_cols]
    y_test = test_df["is_ultra_processed"]

    # Train a Random Forest classifier
    print("\nTraining Random Forest model")
    model = RandomForestClassifier(
        n_estimators = 100,
        max_depth = 10,
        random_state = 42,
        n_jobs = -1
    )
    model.fit(X_train, y_train)
    print("Training complete")

    # Evaluate on test data 
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives: {cm[0][0]:,} (correctly identified non-UPF)")
    print(f"False Positives: {cm[0][1]:,} (non-UPF flagged as UPF)")
    print(f"False Negatives: {cm[1][0]:,} (UPF missed)")
    print(f"True Positives: {cm[1][1]:,} (correctly caught UPFs)")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-UPF', 'UPF']))

    # Feature importance
    print("\nFeature Importance:")
    for name, importance in sorted(
        zip(feature_cols, model.feature_importances_),
        key = lambda x: x[1],
        reverse = True
    ):
        print(f"{name}: {importance:.4f}")

    # Save the model and encoders together
    print("\nSaving model to models/model.pkl")
    with open("models/model.pkl", "wb") as f:
        pickle.dump((model, encoders), f)

    print("\nModel trained and saved successfully.")
    print("\nWarning: This is a naive approach which has several problems:")
    print(" - No record of hyperparameters or metrics")
    print(" - No model versioning")
    print(" - No way to preproduce this exact model")
    print(" - This will be fixed in the next sections of the project")

if __name__ == "__main__":
    main()


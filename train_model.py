"""Train and save a Random Forest model for Iris flower classification."""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = "Iris.csv"
MODEL_FILE = "iris_model.pkl"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

def train():
    data = pd.read_csv(DATA_FILE)
    X = data[FEATURES]
    y = data["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Model accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(model, MODEL_FILE)
    print(f"Saved trained model to {MODEL_FILE}")

if __name__ == "__main__":
    train()

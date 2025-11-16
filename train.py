from poisoning import poison_labels
import os
import io
import json
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
import seaborn as sns

# Simple helper to ensure deterministic behaviour
SEED = int(os.getenv("RANDOM_SEED", 42))
np.random.seed(SEED)

MLFLOW_ARTIFACT_URI = os.getenv("MLFLOW_ARTIFACT_URI")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")  # optional
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

if not MLFLOW_ARTIFACT_URI:
    raise RuntimeError("Please set MLFLOW_ARTIFACT_URI environment variable, e.g. gs://your-mlflow-bucket")

mlflow.set_experiment("iris-label-poisoning")


def plot_and_log_confusion(cm, labels, artifact_path, run):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    # MLflow artifact path expects a file name
    artifact_name = os.path.join(artifact_path, 'confusion_matrix.png')
    mlflow.log_image(buf, artifact_name)
    plt.close()


def run_experiment(X_train, y_train, X_test, y_test, poison_percent, model_params=None):
    model_params = model_params or {"n_estimators": 100, "random_state": SEED}
    run_name = f"poison_{poison_percent}pct" if poison_percent > 0 else "baseline"

    # Apply poisoning to the training labels only
    y_train_poisoned = poison_labels(y_train, poison_percent)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("poison_percent", poison_percent)
        mlflow.log_params(model_params)

        clf = RandomForestClassifier(**model_params)
        clf.fit(X_train, y_train_poisoned)

        preds = clf.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')

        cm = confusion_matrix(y_test, preds)

        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_macro", float(f1))

        # Log model
        mlflow.sklearn.log_model(clf, artifact_path="model")

        # Save confusion matrix image and classification report to artifacts
        labels = [str(l) for l in np.unique(y_test)]
        # Use a temporary file to save matplotlib output and then log via mlflow.log_artifact
        tmpdir = tempfile.mkdtemp()
        cm_path = os.path.join(tmpdir, "confusion_matrix.png")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix ({run_name})')
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")

        report = classification_report(y_test, preds, output_dict=True)
        report_path = os.path.join(tmpdir, "classification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path, artifact_path="reports")

        print(f"Run: {run_name} | accuracy: {acc:.4f} | f1_macro: {f1:.4f}")


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # split once; test set remains clean
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    # Training config
    model_params = {"n_estimators": 100, "random_state": SEED}

    # Make sure MLflow knows where artifacts go
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', mlflow.get_tracking_uri()))
    mlflow.set_experiment("iris-label-poisoning")

    # Log a baseline run (no poisoning)
    run_experiment(X_train, y_train, X_test, y_test, poison_percent=0, model_params=model_params)

    for p in [5, 10, 50]:
        run_experiment(X_train, y_train, X_test, y_test, poison_percent=p, model_params=model_params)


if __name__ == '__main__':
    main()


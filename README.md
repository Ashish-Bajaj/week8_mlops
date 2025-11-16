# Iris Label Poisoning Experiment with MLflow & GCS (Week 8 MLOps)

This project demonstrates how label poisoning affects machine learning model performance using the Iris dataset.  
We track all experiments using **MLflow**, store artifacts in **Google Cloud Storage (GCS)**, and run everything from a clean **GCP VM**.

---

## 📌 Project Overview  
We perform and compare four training runs of a Random Forest classifier:

1. **Baseline (no poisoning)**
2. **5% label poisoning**
3. **10% label poisoning**
4. **50% label poisoning**

Label poisoning is simulated by randomly flipping the labels of a given percentage of samples.

All runs are logged into MLflow with:
- Accuracy & F1 metrics  
- Confusion matrix  
- Classification report  
- Trained model  
- Parameters

Artifacts are stored in **GCS**, and the MLflow tracking server runs on the VM.

---

## 📂 Project Structure
week8/
├── train.py # Main training script with MLflow logging
├── poisoning.py # Helper to generate label-poisoned datasets
├── requirements.txt # Python dependencies (MLflow, sklearn, etc.)
├── README.md # Project documentation
├── .gitignore # Ignore venv, mlflow.db, GCS key, etc.
└── venv/ # Virtual environment (not pushed to GitHub)



---

## 🚀 How to Run

### 1. Set up virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Export environment variables
export GOOGLE_APPLICATION_CREDENTIALS="/home/<user>/GCS-KEY.json"
export MLFLOW_ARTIFACT_URI="gs://<your-gcs-bucket>"
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

3. Start MLflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root $MLFLOW_ARTIFACT_URI \
  --host 0.0.0.0 \
  --port 5000


4. Run training

Open a second terminal:

source venv/bin/activate
python train.py


Results appear in MLflow UI at:

http://<VM-IP>:5000
## 📂 Project Structure


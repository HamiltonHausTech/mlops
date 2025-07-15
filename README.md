# MLflow + BentoML Demo: End-to-End MLOps Prototype

This project demonstrates a minimal MLOps pipeline for training, tracking, and deploying a machine learning model using modern tools:

- **MLflow** for experiment tracking and model versioning
- **BentoML** for packaging and serving the model via a REST API
- **Docker Compose** for local deployment
- **cURL** for testing inference

---

## 🚀 What It Does

- Trains a simple `scikit-learn` model (`RandomForestClassifier`) on the Iris dataset
- Logs model artifacts to MLflow
- Loads the model dynamically into a BentoML service at runtime
- Serves a REST API to perform real-time predictions

---

## 📂 Project Structure

mlops-demo/
├── model/
│ └── train.py # Train and log model to MLflow
├── serve/
│ ├── api.py # BentoML service definition
│ ├── bentofile.yaml # BentoML build config
│ ├── Dockerfile # Optional containerization
│ └── requirements.txt
├── infra/
│ └── docker-compose.yml # Starts MLflow tracking server
├── client/
│ └── test_client.py # (Optional) API test script
└── README.md


---

## 🛠️ Getting Started

### 1. Start MLflow Server

cd infra/
docker-compose up -d

Access MLflow UI at: http://localhost:5001

### 2. Train and Log Model
cd model/
python train.py


### 3. Build and Serve the Model
cd serve/
bentoml build
bentoml serve api:IrisService --reload

### 4. Test the API
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"payload": {"data": [[5.1, 3.5, 1.4, 0.2]]}}'

Expected output:
{"predictions": [0]}

## Technologies
-Python 3.10
-scikit-learn
-MLflow
-BentoML 1.x
-Docker + Compose


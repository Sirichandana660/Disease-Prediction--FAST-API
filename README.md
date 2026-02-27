Disease Prediction Using Symptoms(FastAPI + Machine Learning)
A Machine Learning powered web application that predicts diseases based on user-selected symptoms.

Built using FastAPI and Scikit-learn, this project demonstrates full ML pipeline integration into a production-ready API.

🚀 Features
Predicts diseases from symptom inputs
Trained using Decision Tree & Random Forest models
Integrated ML model using joblib
FastAPI backend with HTML templates
Interactive API documentation at /doc
Real-time predictions

🛠 Tech Stack
Python 3.13
FastAPI
Uvicorn
Scikit-learn
Panda
NumPy
HTML (Jinja Templates)

📂 Project Structure
disease prediction/
│
├── app.py
├── train_model.py
├── create_dataset.py
├── disease_model.joblib
├── disease_symptoms_extended.csv
│
└── templates/
    ├── index.html
    └── result.html

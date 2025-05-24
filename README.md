# 🌦️ Australian Weather Prediction — End-to-End ML Pipeline

An end-to-end machine learning pipeline built for predicting weather outcomes in Australia. This project demonstrates a complete ML workflow — from data ingestion to model deployment — using a clean and scalable codebase. Ideal for understanding the fundamentals of data pipelines, preprocessing, model training, and interactive web deployment using Flask.

---

## 🌟 Key Features

* **End-to-End ML Pipeline**: Covers data ingestion, preprocessing, training, and prediction.
* **Interactive Web App**: Users can input weather-related data and receive real-time predictions.
* **Preprocessing Pipeline**: Uses `ColumnTransformer` with numerical and categorical pipelines.
* **Model Persistence**: Trained model and preprocessing pipeline saved with `joblib` or `pickle`.
* **Error Handling**: Supports unseen categories with robust preprocessing (`handle_unknown='ignore'`).
* **Modern UI**: Sleek dark-themed frontend built with HTML, CSS, and JavaScript.

---

## 🛠️ Tech Stack and Tools

* **Programming Language**: Python
* **ML Libraries**: scikit-learn, pandas, numpy
* **Web Framework**: Flask
* **Frontend**: HTML5, CSS3 (Custom dark theme)
* **Model Persistence**: joblib / pickle

---

## ⚙️ Architecture Overview

This project follows a modular architecture:

1. **Data Ingestion**: Weather dataset loaded from CSV or database.
2. **Preprocessing**: Features transformed using a `ColumnTransformer` (imputation, encoding, scaling).
3. **Model Training**: Trains a classification model to predict rainfall (or other outcomes).
4. **Model Inference**: Accepts input from the web UI, preprocesses it, and predicts using the saved model.
5. **Deployment**: A Flask web app serves the model locally.

---

## 📂 Project Structure

```
weather-predictor/
├── static/                 # CSS and static files
├── templates/              # HTML templates for the UI
├── artifacts/              # Stored model and preprocessor (model.pkl, preprocessor.pkl)
├── app.py                  # Flask application
├── train.py                # Model training script
├── preprocess.py           # Preprocessing pipeline
├── utils.py                # Helper functions
├── weather.csv             # Dataset (optional)
├── requirements.txt        # Python dependencies
└── README.md               # You're reading this!
```

---

## 🚀 Getting Started

### ✅ Prerequisites

* Python 3.8+
* pip

### 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/weather-predictor.git
cd weather-predictor

# Create virtual environment and activate
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 🧠 Train the Model

```bash
python train.py
```

This script will:

* Load the dataset
* Train the model and preprocessing pipeline
* Save `model.pkl` and `preprocessor.pkl` to `artifacts/`

---

## 🌐 Run the App

```bash
python app.py
```

Visit the app at:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📈 Challenges and Learnings

### 🔍 Challenges

* Handling unknown categories in categorical features
* Designing a user-friendly UI with robust validation
* Ensuring consistent preprocessing during training and inference

### 🧠 Learnings

* Developed understanding of end-to-end ML workflows
* Improved Flask integration and model deployment skills
* Learned robust ways to handle categorical encoding with `OneHotEncoder`

---

## 🚀 Future Improvements

* Add model evaluation and performance metrics
* Implement logging and exception handling
* Deploy to a cloud platform (e.g., AWS, Render, Heroku)
* Add support for continuous training and model versioning

---

## 👨‍💻 Author

**Lakshay**
[GitHub](https://github.com/lakshaygoel4321) 
[LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📝 License

This project is licensed under the MIT License — see the `LICENSE` file for details.

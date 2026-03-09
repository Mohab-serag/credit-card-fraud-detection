# Credit Card Fraud Detection
[Live App](https://credit-card-fraud-detection-mohab.streamlit.app)
![Application Screenshot](card.png)

A Machine Learning web application that detects fraudulent credit card transactions using a trained classification model.
The application allows users to upload transaction data and instantly identify suspicious activities.
---

## Project Overview

Credit card fraud is a major problem in the financial industry.
This project builds a machine learning model capable of detecting fraudulent transactions based on historical transaction data.

The application allows users to:

* Upload transaction data as a CSV file
* Automatically preprocess the data
* Predict whether transactions are fraudulent or normal
* View results in a simple interactive web interface

---

## Machine Learning Pipeline

### Exploratory Data Analysis (EDA)

The dataset was analyzed to understand the distribution of transactions and detect class imbalance between fraudulent and normal transactions.

### Data Preprocessing

The preprocessing stage includes:

* Feature scaling using StandardScaler
* Handling class imbalance using SMOTE

### Model Training

The model used in this project:

Random Forest Classifier

The model was trained on resampled data to improve fraud detection performance.

---

## Web Application

The trained model is deployed using Streamlit.

Users can:

* Upload a CSV file containing transaction data
* Preview the uploaded dataset
* Run fraud detection predictions
* View the prediction results directly in the browser

---

## Technologies Used

Python
Pandas
NumPy
Scikit-learn
Imbalanced-learn (SMOTE)
Streamlit
Joblib

---

## Project Structure

```
credit-card-fraud-detection
│
├── app
│   └── app.py
│
├── notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model.ipynb
│
├── model.pkl
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## Running the Project Locally

Clone the repository:

```
git clone https://github.com/Mohab-serag/credit-card-fraud-detection.git
```

Install the required libraries:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app/app.py
```

---

## Dataset

This project uses the Credit Card Fraud Detection dataset containing anonymized transaction data.

The dataset includes:

* Time
* Amount
* PCA transformed variables (V1 – V28)
* Class (fraud or normal)

---

## Author

Mohab Serag

GitHub:
https://github.com/Mohab-serag

# 📧 Spam Message Classifier

This project is a machine learning-based spam message classifier that detects whether an SMS message is **spam** or **ham (not spam)** using natural language processing techniques and a trained model.

## 📂 Project Structure

- `main.ipynb` - Contains the code for preprocessing, training, evaluating, and using the spam classifier.
- `spam.csv` - Dataset containing SMS messages labeled as spam or ham.
- `spam_classifier.pkl` - The trained model file saved for reuse.

## 🧠 Model Overview

- **Type**: Binary Text Classification
- **Techniques Used**: 
  - Text preprocessing (cleaning, tokenization)
  - Feature extraction (e.g., TF-IDF or CountVectorizer)
  - Classification algorithm (e.g., Naive Bayes, Logistic Regression)

## 🛠️ Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/spam-classifier.git
   cd spam-classifier

## 🧪 How It Works
- Load and preprocess the dataset.

- Split into training and testing sets.

- Transform text into numeric features.

- Train a machine learning model.

- Save and use the model for predictions.

  ## Example Usage
- Once the model is trained, you can use the following to make predictions:

- import pickle

## Load the model
- with open("spam_classifier.pkl", "rb") as file:
    -  model = pickle.load(file)

## Predict
- sample = ["Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/123456 to claim now."]
- result = model.predict(sample)
- print("Spam" if result[0] == 1 else "Ham")



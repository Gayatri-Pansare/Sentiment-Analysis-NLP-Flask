
```markdown
# Sentiment Analysis of Flipkart Product Reviews

## Project Overview:
This project performs **sentiment analysis** on real-time Flipkart product reviews to classify them as **Positive** or **Negative**.  
It helps understand **customer satisfaction and pain points** by analyzing textual reviews using **Natural Language Processing (NLP)** and **Machine Learning / Deep Learning models**.

The final solution is deployed as a **Flask web application** that predicts sentiment for user-entered reviews in real time.

---

## Objectives:
- Classify customer reviews into **Positive** or **Negative**
- Identify common **pain points** from negative reviews
- Experiment with multiple **text embeddings and models**
- Optimize performance using **Optuna hyperparameter tuning**
- Deploy the best model using **Flask**

---

## Dataset:
- Product: **YONEX MAVIS 350 Nylon Shuttle**
- Source: Flipkart (scraped by Data Engineering team)
- Total Reviews: **8,518**
- Features include:
  - Review Text
  - Rating
  - Review Title
  - Reviewer Name
  - Date
  - Upvotes / Downvotes

**Note:** Data scraping was not performed in this project.

---

## Project Workflow:
1. Data Loading & Exploration  
2. Data Cleaning & Text Preprocessing  
3. Feature Engineering (BoW, TF-IDF)  
4. Model Training (ML & DL)  
5. Hyperparameter Tuning using Optuna  
6. Cross-Validation for Robust Evaluation  
7. Model Selection using F1-Score  
8. Flask App Development  
9. Model Deployment  

---

## Text Preprocessing:
- Lowercasing
- Removal of punctuation & special characters
- Stopword removal
- Lemmatization / Stemming (tested via Optuna)

---

## Feature Extraction :
- **Bag of Words (BoW)**
- **TF-IDF**

---

## Models Used
### Machine Learning
- Logistic Regression
- Linear SVM
- Naive Bayes
- Random Forest
- XGBoost

### Deep Learning
- LSTM (TensorFlow / Keras)

---

## Hyperparameter Tuning:
- Tool: **Optuna**
- Tuned:
  - Preprocessing method
  - Vectorizer type
  - Model selection
  - Model hyperparameters
- Evaluation Metric: **F1-Score**
- Validation: **Stratified K-Fold Cross-Validation**

### Best Model:
- **Model:** Linear SVM  
- **Vectorizer:** Bag of Words  
- **Max Features:** 6114  
- **F1 Score:** **0.9575**

---

## Flask Web Application:
The Flask app:
- Accepts a review from the user
- Converts text using the saved vectorizer
- Predicts sentiment using the trained model
- Displays result as **Positive** or **Negative**

---

## Project Structure:
```

sentiment_analysis_flask/
│
├── app.py
├── requirements.txt
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── data/
│   └── data.csv
│
├── notebookes/
│   └── sentiment_analysis_project.ipynb
|
├── venv/
└── README.md

````

---

## Virtual Environment Setup
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
````

---

## Install Dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Flask Application:

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## Example Predictions:

**Input:**

> "Shuttle quality is very poor and breaks easily"

**Output:**
❌ Negative

---

## Tools & Technologies:

* Python
* Pandas, NumPy
* Scikit-learn
* TensorFlow / Keras
* Optuna
* Flask
* HTML / CSS

---

## Evaluation Metric:

* **F1-Score** (chosen due to class imbalance)

---

## Future Improvements:

* BERT-based sentiment analysis
* REST API version
* Dockerization
* Confidence score display
* Cloud deployment (AWS EC2)


---

## Author:

**Gayatri Pansare**

---



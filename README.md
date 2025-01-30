# Bag of Words NLP Project

## 📌 Project Overview
This project explores the **Bag of Words (BoW) model** in **Natural Language Processing (NLP)** using different text classification tasks such as **email spam detection** and **movie review sentiment analysis**. The project applies multiple machine learning classifiers to understand the effectiveness of BoW in text processing.

The classifiers used include:

- **RandomForestClassifier**
- **KNeighborsClassifier**
- **Multinomial Naïve Bayes**

The project also includes **data preprocessing**, **feature extraction (Bag of Words - CountVectorizer)**, and **performance visualization** using **graphs and charts**.

---
## 🚀 Features
- **Text Preprocessing**: Tokenization, stopword removal, and vectorization using CountVectorizer.
- **Multiple Classifiers**: Trained and compared different models (Random Forest, KNN, Naïve Bayes).
- **Performance Metrics**: Confusion matrix, accuracy comparison, and ROC curves.
- **Visualization**: Graphical analysis of model performance.

---
## 📂 Dataset
The datasets used in this project include:
- **Email Spam Dataset**: Labeled emails as **Spam** or **Not Spam**.
- **Movie Review Dataset**: Reviews classified as **Positive** or **Negative**.

The datasets are preprocessed and split into training and testing sets using `train_test_split()`.

---
## 🛠 Installation & Setup
### 1️⃣ Clone the Repository
```bash
gh repo clone Sarthak2403/BagOfWords-NLP-Classifier
cd BagOfWords-NLP-Classifier
```

### 2️⃣ Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Jupyter Notebook
Launch Jupyter Notebook and open `BOW_using_nlp.ipynb`:
```bash
jupyter notebook
```

---
## 📊 Model Training & Evaluation
### **1️⃣ Train the Models**
The following classifiers are trained on the datasets:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
```
### **2️⃣ Performance Evaluation**
The models are evaluated using:
- **Accuracy Scores** (Bar Chart)
- **Confusion Matrices** (Heatmaps)
- **ROC Curves** (AUC Comparison)

---
## 📜 Results
1. Email Spam Detection: MultinomialNB:- Accuracy = 98%

2. After training and evaluating the models, the **best performing classifier** is determined based on accuracy and other metrics.

| Model               | Accuracy |
|--------------------|----------|
| Random Forest      | 84%      |
| KNeighbors        | 65%      |
| MultinomialNB     | 85%      |

---
**⭐ If you find this project helpful, please star the repository!**
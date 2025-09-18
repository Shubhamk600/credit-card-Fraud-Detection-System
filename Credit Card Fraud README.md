# Credit Card Fraud Detection

## Project Overview
The goal of this project is to detect fraudulent credit card transactions using machine learning. The dataset used is the Kaggle Credit Card Fraud Detection dataset, which contains transactions made by European cardholders in September 2013. It is highly imbalanced, with fraudulent transactions representing a very small fraction of the total.  

Dataset source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Setup Instructions
1. Clone or download this repository.  
2. Make sure you have Python 3.8+ installed.  
3. Install the required libraries:  
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
4. Download the dataset (`creditcard.csv`) from Kaggle and place it in the project folder.  
5. Run the notebook or script to train and evaluate the model.  

---

## Libraries Used
- **pandas** – data loading and preprocessing  
- **numpy** – numerical computations  
- **scikit-learn** – machine learning models and evaluation metrics  
  - Isolation Forest  
  - Random Forest  
  - Train-test split, accuracy, precision, recall, F1-score, AUC-ROC  
- **matplotlib / seaborn** – visualizations  

---

## Approach
1. **Data Preparation**  
   - Loaded and explored the dataset.  
   - Normalized the data and split into train/test sets.  

2. **Modeling**  
   - Started with **Isolation Forest** for anomaly detection. This approach worked as an unsupervised method but struggled to capture fraud cases effectively.  
   - Switched to **Random Forest Classifier**, which significantly improved recall and F1-score for fraud detection.  

3. **Evaluation**  
   - Accuracy was not treated as the main metric because of imbalance.  
   - Focused on **Precision, Recall, F1-score, and AUC-ROC** to evaluate how well the model detects rare fraud cases.  

---

## Challenges Faced and Solutions
- **Severe Class Imbalance**  
  Fraudulent transactions make up less than 0.2% of the data. Initially, this caused models to be biased toward predicting only normal transactions. To handle this, we first tried an anomaly detection method (Isolation Forest). Since performance was limited, we moved to a supervised method (Random Forest), which gave much better recall on fraud detection.  

- **Overfitting Risk**  
  Tree-based models tended to overfit the training data. We reduced this risk by tuning hyperparameters and validating on unseen test data.  

- **Evaluation Metrics**  
  Standard accuracy was misleading due to the imbalance. We relied on **AUC-ROC, Recall, and F1-score**, which are more meaningful in fraud detection.  

---

## Results
- **Isolation Forest**: Worked as a baseline but could not achieve strong recall for fraud detection.  
- **Random Forest**: Outperformed Isolation Forest, delivering better AUC-ROC and recall, making it more practical for real-world fraud detection.  

# Online Shoppers Purchasing Intention — Classification & Deployment

**BITS Pilani Work Integrated Learning Programmes (WILP)**  
**M.Tech (AIML) — Machine Learning (Assignment 2)**  
**Student ID:** 2026AJ05386

---

## 1. Problem Statement

Given a dataset of 12,330 online shopping sessions from an e-commerce website, the objective is to **predict whether a visitor will complete a purchase** (generate revenue) or leave without buying.  
This is a **Binary Classification** task where:
- **Class 1 (Revenue = True):** The session ended with a purchase (~15.5% of sessions).
- **Class 0 (Revenue = False):** The session did not result in a purchase (~84.5% of sessions).

Five supervised classification algorithms are trained, evaluated on six metrics, and compared to identify the best-performing model.

---

## 2. Dataset Description

| Property | Value |
|:---|:---|
| **Name** | Online Shoppers Purchasing Intention Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| **Total Instances** | 12,330 |
| **Total Features** | 17 (input) + 1 (target) |
| **Target Variable** | `Revenue` (Boolean → converted to 0/1) |
| **Class Balance** | Negative: 84.5% · Positive: 15.5% (imbalanced) |
| **Missing Values** | None |
| **Train-Test Split** | 80% Train (9,864) / 20% Test (2,466), stratified |

### Feature Dictionary

| # | Feature | Type | Description |
|:--|:---|:---:|:---|
| 1 | `Administrative` | Integer | Number of administrative pages visited |
| 2 | `Administrative_Duration` | Float | Total time (seconds) spent on administrative pages |
| 3 | `Informational` | Integer | Number of informational pages visited |
| 4 | `Informational_Duration` | Float | Total time (seconds) spent on informational pages |
| 5 | `ProductRelated` | Integer | Number of product-related pages visited |
| 6 | `ProductRelated_Duration` | Float | Total time (seconds) spent on product-related pages |
| 7 | `BounceRates` | Float | Average bounce rate of pages visited by the user |
| 8 | `ExitRates` | Float | Average exit rate of pages visited by the user |
| 9 | `PageValues` | Float | Average page value of pages visited by the user |
| 10 | `SpecialDay` | Float | Closeness of browsing date to a special day (e.g., Mother's Day) |
| 11 | `Month` | Categorical | Month of the year (Feb–Dec) |
| 12 | `OperatingSystems` | Integer | Operating system of the visitor |
| 13 | `Browser` | Integer | Browser used by the visitor |
| 14 | `Region` | Integer | Geographic region of the visitor |
| 15 | `TrafficType` | Integer | Traffic source type (e.g., direct, referral, search) |
| 16 | `VisitorType` | Categorical | Returning_Visitor / New_Visitor / Other |
| 17 | `Weekend` | Boolean | Whether the session occurred on a weekend |
| **Target** | `Revenue` | Boolean | Whether the session generated revenue (purchase) |

---

## 3. Models Trained

All five models use a **shared preprocessing pipeline**:
- **Numerical features (15):** StandardScaler normalization.
- **Categorical features (2: Month, VisitorType):** OneHotEncoder (drop='first').

| # | Model | Key Hyperparameters |
|:--|:---|:---|
| 1 | Logistic Regression | `solver='liblinear'`, `max_iter=1000` |
| 2 | Decision Tree Classifier | `max_depth=6`, `min_samples_leaf=10` |
| 3 | K-Nearest Neighbors (KNN) | `n_neighbors=7`, `weights='distance'` |
| 4 | Gaussian Naive Bayes | `var_smoothing=0.1` |
| 5 | Support Vector Machine (SVM) | `kernel='rbf'`, `C=1.0` |

---

## 4. Evaluation Metrics — 5-Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.8812 | 0.8878 | 0.7432 | 0.3560 | 0.4814 | 0.4603 |
| **Decision Tree** | **0.8954** | **0.9141** | 0.7246 | **0.5236** | **0.6079** | **0.5590** |
| KNN | 0.8719 | 0.7999 | 0.6528 | 0.3691 | 0.4716 | 0.4263 |
| Naive Bayes | 0.8163 | 0.8137 | 0.4334 | 0.6047 | 0.5049 | 0.4042 |
| SVM | 0.8865 | 0.8727 | 0.7008 | 0.4660 | 0.5597 | 0.5113 |

---

## 5. Model Observations

| Model | Observation |
|:---|:---|
| **Logistic Regression** | Achieves the highest precision (74.3%) with very few false positives, but suffers from low recall (35.6%). The linear decision boundary misses complex multi-feature interaction patterns in browsing behavior (e.g., combined effect of low exit rates + high page values). AUC is strong at 0.8878, indicating good ranking ability even if the threshold is conservative. |
| **Decision Tree** | **Best overall performer.** Balances precision (72.5%) and recall (52.4%) effectively. Its non-linear, threshold-based splits on `PageValues` and `ExitRates` directly capture the hierarchical rules governing purchase intent. Achieves the highest ROC-AUC (0.9141), F1 (0.6079), and MCC (0.5590), making it robust against the class imbalance. |
| **KNN** | Moderate performance with accuracy of 87.2% and AUC of 0.80. Suffers from the curse of dimensionality in the 17-feature session space — distance-based similarity becomes less meaningful as dimensionality rises. Recall is low (36.9%), indicating difficulty distinguishing minority-class buyers from non-buyers in dense feature neighborhoods. |
| **Naive Bayes** | Achieves the highest recall (60.5%) among all models, successfully identifying a majority of actual buyers. However, precision drops to 43.3% because the conditional independence assumption is violated — features like `BounceRates`, `ExitRates`, and `PageValues` are inherently correlated. This leads to many false positives. Accuracy (81.6%) is the lowest, confirming the precision-recall trade-off. |
| **SVM** | Second-best overall with strong accuracy (88.65%) and MCC (0.5113). The RBF kernel successfully separates non-linearly clustered sessions in scaled feature space. However, it is computationally heavier and slightly underperforms Decision Tree on AUC (0.8727 vs 0.9141) and recall (46.6% vs 52.4%). |

---

## 6. Overall Winner

### Winner: Decision Tree Classifier

The Decision Tree Classifier is selected as the **overall best model** for this dataset based on the following justification:

1. **Highest ROC-AUC (0.9141):** Best ability to distinguish between buyers and non-buyers across all classification thresholds.
2. **Highest F1-Score (0.6079):** Best harmonic balance between precision and recall on the imbalanced dataset.
3. **Highest MCC (0.5590):** Most reliable single metric for binary classification with class imbalance; confirms the model's predictions are well-correlated with actual outcomes.
4. **Interpretability:** Decision tree splits can be directly explained to business stakeholders (e.g., "If PageValues > X and ExitRates < Y, predict purchase"), making it suitable for actionable e-commerce insights.

---

## 7. Repository Structure

```
├── app.py                  # Interactive Streamlit web application
├── requirements.txt        # Python dependencies for deployment
├── README.md               # This file
├── test_data.csv            # Holdout test set (2,466 rows, 18 columns)
├── .gitignore
└── model/
    ├── train_models.py                 # Training & evaluation script
    ├── online_shoppers_modeling.ipynb   # Jupyter notebook with full EDA & training
    ├── preprocessor.joblib             # Fitted ColumnTransformer
    ├── logistic_regression.joblib      # Saved pipeline: Logistic Regression
    ├── decision_tree.joblib            # Saved pipeline: Decision Tree
    ├── knn.joblib                      # Saved pipeline: KNN
    ├── naive_bayes.joblib              # Saved pipeline: Naive Bayes
    ├── svm.joblib                      # Saved pipeline: SVM
    ├── metrics_comparison.csv          # 5-model × 6-metric comparison table
    └── evaluation_details.json         # Detailed per-model confusion matrices & reports
```

---

## 8. How to Run

### Local Testing
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit App Features
1. **CSV Upload:** Upload any test CSV with the same 17 features + Revenue column.
2. **Model Selection:** Dropdown to pick any of the 5 classifiers or compare all.
3. **Metrics Display:** 6 evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC).
4. **Confusion Matrix & Classification Report:** Visual heatmap and tabular report for the selected model.


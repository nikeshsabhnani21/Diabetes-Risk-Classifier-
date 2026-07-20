# Diabetes Risk Classifier

Predicts diabetes risk from CDC health survey data (253,680 records), 
prioritizing recall to catch high-risk cases despite severe class imbalance.

## Overview

Standard accuracy is misleading on imbalanced medical data — a model can 
score 87% accuracy while catching only 17% of actual diabetic cases. This 
project benchmarks three classifiers with a focus on recall and 
interpretability, since missing a true positive (an undiagnosed diabetic) 
carries far higher real-world cost than a false alarm.

## Results

| Model | Accuracy | Diabetes Recall | Diabetes Precision |
|---|---|---|---|
| Logistic Regression (weighted) | 73% | **76%** | 31% |
| Random Forest | 86% | 17% | 48% |
| Gradient Boosting | 87% | 17% | 57% |

Logistic Regression had the lowest overall accuracy but by far the highest 
recall on the diabetic class — catching 76% of true diabetic cases versus 
just 17% for Random Forest and Gradient Boosting. In a healthcare screening 
context, this tradeoff (more false positives, far fewer missed diagnoses) 
is the right one.

Top predictive features: **HighBP** (28.7%), **general health** (27.3%), and 
**BMI** (17.0%), together accounting for ~73% of total model importance.

## Key Techniques

- Class imbalance handling via class weighting
- Model comparison: Logistic Regression, Random Forest, Gradient Boosting
- Evaluation beyond accuracy — precision, recall, F1-score, confusion matrices

## Installation

```bash
git clone [repo-url]
pip install scikit-learn pandas matplotlib
python main.py
```

Requires Python 3.10+.

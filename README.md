CreditWise Loan System



An end-to-end supervised machine learning project for predicting whether a loan application is approved or rejected.



\## Business Problem



SecureTrust Bank receives personal and home loan applications from customers across urban and rural India. Reviewing income proofs, employment details, credit history, and other documents manually can be slow and inconsistent. This project builds a binary classification model to provide a data-informed preliminary recommendation before final human verification.



The target variable is `Loan\_Approved`:



\- `1` / `Yes` - loan approved

\- `0` / `No` - loan rejected



\## Project Goal



Use historical loan-application information to help reduce two costly decision errors:



\- Rejecting customers who may be eligible for a loan

\- Approving applications that may present a higher risk



The model is intended to support a loan officer's review, not replace the final human lending decision.



\## Dataset



The notebook uses `loan\_approval\_data.csv`, containing 1,000 loan-application records. Before preprocessing, it has 20 columns, including an applicant identifier and the target.



| Category | Example fields |

| --- | --- |

| Applicant profile | `Age`, `Gender`, `Marital\_Status`, `Dependents`, `Education\_Level` |

| Employment | `Employment\_Status`, `Employer\_Category` |

| Financial information | `Applicant\_Income`, `Coapplicant\_Income`, `Savings`, `Existing\_Loans`, `DTI\_Ratio` |

| Loan information | `Loan\_Amount`, `Loan\_Term`, `Loan\_Purpose`, `Collateral\_Value`, `Property\_Area` |

| Credit and target | `Credit\_Score`, `Loan\_Approved` |



`Applicant\_ID` was removed because it is an identifier rather than a meaningful predictor.



\## Workflow



\### 1. Data preparation



\- Dropped `Applicant\_ID`

\- Filled missing numerical values with the mean

\- Filled missing categorical values with the most frequent value

\- Label-encoded `Education\_Level` and the target variable

\- One-hot encoded `Gender`, `Employer\_Category`, `Property\_Area`, `Loan\_Purpose`, `Marital\_Status`, and `Employment\_Status`



\### 2. Exploratory Data Analysis



The notebook explores:



\- Loan approval versus rejection distribution

\- Gender and education-level distributions

\- Applicant and co-applicant income distributions

\- How applicant income, age, credit score, and loan amount vary by approval outcome

\- Credit-score and applicant-income distributions by approval outcome

\- Correlations among numerical features



\### 3. Feature engineering



Two non-linear features were created:



\- `Credit\_Score\_sq` = `Credit\_Score` squared

\- `DTI\_Ratio\_sq` = `DTI\_Ratio` squared



The original `Credit\_Score` and `DTI\_Ratio` columns were then removed for the feature-engineered experiment.



\### 4. Model training



The data was split into 80% training data and 20% test data with `random\_state=42`. Features were standardised using `StandardScaler`.



The following binary classifiers were trained and compared:



\- Logistic Regression

\- K-Nearest Neighbors (KNN), with `n\_neighbors=5`

\- Gaussian Naive Bayes



\## Evaluation: Precision and Recall First



Models were evaluated using precision, recall, F1-score, accuracy, and confusion matrices. For loan approval screening, accuracy alone does not explain the type of decision error a model makes, so the main focus is precision and recall for the approved class.



\- \*\*Precision:\*\* Of the applications predicted as approved, how many were actually approved? Higher precision helps reduce incorrect approval recommendations.

\- \*\*Recall:\*\* Of the applications that were actually approved, how many did the model correctly find? Higher recall helps reduce the chance of rejecting potentially eligible applicants.

\- \*\*F1-score:\*\* Balances precision and recall.



\## Results



\### Before feature engineering



| Model | Precision | Recall | F1-score | Accuracy |

| --- | ---: | ---: | ---: | ---: |

| Logistic Regression | 78.33% | 77.05% | 77.69% | 86.50% |

| KNN | 62.75% | 52.46% | 57.14% | 76.00% |

| Gaussian Naive Bayes | 78.33% | 77.05% | 77.69% | 86.50% |



\### After feature engineering



| Model | Precision | Recall | F1-score | Accuracy |

| --- | ---: | ---: | ---: | ---: |

| Logistic Regression | 79.03% | \*\*80.33%\*\* | \*\*79.67%\*\* | 87.50% |

| KNN | 62.00% | 50.82% | 55.86% | 75.50% |

| Gaussian Naive Bayes | \*\*80.36%\*\* | 73.77% | 76.92% | 86.50% |



\*\*Best balanced model: Logistic Regression.\*\* It achieved the highest recall (80.33%) and F1-score (79.67%), correctly identifying 49 of the 61 actually approved applications in the test set.



Gaussian Naive Bayes achieved slightly higher precision (80.36% versus 79.03%), meaning it made marginally fewer incorrect approval recommendations. Logistic Regression was selected because it offered the strongest overall precision-recall balance.



Its confusion matrix was `\[\[126, 13], \[12, 49]]`, meaning it correctly classified 126 rejected and 49 approved applications. It made 13 incorrect approval predictions and missed 12 approvals.



\## Tools Used



\- Python

\- Jupyter Notebook

\- Pandas and NumPy

\- Matplotlib and Seaborn

\- Scikit-learn



\## How to Run



1\. Download or clone this repository.

2\. Keep `loan\_approval\_data.csv` in the same folder as `loan\_approval\_pred (1).ipynb`.

3\. Install the libraries used in the notebook: `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.

4\. Open the notebook in JupyterLab or Jupyter Notebook.

5\. Run the cells from top to bottom.



\## Responsible Use



Loan approval decisions can materially affect people. This project is for learning and demonstration purposes. A real deployment should include data-quality checks, fairness testing, privacy protection, regulatory review, monitoring, and final human oversight.




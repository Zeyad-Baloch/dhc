
## Task: Telco Customer Churn Pipeline (`task_2.ipynb` + `predict.py`)

**Objective:** Build a reusable, production-ready ML pipeline using Scikit-learn's `Pipeline` API to predict whether a telecom customer will churn, and export the best model with `joblib`.

**Dataset:** IBM Telco Customer Churn: 7,043 customers, 20 features covering demographics, services subscribed, contract type, and billing information.

**Models applied:** Logistic Regression and Random Forest, both tuned with `GridSearchCV`.

**Working:**
- EDA: churn class distribution , box plots comparing tenure/MonthlyCharges/TotalCharges between churners and non-churners, churn rate by contract type / internet service / payment method / TechSupport, correlation heatmap
- Cleaned `TotalCharges` blank strings (coerced to NaN, dropped 11 rows), dropped `customerID`, binary encoded target
- Built a `ColumnTransformer` preprocessor: `StandardScaler` on numerical features, `OneHotEncoder` (drop='first', handle_unknown='ignore') on categorical features
- Wrapped preprocessor + each model in a `Pipeline` so preprocessing is fit only on training data
- `GridSearchCV` with 3-fold `StratifiedKFold`, scoring=F1, tuning C/solver/class_weight for LR and n_estimators/max_depth/class_weight for RF
- Evaluated both best models on test set: accuracy, F1, ROC-AUC, confusion matrices, ROC curves, Random Forest feature importances
- Exported the higher-F1 pipeline to `churn_pipeline.pkl` with `joblib`
- Built `predict.py`: loads the pipeline in one line, runs predictions on new customer records, outputs label + churn probability + risk tier (LOW / MEDIUM / HIGH)

**Key findings:**
- Month-to-month contract customers churn at a dramatically higher rate than one-year or two-year customers, contract type is the single strongest predictor
- Short tenure strongly predicts churn; customers who leave tend to do so within the first 12 months
- Fiber optic customers churn more than DSL customers, likely reflecting pricing dissatisfaction relative to perceived value
- Customers without TechSupport churn at a higher rate, bundled services meaningfully increase retention
- F1 was chosen as the primary metric over accuracy because ~74% of customers did not churn; a naive model predicting "no churn" would score 74% accuracy while being completely useless

---


```bash
pip install scikit-learn pandas matplotlib seaborn joblib
```


Open any notebook in Jupyter and run all cells. For Streamlit apps, run `streamlit run <filename>.py` from the project directory.

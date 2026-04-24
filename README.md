## AI/ML Internship Tasks

This repo contains tasks completed during my AI/ML internship. Each task is a self-contained Jupyter notebook.

---

### Task 1 — Iris EDA (`iris_eda.ipynb`)

**Objective:** Explore the classic Iris dataset and understand how the three flower species differ across their physical measurements.

**Dataset:** Iris dataset — 150 samples, 4 features (sepal length, sepal width, petal length, petal width), 3 species (setosa, versicolor, virginica).

**What was done:**
- Loaded and inspected the dataset (shape, dtypes, summary statistics)
- Scatter plot: sepal length vs sepal width, colored by species
- Histogram with KDE: sepal length distribution per species
- Box plot: sepal length spread per species

**Key findings:**
- Setosa is clearly separable from the other two species across most measurements
- Versicolor and virginica overlap significantly when using sepal dimensions alone
- Setosa has wider sepals despite being the smallest species overall

---

### Task 3 — Heart Disease Classification (`heart_disease_pred.ipynb`)

**Objective:** Build a binary classifier to predict the presence of heart disease using clinical features from the UCI Heart Disease dataset.

**Dataset:** 920 patients across multiple hospital sites, 16 features including age, sex, chest pain type, resting blood pressure, cholesterol, and ECG results.

**Models applied:** Logistic Regression

**What was done:**
- Binarized the target: original values 1-4 (severity) collapsed to 1 (disease present)
- Handled missing values: mode imputation for categorical columns, median for numerical
- Scaled numerical features with StandardScaler
- One-hot encoded categorical features 
- Trained logistic regression and evaluated with classification report, confusion matrix, and ROC curve

**Key findings:**
- Asymptomatic chest pain (`cp_asymptomatic`) and exercise-induced angina (`exang`) were the strongest positive predictors of disease
- Maximum heart rate achieved (`thalch`) had the strongest negative correlation — lower max HR strongly associated with disease presence
- AUC of ~0.91 indicates the model has strong discriminating ability between disease and no disease

---

### Task 6 — Housing Price Prediction (`house_price_pred.ipynb`)

**Objective:** Build a linear regression model to predict house prices.

**Dataset:** 545 houses with 13 features including area, bedrooms, bathrooms, stories, and several yes/no amenity columns.

**Models applied:** Linear Regression 

**What was done:**
- Explored the dataset: distributions, correlations, outlier detection
- Binary encoded yes/no columns (mainroad, guestroom, basement, etc.)
- One-hot encoded `furnishingstatus` 
- Capped outliers at 95th percentile for both price and area
- Scaled features with StandardScaler


**Notes on results:**
- R² of 0.68 means the model explains about 68% of the variance in price.
- A more complex model (Random Forest, XGBoost) would likely push R² higher, but linear regression gives an interpretable starting point.

---

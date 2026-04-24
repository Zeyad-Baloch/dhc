# AI/ML Internship Tasks

This repo contains tasks completed during my AI/ML internship. Each task is a self-contained Jupyter notebook with its own dataset, analysis, and model.

---

## Task 1 — Iris EDA (`task_1.ipynb`)

**Objective:** Explore the classic Iris dataset and understand how the three flower species differ across their physical measurements.

**Dataset:** Iris dataset (built into seaborn) — 150 samples, 4 features (sepal length, sepal width, petal length, petal width), 3 species (setosa, versicolor, virginica).

**What was done:**
- Loaded and inspected the dataset (shape, dtypes, summary statistics)
- Scatter plot: sepal length vs sepal width, colored by species
- Histogram with KDE: sepal length distribution per species
- Box plot: sepal length spread per species

**Key findings:**
- Setosa is clearly separable from the other two species across most measurements
- Versicolor and virginica overlap significantly when using sepal dimensions alone — petal measurements would give cleaner separation
- Sepal width shows an inverse pattern: setosa has wider sepals despite being the smallest species overall

---

## Task 3 — Heart Disease Classification (`task_3.ipynb`)

**Objective:** Build a binary classifier to predict the presence of heart disease using clinical features from the UCI Heart Disease dataset.

**Dataset:** heart_disease_uci.csv — 920 patients across multiple hospital sites, 16 features including age, sex, chest pain type, resting blood pressure, cholesterol, and ECG results.

**Models applied:** Logistic Regression (scikit-learn)

**What was done:**
- Binarized the target: original values 1-4 (severity) collapsed to 1 (disease present)
- Handled missing values: mode imputation for categorical columns, median for numerical
- Dropped non-clinical columns: `id` and `dataset` (hospital site)
- Scaled numerical features with StandardScaler
- One-hot encoded categorical features (sex, cp, restecg, slope, thal)
- Split: 80/20 with stratification to preserve class balance
- Trained logistic regression and evaluated with classification report, confusion matrix, and ROC curve

**Results:**

| Metric | Value |
|--------|-------|
| Test Accuracy | ~83% |
| ROC-AUC | ~0.91 |

**Key findings:**
- Asymptomatic chest pain (`cp_asymptomatic`) and exercise-induced angina (`exang`) were the strongest positive predictors of disease
- Maximum heart rate achieved (`thalch`) had the strongest negative correlation — lower max HR strongly associated with disease presence
- AUC of ~0.91 indicates the model has strong discriminating ability between disease and no disease
- In a medical context recall matters more than precision — missing a real case is worse than a false alarm

---

## Task 6 — Housing Price Prediction (`task_6.ipynb`)

**Objective:** Build a linear regression model to predict house prices based on physical and categorical features.

**Dataset:** Housing.csv — 545 houses with 13 features including area, bedrooms, bathrooms, stories, and several yes/no amenity columns.

**Models applied:** Linear Regression (scikit-learn)

**What was done:**
- Explored the dataset: distributions, correlations, outlier detection
- Binary encoded yes/no columns (mainroad, guestroom, basement, etc.)
- One-hot encoded `furnishingstatus` (3 categories → 2 dummy columns, drop='first')
- Capped outliers at 95th percentile for both price and area
- Scaled features with StandardScaler
- Split: 80/20 train/test

**Results:**

| Metric | Value |
|--------|-------|
| MAE | ~847,000 |
| RMSE | ~1,066,000 |
| R² | 0.68 |

**Notes on results:**
- R² of 0.68 means the model explains about 68% of the variance in price — reasonable for a linear baseline on this kind of data
- The gap between MAE and RMSE suggests a few predictions are quite far off, which is expected given the price range spans ~1.75M to 13.3M
- A more complex model (Random Forest, XGBoost) would likely push R² higher, but linear regression gives an interpretable starting point

---

---

## Task — News Topic Classifier (`bert_news_classifier.ipynb` + `app.py`)

**Objective:** Fine-tune `bert-base-uncased` on the AG News dataset to classify news headlines into 4 topic categories: World, Sports, Business, and Sci/Tech.

**Dataset:** AG News (via Hugging Face Datasets) — 120,000 training samples, 7,600 test samples, 4 balanced classes. Training used a 3,000-sample subset for CPU feasibility; the same pipeline runs on the full dataset unchanged.

**Models applied:** `bert-base-uncased` (HuggingFace Transformers) — fine-tuned with the `Trainer` API.

**What was done:**
- Loaded AG News via `datasets`, shuffled and subsetted (3,000 train / 500 val / 500 test)
- EDA: class distribution bar chart, word count histogram with MAX_LEN marker, sample headlines per category
- Tokenized with `BertTokenizer`, wrapped in a custom `NewsDataset(Dataset)` PyTorch class
- Fine-tuned with `TrainingArguments`: 3 epochs, batch size 16, lr 2e-5, weight decay 0.01, best model saved by macro F1
- Evaluated on held-out test set: classification report, confusion matrix heatmap, per-epoch accuracy and F1 training curves
- Saved fine-tuned model and tokenizer to `./bert_news_model` for use by the Streamlit app
- Built `app.py`: text input → prediction label + confidence scores across all 4 categories

**Results:**

| Metric | Value |
|--------|-------|
| Test Accuracy | ~90% |
| Test F1 (macro) | ~0.90 |

**Key findings:**
- BERT reaches 90% accuracy on AG News with only 3,000 training samples — the pre-trained language representations transfer very effectively even with minimal fine-tuning data
- Sports and Business are the easiest categories to separate; their vocabulary is highly domain-specific
- World and Sci/Tech cause the most confusion — both categories regularly contain technical and geopolitical language that overlaps
- Scaling to the full 120k dataset on a GPU typically pushes accuracy to ~94-95%

**Notes:**
- The "missing keys / unexpected keys" warnings on load (`LayerNorm.gamma` vs `LayerNorm.weight`) are a known HuggingFace naming migration artifact — the model loads correctly regardless
- Run `app.py` from the same directory as `./bert_news_model` after training: `streamlit run app.py`

---

## Task — Telco Customer Churn Pipeline (`task_2.ipynb` + `predict.py`)

**Objective:** Build a reusable, production-ready ML pipeline using Scikit-learn's `Pipeline` API to predict whether a telecom customer will churn, and export the best model with `joblib`.

**Dataset:** IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) — 7,043 customers, 20 features covering demographics, services subscribed, contract type, and billing information. Available on Kaggle.

**Models applied:** Logistic Regression and Random Forest (scikit-learn), both tuned with `GridSearchCV`.

**What was done:**
- EDA: churn class distribution (bar + pie), box plots comparing tenure/MonthlyCharges/TotalCharges between churners and non-churners, churn rate by contract type / internet service / payment method / TechSupport, correlation heatmap
- Cleaned `TotalCharges` blank strings (coerced to NaN, dropped 11 rows), dropped `customerID`, binary encoded target
- Built a `ColumnTransformer` preprocessor: `StandardScaler` on numerical features, `OneHotEncoder` (drop='first', handle_unknown='ignore') on categorical features
- Wrapped preprocessor + each model in a `Pipeline` so preprocessing is fit only on training data — no leakage
- `GridSearchCV` with 3-fold `StratifiedKFold`, scoring=F1, tuning C/solver/class_weight for LR and n_estimators/max_depth/class_weight for RF
- Evaluated both best models on test set: accuracy, F1, ROC-AUC, confusion matrices, ROC curves, Random Forest feature importances
- Exported the higher-F1 pipeline to `churn_pipeline.pkl` with `joblib`
- Built `predict.py`: loads the pipeline in one line, runs predictions on new customer records, outputs label + churn probability + risk tier (LOW / MEDIUM / HIGH)

**Results:**

| Model | CV F1 | Test Accuracy | Test F1 | ROC-AUC |
|-------|-------|---------------|---------|---------|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |

> Fill in results after running — exact values depend on which hyperparameters GridSearch selects.

**Key findings:**
- Month-to-month contract customers churn at a dramatically higher rate than one-year or two-year customers — contract type is the single strongest predictor
- Short tenure strongly predicts churn; customers who leave tend to do so within the first 12 months
- Fiber optic customers churn more than DSL customers, likely reflecting pricing dissatisfaction relative to perceived value
- Customers without TechSupport churn at a higher rate — bundled services meaningfully increase retention
- F1 was chosen as the primary metric over accuracy because ~74% of customers did not churn; a naive model predicting "no churn" would score 74% accuracy while being completely useless

**Notes:**
- `predict.py` requires `churn_pipeline.pkl` to exist in the same directory — run the notebook first
- To retrain on new data, just call `pipeline.fit(X_new, y_new)` — the preprocessing steps refit automatically

---

## Task — Context-Aware RAG Chatbot (`rag_app.py`)

**Objective:** Build a conversational chatbot that retrieves answers from a vectorized knowledge base and maintains conversation history across turns.

**Knowledge base:** 15 Wikipedia articles on AI/ML concepts (Machine Learning, Deep Learning, NLP, Transformers, CNNs, RNNs, Reinforcement Learning, GANs, BERT, LLMs, RAG, Random Forest, SVM, Gradient Boosting, Overfitting) — fetched programmatically via `wikipedia-api`, no manual downloads needed.

**Stack:** Groq LLaMA3-8b · FAISS · sentence-transformers · LangChain · Streamlit

**What was done:**
- Fetched 15 Wikipedia pages at runtime and wrapped them as `Document` objects with source metadata
- Split into 500-token chunks with 50-token overlap using `RecursiveCharacterTextSplitter`
- Embedded all chunks with `sentence-transformers/all-MiniLM-L6-v2` (local, CPU, ~80MB) and indexed into FAISS
- Built a `history_aware_retriever` that rewrites the user's question using chat history before querying FAISS — so follow-up questions like "explain more" retrieve the right documents rather than searching literally
- Passed retrieved chunks + conversation history into a QA prompt answered by Groq's `llama3-8b-8192`
- Wrapped the full chain in `RunnableWithMessageHistory` so history is genuinely injected on every call
- Deployed as a Streamlit app: chat interface, collapsible source citations per answer, clear history button, API key read from environment variable

**Key findings:**
- `history_aware_retriever` is essential for multi-turn coherence — without it, follow-up questions retrieve irrelevant chunks because FAISS searches the literal follow-up string rather than the full intent
- `all-MiniLM-L6-v2` is a strong embeddings choice for CPU deployment: fast to load, small footprint, and good semantic similarity for domain-specific retrieval
- `temperature=0.3` keeps Groq's responses factual and grounded in the retrieved context rather than drifting into hallucination
- The chatbot correctly declines to answer questions outside the knowledge base rather than making things up, because the QA prompt explicitly instructs it to say "I don't know" when the context doesn't support an answer

**Notes:**
- Set `GROQ_API_KEY` as an environment variable before running — free key available at `console.groq.com`
- The vector store builds on first launch (~30 seconds) and is cached for the rest of the session
- Run with: `streamlit run rag_app.py`

---

## Setup

**Tasks 1, 3, 6 (notebooks):**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

**News Classifier:**
```bash
pip install datasets transformers torch scikit-learn streamlit
```

**Telco Churn Pipeline:**
```bash
pip install scikit-learn pandas matplotlib seaborn joblib
```

**RAG Chatbot:**
```bash
pip install streamlit langchain-classic langchain-groq langchain-core
pip install langchain-text-splitters langchain-community faiss-cpu
pip install sentence-transformers wikipedia-api
export GROQ_API_KEY=your_key_here
```

Then open any notebook in Jupyter and run all cells. For Streamlit apps, run `streamlit run <filename>.py` from the project directory.

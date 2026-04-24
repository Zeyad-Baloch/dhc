## Task: News Topic Classifier (`bert_news_classifier.ipynb` + `app.py`)

**Objective:** Fine-tune `bert-base-uncased` on the AG News dataset to classify news headlines into 4 topic categories: World, Sports, Business, and Sci/Tech.

**Dataset:** AG News: 120,000 training samples, 7,600 test samples, 4 balanced classes.

**Models applied:** `bert-base-uncased`, fine-tuned with the `Trainer` API.

**Working:**
- Loaded AG News via `datasets`, shuffled and subsetted 
- EDA: class distribution bar chart, word count histogram with MAX_LEN marker, sample headlines per category
- Tokenized with `BertTokenizer`, wrapped in a custom `NewsDataset(Dataset)` PyTorch class
- Fine-tuned with `TrainingArguments`: 3 epochs, batch size 16, lr 2e-5, weight decay 0.01, best model saved by macro F1
- Evaluated on held-out test set: classification report, confusion matrix heatmap, per-epoch accuracy and F1 training curves
- Saved fine-tuned model and tokenizer to `./bert_news_model` for use by the Streamlit app
- Built `app.py`: text input to prediction label + confidence scores across all 4 categories

**Results:**

| Metric | Value |
|--------|-------|
| Test Accuracy | ~90% |
| Test F1 (macro) | ~0.90 |

**Key findings:**
- BERT reaches 90% accuracy on AG News with only 3,000 training samples, the pre-trained language representations transfer very effectively even with minimal fine-tuning data
- Sports and Business are the easiest categories to separate; their vocabulary is highly domain-specific
- World and Sci/Tech cause the most confusion, both categories regularly contain technical and geopolitical language that overlaps

---

```bash
pip install datasets transformers torch scikit-learn streamlit
```



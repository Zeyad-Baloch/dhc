import streamlit as st
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# Constants 

MODEL_DIR  = './bert_news_model' 
MAX_LEN    = 64
LABEL_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']


LABEL_COLORS = {
    'World'    : '#4A90D9',
    'Sports'   : '#27AE60',
    'Business' : '#E67E22',
    'Sci/Tech' : '#8E44AD'
}

# Model loading 
# @st.cache_resource loads the model once and reuses it across user interactions
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model     = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()   # inference mode - disables dropout
    return tokenizer, model


# Prediction 

def predict(text, tokenizer, model):
   
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():   
        outputs = model(
            input_ids      = encoding['input_ids'],
            attention_mask = encoding['attention_mask']
        )

    logits      = outputs.logits.squeeze()
    probs       = torch.softmax(logits, dim=0).numpy()   
    pred_idx    = int(np.argmax(probs))
    pred_label  = LABEL_NAMES[pred_idx]
    confidence  = float(probs[pred_idx])

    return pred_label, confidence, probs


#  Page config 

st.set_page_config(
    page_title = 'News Topic Classifier',
    layout     = 'centered'
)


st.title(' News Topic Classifier')
st.markdown(
    'Paste a news headline or short article excerpt below. '
    'The model will predict which of the 4 AG News categories it belongs to.'
)
st.markdown('---')

# Load model
with st.spinner('Loading model...'):
    tokenizer, model = load_model()

# Text input
user_input = st.text_area(
    label       = 'Enter news text',
    placeholder = 'e.g. Scientists discover a new exoplanet in the habitable zone...',
    height      = 120
)


if st.button('Classify', type='primary'):
    if not user_input.strip():
        st.warning('Please enter some text first.')
    else:
        with st.spinner('Classifying...'):
            label, confidence, probs = predict(user_input.strip(), tokenizer, model)

        # Result card 
        color = LABEL_COLORS[label]
        st.markdown(
            f"""
            <div style="
                background-color: {color}22;
                border-left: 5px solid {color};
                padding: 16px 20px;
                border-radius: 6px;
                margin-bottom: 20px;
            ">
                <h3 style="margin:0; color:{color};">
                    {label}
                </h3>
                <p style="margin:4px 0 0 0; color:#555;">
                    Confidence: <strong>{confidence*100:.1f}%</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence bar chart 
        st.markdown('**Confidence scores across all categories:**')

        for i, name in enumerate(LABEL_NAMES):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f'**{name}**')
            with col2:
                # Using st.progress for a clean visual bar
                st.progress(float(probs[i]))
                st.caption(f'{probs[i]*100:.1f}%')

#  Sidebar info 
with st.sidebar:
    st.header('About')
    st.markdown(
        '**Model:** `bert-base-uncased`  \n'
        '**Fine-tuned on:** AG News (3,000 samples)  \n'
        '**Categories:** World · Sports · Business · Sci/Tech  \n'
        '**Max input length:** 64 tokens'
    )
    st.markdown('---')
    st.markdown('**Example headlines to try:**')
    examples = [
        'NASA announces plans to return astronauts to the Moon by 2026',
        'Manchester United beats Arsenal 3-1 in Premier League clash',
        'Federal Reserve raises interest rates by 25 basis points',
        'Tensions rise as North Korea conducts missile test over Japan'
    ]
    for ex in examples:
        st.markdown(f'- _{ex}_')

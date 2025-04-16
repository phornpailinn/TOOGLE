
import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, MBart50Tokenizer, MBartForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objs as go
import langid
import uuid
from datetime import datetime

# Load the summarization tokenizer and model
summarization_model_name = 'facebook/bart-large-cnn'
summarization_tokenizer = BartTokenizer.from_pretrained(summarization_model_name)
summarization_model = BartForConditionalGeneration.from_pretrained(summarization_model_name)

# Initialize the translation model and tokenizer
translation_model_name = 'facebook/mbart-large-50-many-to-many-mmt'
translation_tokenizer = MBart50Tokenizer.from_pretrained(translation_model_name)
translation_model = MBartForConditionalGeneration.from_pretrained(translation_model_name)

# Load the classification tokenizer and model
classification_model_name = 'cardiffnlp/tweet-topic-21-multi'
classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_name)
classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_name)

# Define language options for translation
languages = {
    "en_XX": "English",
    "es_XX": "Spanish",
    "fr_XX": "French",
    "nl_XX": "Dutch",
    "pt_XX": "Portuguese",
    "ja_XX": "Japanese",
    "tl_XX": "Tagalog"
}

topic_names = {
    0: "arts_&_culture",
    1: "business_&_entrepreneurs",
    2: "celebrity_&_pop_culture",
    3: "diaries_&_daily_life",
    4: "family",
    5: "fashion_&_style",
    6: "film_tv_&_video",
    7: "fitness_&_health",
    8: "food_&_dining",
    9: "gaming",
    10: "learning_&_educational",
    11: "music",
    12: "news_&_social_concern",
    13: "other_hobbies",
    14: "relationships",
    15: "science_&_technology",
    16: "sports",
    17: "travel_&_outdoors"
}

# Streamlit app
st.title("NLP Application with Streamlit")

# Summarization
st.header("Summarization")
summarization_input = st.text_area("Enter text to summarize")
if st.button("Summarize"):
    inputs = summarization_tokenizer(summarization_input, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    st.write("Summary:", summary)

# Translation
st.header("Translation")
translation_input = st.text_area("Enter text to translate")
target_language = st.selectbox("Select target language", list(languages.keys()), format_func=lambda x: languages[x])
if st.button("Translate"):
    inputs = translation_tokenizer(translation_input, return_tensors="pt")
    translated_ids = translation_model.generate(inputs["input_ids"], forced_bos_token_id=translation_tokenizer.lang_code_to_id[target_language])
    translated_text = translation_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    st.write("Translated Text:", translated_text)

# Classification
st.header("Classification")
classification_input = st.text_area("Enter text to classify")
if st.button("Classify"):
    inputs = classification_tokenizer(classification_input, return_tensors="pt")
    outputs = classification_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_prob, top_label = torch.topk(probs, 1)
    st.write("Predicted Topic:", topic_names[top_label.item()])

# Plotting (example code, adjust based on your needs)
st.header("Plotting Example")
if st.button("Show Plot"):
    fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    st.plotly_chart(fig)

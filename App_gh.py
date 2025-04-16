import torch
import plotly.graph_objs as go
from transformers import BartTokenizer, BartForConditionalGeneration, MBart50Tokenizer, MBartForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
import dash_bootstrap_components as dbc
from dash import dash, dcc, html
from dash.dependencies import Input, Output, State
import langid
import psycopg2
from psycopg2 import sql
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
    17: "travel_&_adventure",
    18: "youth_&_student_life"
}

"""
# Define your PostgreSQL connection parameters
dbname = "database"
user = "username"
password = "password"
host = "localhost"
port = "5432"

# Connect to your PostgreSQL database
conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)

# Function to insert data into PostgreSQL database
def insert_data(id, timestamp, max_prob_topic, max_prob_value):
    cursor = conn.cursor()
    insert_query = sql.SQL("INSERT INTO results (id, timestamp, topic, probability) VALUES (%s, %s, %s, %s)")
    cursor.execute(insert_query, (id, timestamp, max_prob_topic, max_prob_value))
    conn.commit()
    cursor.close()"""

# Function to generate UUID as primary key
def generate_id():
    return uuid.uuid4().hex

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=["assets/styles.css"], title="Toogle")

# Define the app layout
app.layout = html.Div([
    dbc.Navbar(className="top", color="dark", dark=True, children=[
        dbc.Container(style={'display': 'flex', 'align-items': 'center'}, children=[
            html.Img(src="https://raw.githubusercontent.com/bubblebolt/dads/main/DADS5001/ASM5-LLM/Pics/Toogle.png", height="110px"),
            html.H1("Text Analysis : Summarization, Translation and Classification Topic", style={'color': 'white', 'margin-left': '20px'}),
        ]),
    ]),
    html.Div([
        html.Div([
            html.H3('Input language:'),
            dcc.Dropdown(
                id='input-language-dropdown',
                options=[{'label': lang_name, 'value': lang_code} for lang_code, lang_name in languages.items()],
                value='en_XX',
                style={'width': '100%'}
            ),
        ], style={'width': '49%', 'display': 'inline-block','margin-top': '120px'}),
        html.Div([
            html.H3('Target language:'),
            dcc.Dropdown(
                id='output-language-dropdown',
                options=[{'label': lang_name, 'value': lang_code} for lang_code, lang_name in languages.items()],
                value='en_XX',
                style={'width': '100%'}
            ),
        ], style={'width': '49%', 'display': 'inline-block', 'float': 'right','margin-top': '120px'}),
    ]),
    html.Div([
        html.H3('Enter text to summarize :'), 
        dcc.Textarea(className="dash-textarea-container",
            id='input-text',
            placeholder='Enter text to summarize ...',
            style={'width': '100%', 'height': 200},
        ),
        html.Div(id='char-count')  # เพิ่มตัวนับความยาว character ที่นี่
    ], style={'margin-top': '20px'}),
    html.Div([
        html.H3('Summary length (min and max):'), 
        dcc.RangeSlider(className="dash-rangeslider",
            id='summary-length-slider',
            min=20,
            max=200,
            step=10,
            value=[50, 100],
            marks={i: str(i) for i in range(20, 201, 10)}
        ),
    ], style={'margin-top': '20px'}),  

    html.Div([
        html.Button('Compute', id='submit-val', n_clicks=0), 
    ], style={'margin-top': '20px'}),  
    html.Div(id='output-text', style={'margin-top': '20px'}),
    html.Footer(className="footer", children=[
    html.Div([
        html.P("THIS PROJECT IS AFFILIATED WITH DADS5001"),
        html.P("Copyright © 2024")
    ], style={'text-align': 'center', 'color': 'white', 'font-size': '14px', 'margin-top': '20px'}),
])])
    

# Define the callback to summarize text, translate to English, and classify
@app.callback(
    Output('output-text', 'children'),
    [Input('submit-val', 'n_clicks')],
    [State('input-text', 'value'),
     State('input-language-dropdown', 'value'),
     State('output-language-dropdown', 'value'),
     State('summary-length-slider', 'value')]
)
def update_output(n_clicks, input_text, input_language, output_language, summary_length):
    if n_clicks > 0:
        # Detect input language if it's not English
        if input_language != 'en_XX':
            # Translate input text to English
            translated_input = translation_tokenizer.decode(
                translation_model.generate(
                    translation_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=1024, src_lang=input_language)['input_ids'],
                    forced_bos_token_id=translation_tokenizer.lang_code_to_id['en_XX']
                )[0],
                skip_special_tokens=True
            )
        else:
            translated_input = input_text

        # Generate summary
        summary_ids = summarization_model.generate(
            summarization_tokenizer(translated_input, return_tensors='pt', max_length=1024, truncation=True)['input_ids'],
            num_beams=4,
            min_length=summary_length[0],
            max_length=summary_length[1],
            early_stopping=True
        )

        # Decode summary
        summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Translate summary to target language if necessary
        if output_language != 'en_XX':
            translated_summary = translation_tokenizer.decode(
                translation_model.generate(
                    translation_tokenizer(summary, return_tensors='pt', padding=True, truncation=True, max_length=1024, src_lang='en_XX')['input_ids'],
                    forced_bos_token_id=translation_tokenizer.lang_code_to_id[output_language]
                )[0],
                skip_special_tokens=True
            )
        else:
            translated_summary = summary

        # Classify translated text
        classification_outputs = classification_model(
            **classification_tokenizer(
                translated_input if input_language != 'en_XX' else translated_summary,
                return_tensors='pt',
                max_length=512,
                truncation=True
            )
        )

        logits = classification_outputs.logits[0]
        probabilities = torch.softmax(logits, dim=0)

        labels = classification_tokenizer.convert_ids_to_tokens(range(logits.size(0)))
        label_probabilities = {topic_names[i]: probabilities[i].item() for i in range(logits.size(0))}

        sorted_labels = sorted(label_probabilities.items(), key=lambda x: x[1], reverse=False)

        labels = [f"{label}: {prob:.3f}" for label, prob in sorted_labels]
        probabilities = [prob for _, prob in sorted_labels]

        # Create horizontal bar chart
        fig = go.Figure(data=[go.Bar(y=labels, x=probabilities, orientation='h', marker=dict(color='#4285f4'))])

        fig.update_layout(
            title={'text': "Topics and Probabilities", 'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top', 'font': {'color': 'black'}},
            xaxis_title="Probabilities",
            plot_bgcolor='lightgray',  # สีพื้นหลัง
        )

        # Generate new UUID as id
        id = generate_id()
        # Insert data into PostgreSQL database
        timestamp = datetime.now()  # Or provide the actual timestamp if needed
        max_prob_topic = sorted_labels[-1][0]
        max_prob_value = sorted_labels[-1][1]

        #insert_data(id, timestamp, max_prob_topic, max_prob_value)

        return html.Div([
            html.H3(f"Translated Summary ({languages[input_language]} to {languages[output_language]}):"),
            html.P(translated_summary),
            dcc.Graph(id='classification-graph', figure=fig)
        ], style={'margin-top': '20px'})

    return ''

# Callback สำหรับตัวนับความยาว character
@app.callback(
    Output('char-count', 'children'),
    [Input('input-text', 'value')]
)
def update_char_count(input_text):
    if input_text:
        char_count = len(input_text)
        return f"Input text length: {char_count}"
    else:
        return "Input text length: 0"

if __name__ == '__main__':
    app.run_server(debug=True)

import os
import io
import base64
import re
import csv
import nltk
import torch  # Untuk deteksi GPU
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources (jika belum)
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------
# 1. DETEKSI GPU & INISIALISASI MODEL
# -----------------------
# Tentukan device: gunakan GPU jika tersedia, selainnya CPU
device = 0 if torch.cuda.is_available() else -1  # 0 = GPU pertama, -1 = CPU

# Gunakan model sentiment analysis bahasa Inggris
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Buat pipeline dengan device
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# Inisialisasi Porter Stemmer untuk Bahasa Inggris
porter_stemmer = PorterStemmer()

# -----------------------
# Fungsi Preprocessing
# -----------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)  # Menghapus URL
    text = re.sub(r'@\w+', '', text)     # Menghapus mention
    text = re.sub(r'#\w+', '', text)     # Menghapus hashtag
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus karakter khusus
    return text

def case_folding(text):
    return text.lower()

def tokenizing(text):
    return word_tokenize(text)

def filtering(tokens):
    eng_stopwords = set(stopwords.words('english'))
    return [word for word in tokens if word not in eng_stopwords and len(word) > 1]

def stemming(tokens):
    return [porter_stemmer.stem(word) for word in tokens]

def tokens_to_sentence(tokens):
    return ' '.join(tokens)

def preprocess_text(text):
    # Return empty string if text is None or not a string
    if text is None or not isinstance(text, str):
        return ""
    
    cleaned_text = clean_text(text)
    lowercased_text = case_folding(cleaned_text)
    tokens = tokenizing(lowercased_text)
    filtered_tokens = filtering(tokens)
    stemmed_tokens = stemming(filtered_tokens)
    processed_text = tokens_to_sentence(stemmed_tokens)
    return processed_text

# Mapping label model -> {Positive, Negative, Neutral}
def map_label_to_sentiment(label, score):
    """
    Map original model label to sentiment category and add Neutral category
    based on confidence score.
    
    Args:
        label: The original model label (POSITIVE or NEGATIVE)
        score: The confidence score from the model
    
    Returns:
        String: 'Positive', 'Negative', or 'Neutral'
    """
    # Threshold untuk menentukan sentimen netral
    NEUTRAL_THRESHOLD = 0.65
    
    if score < NEUTRAL_THRESHOLD:
        return 'Neutral'
    
    label = label.lower()
    if 'positive' in label:
        return 'Positive'
    elif 'negative' in label:
        return 'Negative'
    else:
        return 'Neutral'

# -----------------------
# 2. BATCH INFERENCE DENGAN BATCH SIZE
# -----------------------
def analyze_sentiment_batch(processed_texts):
    """
    Memanggil pipeline sentiment secara batch.
    Menambahkan kategori Neutral berdasarkan confidence score.
    """
    results = sentiment_analyzer(processed_texts, batch_size=32, truncation=True)
    
    sentiments = []
    scores = []
    for r in results:
        raw_label = r['label']
        score = r['score']
        model_label = map_label_to_sentiment(raw_label, score)
        sentiments.append(model_label)
        scores.append(score)
    return sentiments, scores

# Modified function to analyze sentiment after filtering empty texts
def analyze_with_filtered_data(df, text_column):
    # Make a copy of the original text
    df['original_text'] = df[text_column]
    
    # Apply preprocessing
    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    # Filter out rows with empty processed text
    df_filtered = df[df['processed_text'].str.strip() != ""].copy()
    
    # Only analyze non-empty texts
    if len(df_filtered) > 0:
        processed_texts = df_filtered['processed_text'].tolist()
        sentiments, scores = analyze_sentiment_batch(processed_texts)
        
        df_filtered['sentiment'] = sentiments
        df_filtered['confidence'] = scores
    else:
        # Create empty columns if all rows were filtered out
        df_filtered['sentiment'] = []
        df_filtered['confidence'] = []
    
    return df_filtered

def create_sentiment_chart(df):
    # Obtener conteo de sentimientos
    sentiment_counts = df['sentiment'].value_counts()
    total_count = len(df)
    
    # Calcular porcentajes
    sentiment_percentages = (sentiment_counts / total_count * 100).round(1)
    
    # Crear una figura con dos subplots (bar y pie)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Colores para los diferentes sentimientos
    colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
    bar_colors = [colors.get(sent, 'blue') for sent in sentiment_counts.index]
    
    # Gráfico de barras (subplot 1)
    bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors)
    
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Count')
    ax1.set_title('Sentiment Distribution (Count)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Añadir etiquetas con el número en cada barra
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Gráfico circular con porcentajes (subplot 2)
    pie_labels = [f'{label} ({percentage}%)' for label, percentage in zip(sentiment_counts.index, sentiment_percentages)]
    ax2.pie(sentiment_counts.values, labels=pie_labels, colors=bar_colors, autopct='%1.1f%%', 
            startangle=90, shadow=True, explode=[0.05]*len(sentiment_counts))
    ax2.set_title('Sentiment Distribution (Percentage)')
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Guardar la figura en un buffer
    buffer = io.BytesIO()
    plt.tight_layout()
    FigureCanvas(fig).print_png(buffer)
    chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return chart_data

def detect_csv_dialect(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        sample = file.read(4096)
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample)
        return dialect
    except:
        return csv.excel_tab

def get_text_column(df):
    possible_names = ['full_text', 'text', 'content', 'tweet', 'message']
    for name in possible_names:
        if name in df.columns:
            return name
    for col in df.columns:
        if df[col].dtype == 'object':
            return col
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        
        if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            if file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(filepath, sep=None, engine='python', encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(filepath, sep=None, engine='python', encoding='ISO-8859-1')
            else:
                df = pd.read_excel(filepath)
            
            text_column = get_text_column(df)
            if text_column is None:
                column_list = ', '.join(df.columns.tolist())
                return render_template('index.html', 
                                      error=f"Could not detect a suitable text column. Available columns: {column_list}",
                                      columns=df.columns.tolist())
            
            # Use the new function to filter empty texts and process data
            display_df = analyze_with_filtered_data(df, text_column)
            
            # Check if we have any data after filtering
            if len(display_df) == 0:
                return render_template('index.html', 
                                      error="All rows were filtered out because they contained empty text",
                                      columns=df.columns.tolist())
            
            chart_data = create_sentiment_chart(display_df)
            
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
            if result_path.endswith('.csv'):
                display_df.to_csv(result_path, index=False)
            else:
                display_df.to_excel(result_path, index=False)
            
            return render_template('result.html', 
                                   data=display_df.to_dict('records'),
                                   chart_data=chart_data,
                                   filename=file.filename)
    
    return render_template('index.html')

@app.route('/analyze_with_column', methods=['POST'])
def analyze_with_column():
    if 'filename' not in request.form or 'column' not in request.form:
        return redirect(url_for('index'))
    
    filename = request.form['filename']
    selected_column = request.form['column']
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return render_template('index.html', error="File not found")
    
    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(filepath, sep=None, engine='python', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, sep=None, engine='python', encoding='ISO-8859-1')
    else:
        df = pd.read_excel(filepath)
    
    if selected_column not in df.columns:
        return render_template('index.html', error=f"Column '{selected_column}' not found")
    
    # Use the new function to filter empty texts and process data
    display_df = analyze_with_filtered_data(df, selected_column)
    
    # Check if we have any data after filtering
    if len(display_df) == 0:
        return render_template('index.html', 
                              error="All rows were filtered out because they contained empty text",
                              columns=df.columns.tolist())
    
    chart_data = create_sentiment_chart(display_df)
    
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
    if result_path.endswith('.csv'):
        display_df.to_csv(result_path, index=False)
    else:
        display_df.to_excel(result_path, index=False)
    
    return render_template('result.html', 
                           data=display_df.to_dict('records'),
                           chart_data=chart_data,
                           filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
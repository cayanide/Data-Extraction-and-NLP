import os
import pandas as pd
import requests
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

# Flask app setup
app = Flask(__name__)

# Folder for file uploads
UPLOAD_FOLDER = 'static/uploads'
ARTICLES_FOLDER = 'Articles'
OUTPUT_FOLDER = 'static/output'  # Folder to store the generated output files
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ARTICLES_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Allowed extensions for the uploaded file
ALLOWED_EXTENSIONS = {'xls', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load positive and negative word dictionaries
def load_dictionaries():
    with open("MasterDictionary/positive-words.txt", encoding='utf-8') as f:
        positive_words = set(f.read().split())
    with open("MasterDictionary/negative-words.txt", encoding='utf-8') as f:
        negative_words = set(f.read().split())

    # Ensure all words in dictionaries are lowercase
    positive_words = {word.lower() for word in positive_words}
    negative_words = {word.lower() for word in negative_words}

    return positive_words, negative_words

positive_words, negative_words = load_dictionaries()

# Load stop words
def load_stopwords():
    stopwords = set()
    for file in os.listdir("StopWords"):
        with open(os.path.join("StopWords", file), encoding='utf-8', errors='replace') as f:
            stopwords.update(f.read().split())
    return stopwords

stopwords = load_stopwords()

# Function to clean text
def clean_text(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum() and word not in stopwords]

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extract content from URLs
def extract_articles(input_file):
    data = pd.read_excel(input_file)

    # Check if all articles are already in the ARTICLES_FOLDER
    existing_files = {f.replace('.txt', '') for f in os.listdir(ARTICLES_FOLDER)}
    missing_urls = []

    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Extracting Articles"):
        url_id, url = row['URL_ID'], row['URL']
        if str(url_id) not in existing_files:  # Only scrape if the file doesn't exist
            missing_urls.append(url_id)
        else:
            print(f"Article with URL_ID {url_id} already exists in the folder. Skipping scraping.")

    if missing_urls:
        print(f"Scraping the following URLs: {missing_urls}")
        for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Extracting Articles"):
            url_id, url = row['URL_ID'], row['URL']
            if str(url_id) in missing_urls:
                try:
                    response = requests.get(url)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    title = soup.find('title').get_text()
                    content = ' '.join([p.get_text() for p in soup.find_all('p')])

                    # Save to text file inside ARTICLES_FOLDER
                    file_path = os.path.join(ARTICLES_FOLDER, f'{url_id}.txt')
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(title + '\n' + content)
                except Exception as e:
                    print(f"Error processing {url}: {e}")
    else:
        print("All articles are already in the folder. Skipping scraping.")

# Perform text analysis with VADER for sentiment analysis
def analyze_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Clean text
    cleaned_text = clean_text(text)

    # Use VADER for sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)  # Analyze the entire text
    positive_score = sentiment['pos']
    negative_score = sentiment['neg']
    polarity_score = sentiment['compound']
    subjectivity_score = (positive_score + negative_score) / (len(cleaned_text) + 0.000001)  # Subjectivity: ratio of positive/negative words

    # Additional metrics
    word_count = len(cleaned_text)
    sentence_count = len(sent_tokenize(text))
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    complex_word_count = sum(1 for word in cleaned_text if textstat.syllable_count(word) > 2)
    percentage_complex_words = (complex_word_count / word_count) * 100 if word_count > 0 else 0
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

    syllables_per_word = sum(textstat.syllable_count(word) for word in cleaned_text) / word_count if word_count > 0 else 0
    avg_word_length = sum(len(word) for word in cleaned_text) / word_count if word_count > 0 else 0

    personal_pronouns = len(re.findall(r"\b(I|we|my|ours|us)\b", text, re.I))

    return {
        'POSITIVE SCORE': positive_score,
        'NEGATIVE SCORE': negative_score,
        'POLARITY SCORE': polarity_score,
        'SUBJECTIVITY SCORE': subjectivity_score,
        'AVG SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words,
        'FOG INDEX': fog_index,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': word_count,
        'SYLLABLE PER WORD': syllables_per_word,
        'PERSONAL PRONOUNS': personal_pronouns,
        'AVG WORD LENGTH': avg_word_length
    }

# Generate results and save to Excel
def generate_results(input_file, output_file):
    data = pd.read_excel(input_file)
    results = []

    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Analyzing Texts"):
        url_id = row['URL_ID']
        file_path = os.path.join(ARTICLES_FOLDER, f'{url_id}.txt')
        if os.path.exists(file_path):
            metrics = analyze_text(file_path)
            metrics['URL_ID'] = url_id
            metrics['URL'] = row['URL']
            results.append(metrics)

    df = pd.DataFrame(results)

    # Ensure to save the file with openpyxl to avoid issues
    df.to_excel(output_file, index=False, engine='openpyxl')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Step 1: Extract articles (if not already done)
        extract_articles(filepath)

        # Step 2: Process the extracted data and generate results
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], 'output.xlsx')
        generate_results(filepath, output_file)

        return jsonify({
            "download_link": "/download"
        })

    return jsonify({"error": "Invalid file format"}), 400

@app.route('/download', methods=['GET'])
def download_file():
    try:
        # Path to the output file
        output_file = 'output.xlsx'

        # Check if the output file exists
        if not os.path.exists(os.path.join(app.config['OUTPUT_FOLDER'], output_file)):
            return jsonify({"error": "Output file does not exist."}), 404

        # Send the file for download
        return send_from_directory(
            directory=app.config['OUTPUT_FOLDER'],
            path=output_file,
            as_attachment=True
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
